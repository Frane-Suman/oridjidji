/* Oriđiđi presentation state. Classification facts and Croatian copy arrive from the server.
 * Complexity justification: this file exceeds 300 lines because it holds the one build-free
 * Alpine controller shared by analysis, streaming, history and detail pages. Detector facts and
 * rendering templates remain server-owned, so splitting the small state methods would add a
 * module loader without separating any domain responsibility. */

function initialiseHeaderLogo() {
  const logo = document.querySelector("[data-logo-animated]");
  if (!logo) return;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let played = false;
  try { played = window.sessionStorage.getItem("oridjidji-logo-played") === "1"; } catch (_) { played = true; }
  if (reduceMotion || played) return;
  logo.src = logo.dataset.logoAnimated;
  try { window.sessionStorage.setItem("oridjidji-logo-played", "1"); } catch (_) { /* storage may be blocked */ }
  window.setTimeout(() => { logo.src = logo.dataset.logoStatic; }, 5000);
}

function initialiseStickyHeader() {
  const header = document.querySelector("[data-sticky-header]");
  if (!header) return;
  let framePending = false;
  const update = () => {
    header.classList.toggle("masthead--scrolled", window.scrollY > 0);
    framePending = false;
  };
  const onScroll = () => {
    if (framePending) return;
    framePending = true;
    window.requestAnimationFrame(update);
  };
  update();
  window.addEventListener("scroll", onScroll, { passive: true });
}

function initialiseHeroParallax() {
  const hero = document.querySelector(".hero");
  if (!hero || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  let framePending = false;
  const update = () => {
    const shift = Math.min(window.scrollY * 0.16, 64);
    hero.style.setProperty("--hero-parallax", `${shift}px`);
    framePending = false;
  };
  const onScroll = () => {
    if (framePending) return;
    framePending = true;
    window.requestAnimationFrame(update);
  };
  update();
  window.addEventListener("scroll", onScroll, { passive: true });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    initialiseStickyHeader();
    initialiseHeroParallax();
  }, { once: true });
} else {
  initialiseStickyHeader();
  initialiseHeroParallax();
}

function detailFromResponse(response, fallback) {
  return response.json().then((payload) => payload.detail || fallback).catch(() => fallback);
}

function oridjidji(config) {
  return {
    config,
    text: "",
    mode: config.defaultMode,
    domainHint: "",
    storeText: false,
    state: "idle",
    running: false,
    status: config.preparing,
    announcement: "",
    errorMessage: "",
    elapsedMs: 0,
    source: null,
    latestResponse: null,
    copyStatus: "",
    showTokenNumbers: false,
    agentStates: {},
    _tick: null,
    _announceTimer: null,
    _queuedAnnouncement: "",

    init() {
      let saved = null;
      try { saved = window.localStorage.getItem("oridjidji-mode"); } catch (_) { saved = null; }
      if (saved && Object.prototype.hasOwnProperty.call(config.agentsByMode, saved)) this.mode = saved;
      initialiseHeaderLogo();
    },

    saveMode() {
      try { window.localStorage.setItem("oridjidji-mode", this.mode); } catch (_) { /* optional */ }
    },

    words() {
      const trimmed = this.text.trim();
      return trimmed ? trimmed.split(/\s+/u).length : 0;
    },

    canAnalyse() {
      const count = this.words();
      return count >= config.minWords && count <= config.maxWords;
    },

    wordCountLabel() {
      return config.wordCount.replace("{n_words}", this.words()).replace("{min_words}", config.minWords);
    },

    wordGateMessage() {
      const count = this.words();
      if (count < config.minWords) return config.remaining.replace("{remaining}", config.minWords - count);
      if (count > config.maxWords) return config.overLimit.replace("{excess}", count - config.maxWords);
      return config.ready;
    },

    submitReason() {
      const count = this.words();
      if (count === 0) return config.disabled.empty;
      if (count < config.minWords) return config.disabled.short;
      if (count > config.maxWords) return config.disabled.long;
      return "";
    },

    activeAgents() { return config.agentsByMode[this.mode] || []; },
    agentState(name) { return this.agentStates[name] || "pending"; },
    agentStateLabel(name) { return config.agentStates[this.agentState(name)] || ""; },

    clear() {
      this.stop();
      this.text = "";
      this.storeText = false;
      this.state = "idle";
      this.errorMessage = "";
      this.announcement = "";
      this.latestResponse = null;
      this.agentStates = {};
      const target = document.querySelector("[data-result]");
      if (target) target.replaceChildren();
      this.$nextTick(() => document.getElementById("tekst")?.focus());
    },

    backToEditor() {
      this.clear();
    },

    async analyse() {
      const count = this.words();
      if (count === 0) return this.fail(config.errors.noText);
      if (count < config.minWords) return this.fail(config.errors.tooShort);
      if (count > config.maxWords) return this.fail(config.errors.tooLong);

      this.stop();
      this.state = "running";
      this.running = true;
      this.status = config.preparing;
      this.announce(config.preparing, true);
      this.errorMessage = "";
      this.copyStatus = "";
      this.latestResponse = null;
      this.agentStates = Object.fromEntries(this.activeAgents().map((name) => [name, "pending"]));
      this.startClock();

      const requestBody = { text: this.text, mode: this.mode, store: Boolean(this.storeText) };
      if (this.domainHint) requestBody.domain_hint = this.domainHint;
      this.storeText = false;
      const controller = new AbortController();
      this.source = { close: () => controller.abort() };
      try {
        const response = await fetch("/api/v1/analyze/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(await this.httpError(response));
        if (!response.body) throw new Error(config.errors.dropped);
        await this.readStream(response.body);
        if (this.state === "running") this.fail(config.errors.dropped);
      } catch (error) {
        if (error.name !== "AbortError" && this.state === "running") {
          this.fail(error.message || config.errors.dropped);
        }
      } finally {
        this.stop();
      }
    },

    async httpError(response) {
      if (response.status === 413) return config.errors.tooLarge;
      if (response.status === 429) return config.errors.rateLimited;
      if (response.status === 422) return detailFromResponse(response, config.errors.invalid);
      return detailFromResponse(response, config.errors.failed);
    },

    async readStream(body) {
      const reader = body.getReader();
      const decoder = new TextDecoder();
      let buffered = "";
      while (true) {
        const { value, done } = await reader.read();
        buffered += decoder.decode(value || new Uint8Array(), { stream: !done });
        let boundary = buffered.indexOf("\n\n");
        while (boundary >= 0) {
          const block = buffered.slice(0, boundary);
          buffered = buffered.slice(boundary + 2);
          let name = "message";
          let data = "";
          block.split("\n").forEach((line) => {
            if (line.startsWith("event:")) name = line.slice(6).trim();
            if (line.startsWith("data:")) data += line.slice(5).trim();
          });
          if (data) this.handleStreamEvent(name, JSON.parse(data));
          boundary = buffered.indexOf("\n\n");
        }
        if (done) return;
      }
    },

    handleStreamEvent(name, payload) {
      if (payload.step_hr) {
        this.status = payload.step_hr;
        this.announce(payload.step_hr);
      }
      if (name === "agent_started") this.setAgentState(payload.agent, "started");
      if (["agent", "agent_finished"].includes(name)) this.setAgentState(payload.agent, "finished");
      if (name === "agent_failed") this.setAgentState(payload.agent, "failed");
      if (name === "error") this.fail(payload.step_hr || config.errors.failed);
      if (name === "result") this.render(payload);
    },

    setAgentState(agent, state) {
      if (!agent) return;
      this.agentStates = { ...this.agentStates, [agent]: state };
      const band = document.querySelector(`.ring-band[data-agent="${CSS.escape(agent)}"]`);
      if (!band || state !== "finished") return;
      band.querySelectorAll(".ring-dot").forEach((dot) => {
        dot.setAttribute("r", "4");
        dot.setAttribute("fill-opacity", "1");
      });
    },

    render(payload) {
      const response = payload.response || payload;
      this.latestResponse = response;
      const target = document.querySelector("[data-result]");
      if (target) {
        target.dataset.runId = response.run_id;
        if (payload.html) target.innerHTML = payload.html;
      }
      this.state = "done";
      this.announce(response.headline_hr || this.status, true);
      this.$nextTick(() => {
        /* x-show must first remove display:none. Browsers refuse to focus an element
         * while any ancestor is hidden, so the frame boundary is part of the focus
         * contract rather than a visual animation. */
        window.requestAnimationFrame(() => {
          const heading = target?.querySelector("[tabindex='-1']");
          heading?.focus({ preventScroll: true });
          if (window.matchMedia("(max-width: 1023px)").matches) target?.scrollIntoView({ block: "start" });
        });
      });
      window.dispatchEvent(new CustomEvent("oridjidji:result", { detail: response }));
    },

    async copyReport() {
      const report = this.latestResponse?.copy_report_hr;
      if (!report) { this.copyStatus = config.copyFailed; return; }
      try { await navigator.clipboard.writeText(report); this.copyStatus = config.copied; }
      catch (_) { this.copyStatus = config.copyFailed; }
    },

    fail(message) {
      this.state = "error";
      this.errorMessage = message;
      this.announce(message, true);
      this.stop();
    },

    announce(message, immediate = false) {
      this._queuedAnnouncement = message;
      if (immediate) {
        if (this._announceTimer) window.clearTimeout(this._announceTimer);
        this._announceTimer = null;
        this.announcement = message;
        return;
      }
      if (this._announceTimer) return;
      this._announceTimer = window.setTimeout(() => {
        this.announcement = this._queuedAnnouncement;
        this._announceTimer = null;
      }, 500);
    },

    startClock() {
      const began = Date.now();
      this._tick = window.setInterval(() => { this.elapsedMs = Date.now() - began; }, 1000);
    },

    elapsedLabel() {
      const total = Math.floor(this.elapsedMs / 1000);
      return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
    },

    stop() {
      this.running = false;
      if (this.source) { this.source.close(); this.source = null; }
      if (this._tick) { window.clearInterval(this._tick); this._tick = null; }
      if (this._announceTimer) { window.clearTimeout(this._announceTimer); this._announceTimer = null; }
    },
  };
}

function historyHelpers(config) {
  return {
    config,
    label(mapping, value) { return mapping?.[value ?? ""] || value || config.unknown; },
    formatDate(value) {
      if (!value) return config.unknown;
      return new Intl.DateTimeFormat("hr-HR", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
    },
  };
}

function reviewProgress(status) {
  return {
    status,
    activeAgents() { return []; },
    agentState() { return "pending"; },
    agentStateLabel() { return ""; },
    elapsedLabel() { return "0:00"; },
  };
}

function historyPage(config) {
  return {
    ...historyHelpers(config),
    filters: { mode: "", verdict: "", domain_hint: "", status: "" },
    rows: [],
    loading: false,
    errorMessage: "",
    query() {
      const params = new URLSearchParams();
      Object.entries(this.filters).forEach(([key, value]) => { if (value) params.set(key, value); });
      return params.toString();
    },
    exportUrl() { const query = this.query(); return `/api/v1/runs.csv${query ? `?${query}` : ""}`; },
    async load() {
      this.loading = true; this.errorMessage = "";
      try {
        const query = this.query();
        const response = await fetch(`/api/v1/runs${query ? `?${query}` : ""}`);
        if (!response.ok) throw new Error(config.loadFailed);
        this.rows = await response.json();
      } catch (error) { this.errorMessage = error.message || config.loadFailed; }
      finally { this.loading = false; }
    },
  };
}

function historyDetail(runId, config) {
  return {
    ...historyHelpers(config),
    row: null,
    loading: false,
    errorMessage: "",
    feedbackStatus: "",
    feedback: { expected_label: "unknown", comment: "" },
    percent(value) {
      if (typeof value !== "number") return config.unknown;
      return new Intl.NumberFormat("hr-HR", { style: "percent", maximumFractionDigits: 1 }).format(value);
    },
    milliseconds(value) {
      if (typeof value !== "number") return config.unknown;
      return `${new Intl.NumberFormat("hr-HR").format(value)} ms`;
    },
    agentLabel(name) {
      const evidence = this.row?.response?.agent_evidence || [];
      return evidence.find((agent) => agent.name === name)?.label_hr || this.label(config.agentLabels, name);
    },
    interval(row) {
      if (typeof row?.interval_low !== "number" || typeof row?.interval_high !== "number") return config.unknown;
      return `${this.percent(row.interval_low)} – ${this.percent(row.interval_high)}`;
    },
    async load() {
      this.loading = true;
      try {
        const response = await fetch(`/api/v1/runs/${encodeURIComponent(runId)}`);
        if (!response.ok) throw new Error(await detailFromResponse(response, config.loadFailed));
        this.row = await response.json();
        if (this.row.feedback) this.feedback = { ...this.feedback, ...this.row.feedback };
      } catch (error) { this.errorMessage = error.message || config.loadFailed; }
      finally { this.loading = false; }
    },
    async submitFeedback() {
      this.feedbackStatus = "";
      try {
        const response = await fetch(`/api/v1/runs/${encodeURIComponent(runId)}/feedback`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(this.feedback),
        });
        if (!response.ok) throw new Error(config.feedbackFailed);
        const payload = await response.json();
        this.feedbackStatus = payload.status;
      } catch (_) { this.feedbackStatus = config.feedbackFailed; }
    },
  };
}

document.addEventListener("DOMContentLoaded", initialiseHeaderLogo);
