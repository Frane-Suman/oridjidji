"""End-to-end browser checks for the localhost thesis workbench."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, expect

from oridjidji.i18n import MESSAGES_HR, NOTICE_HR

pytestmark = pytest.mark.integration

CROATIAN = (
    "Gradsko vijeće raspravljalo je o proračunu za sljedeću godinu i ulaganjima "
    "u obrazovanje, promet i obnovu javnih zgrada. Rasprava je obuhvatila izvore "
    "financiranja, rokove provedbe i način izvještavanja građana o rezultatima. "
) * 12
AXE = Path(__file__).parent / "vendor" / "axe.min.js"


def _page(browser: Browser, *, reduced_motion: bool = False) -> Page:
    context = browser.new_context(
        locale="hr-HR",
        reduced_motion="reduce" if reduced_motion else "no-preference",
        permissions=["clipboard-read", "clipboard-write"],
    )
    return context.new_page()


def _submit(page: Page, base_url: str, *, mode: str, store: bool = False) -> str:
    page.goto(base_url)
    _choose_mode(page, mode)
    page.get_by_test_id("editor").fill(CROATIAN)
    if store:
        page.locator(".editor-options > summary").click()
        page.get_by_test_id("store-opt-in").check()
    page.get_by_test_id("submit").click()
    result = page.get_by_test_id("result-container")
    expect(result).to_be_visible(timeout=20_000)
    expect(page.get_by_test_id("result-heading")).to_be_focused()
    expect(page.get_by_test_id("store-opt-in")).not_to_be_checked()
    run_id = result.get_attribute("data-run-id")
    assert run_id
    return run_id


def _choose_mode(page: Page, mode: str) -> None:
    """Choose the visible mode card; native radios are intentionally visually hidden."""
    test_id = f"mode-{'advanced' if mode == 'napredni' else 'basic'}"
    page.get_by_test_id(test_id).locator("..").click()
    expect(page.get_by_test_id(test_id)).to_be_checked()


def _serious_accessibility_issues(page: Page) -> list[dict[str, object]]:
    page.add_script_tag(path=str(AXE))
    result = page.evaluate(
        "axe.run(document, {runOnly: {type: 'tag', values: ['wcag2a', 'wcag2aa']}})"
    )
    return [
        {"id": row["id"], "impact": row["impact"], "nodes": len(row["nodes"])}
        for row in result["violations"]
        if row["impact"] in {"serious", "critical"}
    ]


def test_mode_is_execution_choice_and_preference_persists(browser: Browser, live_site: str) -> None:
    page = _page(browser)
    page.goto(live_site)
    expect(page.get_by_role("heading", level=1)).to_have_text(
        "Koliko je vjerojatno da je ovaj tekst napisao model?"
    )
    _choose_mode(page, "napredni")
    page.reload()
    expect(page.get_by_test_id("mode-advanced")).to_be_checked()
    _submit(page, live_site, mode="napredni")
    expect(page.get_by_text(MESSAGES_HR["ui_runtime_unavailable_title"])).to_be_visible()
    page.context.close()


def test_keyboard_only_submission_moves_focus_to_result(browser: Browser, live_site: str) -> None:
    page = _page(browser)
    page.goto(live_site)
    page.keyboard.press("Tab")
    expect(page.locator(".skip-link")).to_be_focused()
    page.get_by_test_id("mode-advanced").focus()
    page.keyboard.press("Space")
    expect(page.get_by_test_id("mode-advanced")).to_be_checked()
    editor = page.get_by_test_id("editor")
    editor.fill(CROATIAN)
    editor.press("Control+Enter")
    expect(page.get_by_test_id("result-heading")).to_be_focused(timeout=20_000)
    page.context.close()


def test_analysis_replaces_editor_and_back_starts_with_empty_text(
    browser: Browser, live_site: str
) -> None:
    page = _page(browser)
    page.goto(live_site)
    editor_pane = page.get_by_test_id("editor-pane")
    result_pane = page.get_by_test_id("result-pane")
    hero = page.locator(".hero")
    expect(hero).to_be_visible()
    expect(editor_pane).to_be_visible()
    expect(result_pane).to_be_hidden()

    page.get_by_test_id("editor").fill(CROATIAN)
    page.get_by_test_id("submit").click()
    expect(hero).to_be_hidden()
    expect(editor_pane).to_be_hidden()
    expect(result_pane).to_be_visible()
    expect(page.get_by_test_id("result-container")).to_be_visible(timeout=20_000)

    page.get_by_test_id("back-to-editor").click()
    expect(result_pane).to_be_hidden()
    expect(hero).to_be_visible()
    expect(editor_pane).to_be_visible()
    expect(page.get_by_test_id("editor")).to_have_value("")
    expect(page.get_by_test_id("editor")).to_be_focused()
    page.context.close()


def test_basic_result_copy_and_opt_in_history_privacy(browser: Browser, live_site: str) -> None:
    page = _page(browser)
    run_without_text = _submit(page, live_site, mode="osnovni")
    page.get_by_test_id("copy-report").click()
    expect(page.get_by_text(MESSAGES_HR["ui_result_copied"])).to_be_visible()
    copied = page.evaluate("navigator.clipboard.readText()")
    assert all(sentence in copied for sentence in NOTICE_HR)
    assert _serious_accessibility_issues(page) == []

    run_with_text = _submit(page, live_site, mode="osnovni", store=True)
    page.goto(f"{live_site}/povijest/{run_without_text}")
    expect(page.get_by_text(MESSAGES_HR["ui_history_text_absent"])).to_be_visible()
    expect(page.get_by_text(MESSAGES_HR["ui_history_timings_title"])).to_be_visible()
    expect(page.get_by_text(MESSAGES_HR["ui_notice_statistical"])).to_be_visible()
    page.goto(f"{live_site}/povijest/{run_with_text}")
    expect(page.locator("pre.stored-text")).to_contain_text("Gradsko vijeće")
    page.context.close()


def test_history_filters_feedback_and_csv(browser: Browser, live_site: str) -> None:
    page = _page(browser)
    page.goto(f"{live_site}/povijest")
    expect(page.get_by_test_id("history-filters")).to_be_visible()
    page.locator("select").first.select_option("osnovni")
    page.get_by_role("button", name=MESSAGES_HR["ui_history_apply"]).click()
    expect(page.locator(".history-card").first).to_be_visible()
    href = page.locator(".history-card a").first.get_attribute("href")
    assert href
    page.goto(f"{live_site}{href}")
    page.get_by_test_id("feedback-form").locator("select").select_option("unknown")
    page.get_by_test_id("feedback-form").get_by_role("button").click()
    expect(page.get_by_text(MESSAGES_HR["ui_feedback_saved"])).to_be_visible()
    response = page.request.get(f"{live_site}/api/v1/runs.csv?mode=osnovni")
    assert response.ok and "mode" in response.text()
    page.context.close()


def test_reflow_reduced_motion_and_automated_accessibility(
    browser: Browser, live_site: str
) -> None:
    page = _page(browser, reduced_motion=True)
    page.set_viewport_size({"width": 320, "height": 900})
    page.goto(live_site)
    static_src = page.locator(".brand-mark").get_attribute("src")
    assert static_src and static_src.endswith("oridjidji-logo-v2.webp")
    overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    assert overflow <= 0

    serious = _serious_accessibility_issues(page)
    assert serious == [], json.dumps(serious, indent=2)
    page.context.close()

    zoom_page = _page(browser, reduced_motion=True)
    zoom_page.set_viewport_size({"width": 720, "height": 900})
    zoom_page.goto(live_site)
    zoom_overflow = zoom_page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    assert zoom_overflow <= 0
    zoom_page.context.close()


def test_header_stays_visible_and_blurs_after_scroll(browser: Browser, live_site: str) -> None:
    page = _page(browser)
    page.goto(f"{live_site}/pregled")
    header = page.locator("[data-sticky-header]")
    expect(header).to_be_visible()
    expect(header.locator(".wordmark")).to_have_count(0)
    expect(header.locator(".masthead-tagline")).to_have_count(0)
    assert header.evaluate("node => getComputedStyle(node).position") == "sticky"
    assert "masthead--scrolled" not in (header.get_attribute("class") or "")
    page.evaluate("window.scrollTo(0, 500)")
    expect(header).to_have_class("masthead masthead--scrolled")
    blur = header.evaluate("node => getComputedStyle(node).backdropFilter")
    assert blur != "none"
    assert header.bounding_box()["y"] == pytest.approx(0, abs=1)
    page.context.close()


def test_development_gallery_covers_runtime_states_without_script_errors(
    browser: Browser, live_site: str
) -> None:
    page = _page(browser, reduced_motion=True)
    script_errors: list[str] = []
    page.on("pageerror", lambda error: script_errors.append(str(error)))
    page.goto(f"{live_site}/pregled")
    for key in (
        "ui_review_empty",
        "ui_review_loading",
        "ui_runtime_research_title",
        "ui_runtime_fallback_title",
        "ui_runtime_unavailable_title",
        "ui_review_errors",
    ):
        expect(page.get_by_text(MESSAGES_HR[key], exact=True).first).to_be_visible()
    assert script_errors == []
    assert _serious_accessibility_issues(page) == []
    page.context.close()


def test_method_and_history_shells_have_no_serious_accessibility_issues(
    browser: Browser, live_site: str
) -> None:
    page = _page(browser, reduced_motion=True)
    for path in ("/metoda", "/povijest"):
        page.goto(f"{live_site}{path}")
        assert _serious_accessibility_issues(page) == []
    page.context.close()
