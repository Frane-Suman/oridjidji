"""Small asyncio orchestrator for independent detector agents."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from oridjidji.schemas import AgentResult, AgentStatus


class Detector(Protocol):
    name: str

    def score(self, text: str) -> AgentResult: ...


class PreparedDetector(Detector, Protocol):
    """Agent whose expensive evidence can be shared with sibling agents."""

    @property
    def prepare_key(self) -> str: ...

    def prepare(self, text: str) -> Any: ...

    def score_prepared(self, evidence: Any) -> AgentResult: ...


@dataclass(frozen=True)
class PipelineEvent:
    """A truthful transition emitted when the underlying action occurs."""

    event: str
    elapsed_ms: int
    agent: str | None = None
    index: int = 0
    of: int = 0
    detail: str | None = None


@dataclass(frozen=True)
class PipelineOutcome:
    agents: list[AgentResult]
    final: Any | None
    elapsed_ms: int
    prepared_evidence: dict[str, Any] | None = None


EventSink = Callable[[PipelineEvent], Awaitable[None] | None]
Finalizer = Callable[[list[AgentResult]], Any]


async def _emit(sink: EventSink | None, event: PipelineEvent) -> None:
    if sink is None:
        return
    pending = sink(event)
    if inspect.isawaitable(pending):
        await pending


class Orchestrator:
    """Run agents concurrently, bound each wait, and restore declared order."""

    def __init__(
        self,
        detectors: Sequence[Detector],
        *,
        timeout_s: float = 30.0,
        execution_policy: Literal["sequential", "concurrent"] = "concurrent",
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if execution_policy not in {"sequential", "concurrent"}:
            raise ValueError("execution_policy must be sequential or concurrent")
        self.detectors = list(detectors)
        self.timeout_s = timeout_s
        self.execution_policy = execution_policy
        # ``asyncio.wait_for(asyncio.to_thread(...))`` cancels only the asyncio waiter;
        # the native model call keeps running.  Retain those tasks so the API can keep
        # the Advanced admission lease until the underlying CPU work has really ended.
        self._native_tasks: set[asyncio.Task[Any]] = set()

    @property
    def has_pending_native_work(self) -> bool:
        """Whether a timed-out detector is still executing in a worker thread."""
        return any(not task.done() for task in self._native_tasks)

    async def wait_until_native_idle(self) -> None:
        """Wait for timed-out native calls without re-raising their detector errors."""
        pending = [task for task in self._native_tasks if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _native_call(self, function: Callable[..., Any], *args: Any) -> Any:
        """Run one blocking call while preserving its lifetime after an API timeout."""
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        self._native_tasks.add(task)

        def completed(done: asyncio.Task[Any]) -> None:
            # Retrieve an exception even when the HTTP request has already returned.
            if not done.cancelled():
                done.exception()
            self._native_tasks.discard(done)

        task.add_done_callback(completed)
        return await asyncio.wait_for(asyncio.shield(task), timeout=self.timeout_s)

    async def run(
        self,
        text: str,
        *,
        on_event: EventSink | None = None,
        finalizer: Finalizer | None = None,
        emit_run_finished: bool = True,
    ) -> PipelineOutcome:
        started = time.perf_counter()

        def elapsed() -> int:
            return int((time.perf_counter() - started) * 1000)

        await _emit(on_event, PipelineEvent("run_started", elapsed(), of=len(self.detectors)))

        async def one(index: int, detector: Detector) -> tuple[int, AgentResult]:
            await _emit(
                on_event,
                PipelineEvent(
                    "agent_started",
                    elapsed(),
                    agent=detector.name,
                    index=index,
                    of=len(self.detectors),
                ),
            )
            try:
                result = await self._native_call(detector.score, text)
            except TimeoutError:
                result = AgentResult(
                    name=detector.name,
                    version="unknown",
                    status=AgentStatus.FAILED,
                    error=f"timeout after {self.timeout_s:g}s",
                )
            except Exception as exc:  # noqa: BLE001 - one detector must not fail the request
                result = AgentResult(
                    name=detector.name,
                    version="unknown",
                    status=AgentStatus.FAILED,
                    error=f"{type(exc).__name__}: {exc}"[:300],
                )
            event = "agent_finished" if result.status is not AgentStatus.FAILED else "agent_failed"
            await _emit(
                on_event,
                PipelineEvent(
                    event,
                    elapsed(),
                    agent=detector.name,
                    index=index,
                    of=len(self.detectors),
                    detail=result.error,
                ),
            )
            return index, result

        prepared: dict[str, list[tuple[int, PreparedDetector]]] = {}
        prepared_evidence: dict[str, Any] = {}
        ordinary: list[tuple[int, Detector]] = []
        for index, detector in enumerate(self.detectors):
            if all(
                hasattr(detector, name) for name in ("prepare_key", "prepare", "score_prepared")
            ):
                shared = cast(PreparedDetector, detector)  # runtime shape checked above
                prepared.setdefault(shared.prepare_key, []).append((index, shared))
            else:
                ordinary.append((index, detector))

        async def group(
            members: list[tuple[int, PreparedDetector]],
        ) -> list[tuple[int, AgentResult]]:
            group_started = time.perf_counter()
            for index, detector in members:
                await _emit(
                    on_event,
                    PipelineEvent(
                        "agent_started",
                        elapsed(),
                        agent=detector.name,
                        index=index,
                        of=len(self.detectors),
                    ),
                )
            try:
                evidence = await self._native_call(members[0][1].prepare, text)
                prepared_evidence[members[0][1].prepare_key] = evidence
                results = [
                    (index, detector.score_prepared(evidence)) for index, detector in members
                ]
            except TimeoutError:
                results = [
                    (
                        index,
                        AgentResult(
                            name=detector.name,
                            version="unknown",
                            status=AgentStatus.FAILED,
                            error=f"timeout after {self.timeout_s:g}s",
                        ),
                    )
                    for index, detector in members
                ]
            except Exception as exc:  # noqa: BLE001 - shared failure belongs to every consumer
                results = [
                    (
                        index,
                        AgentResult(
                            name=detector.name,
                            version="unknown",
                            status=AgentStatus.FAILED,
                            error=f"{type(exc).__name__}: {exc}"[:300],
                        ),
                    )
                    for index, detector in members
                ]
            total_ms = int((time.perf_counter() - group_started) * 1000)
            final_results = []
            for index, result in results:
                result = result.model_copy(update={"latency_ms": total_ms})
                event = (
                    "agent_finished" if result.status is not AgentStatus.FAILED else "agent_failed"
                )
                await _emit(
                    on_event,
                    PipelineEvent(
                        event,
                        elapsed(),
                        agent=result.name,
                        index=index,
                        of=len(self.detectors),
                        detail=result.error,
                    ),
                )
                final_results.append((index, result))
            return final_results

        tasks: list[tuple[int, Awaitable[Any]]] = [
            (index, one(index, detector)) for index, detector in ordinary
        ]
        tasks.extend((members[0][0], group(members)) for members in prepared.values())
        ordered_tasks = [task for _index, task in sorted(tasks, key=lambda item: item[0])]
        if self.execution_policy == "sequential":
            gathered = [await task for task in ordered_tasks]
        else:
            gathered = list(await asyncio.gather(*ordered_tasks))
        completed: list[tuple[int, AgentResult]] = []
        for item in gathered:
            if isinstance(item, list):
                completed.extend(item)
            else:
                completed.append(item)
        agents = [result for _index, result in sorted(completed, key=lambda row: row[0])]
        final: Any | None = None
        if finalizer is not None:
            await _emit(on_event, PipelineEvent("fusion_started", elapsed(), of=len(agents)))
            final = finalizer(agents)
        if emit_run_finished:
            await _emit(on_event, PipelineEvent("run_finished", elapsed(), of=len(agents)))
        return PipelineOutcome(
            agents=agents,
            final=final,
            elapsed_ms=elapsed(),
            prepared_evidence=prepared_evidence,
        )
