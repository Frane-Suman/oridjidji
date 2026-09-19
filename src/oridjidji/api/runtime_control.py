"""Small process-level controls shared by the HTTP composition root."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections import defaultdict, deque

from fastapi import HTTPException

from oridjidji.i18n import MESSAGES_HR


def rss_mb() -> float:
    """Return current resident memory without adding a serving dependency."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        memory_info.restype = wintypes.BOOL
        handle = get_process()
        if memory_info(handle, ctypes.byref(counters), counters.cb):
            return float(counters.WorkingSetSize) / (1024 * 1024)
        return 0.0
    if sys.platform.startswith("linux"):
        from pathlib import Path

        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        resident_pages = int(fields[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE")) / (1024 * 1024)
    import resource

    resident = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return resident / (1024 * 1024)


class RequestRateLimiter:
    """Small in-process limiter for the two local analysis endpoints."""

    def __init__(self, requests: int, window_s: float = 60.0) -> None:
        self.requests = max(1, requests)
        self.window_s = window_s
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> None:
        now = time.monotonic()
        async with self._lock:
            calls = self._calls[key]
            while calls and now - calls[0] >= self.window_s:
                calls.popleft()
            if len(calls) >= self.requests:
                retry = max(1, int(self.window_s - (now - calls[0]) + 0.999))
                raise HTTPException(
                    status_code=429,
                    detail=MESSAGES_HR["ui_error_analysis_failed"],
                    headers={"Retry-After": str(retry)},
                )
            calls.append(now)


class AdvancedBusy(RuntimeError):
    """The one-active plus one-waiting Advanced capacity is exhausted."""


class AdvancedAdmission:
    """Serialize access to the persistent causal pair and bound its queue to one."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._active = False
        self._waiting = 0

    async def acquire(self) -> None:
        async with self._condition:
            if not self._active:
                self._active = True
                return
            if self._waiting:
                raise AdvancedBusy
            self._waiting = 1
            try:
                while self._active:
                    await self._condition.wait()
                self._active = True
            finally:
                self._waiting = 0

    async def release(self) -> None:
        async with self._condition:
            self._active = False
            self._condition.notify(1)
