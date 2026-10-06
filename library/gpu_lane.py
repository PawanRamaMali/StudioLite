"""Shared GPU-lane coordination between the interactive filmmaker pipeline
and the long-running library jobs.

The filmmaker pipeline is latency-sensitive - a user clicked "render" and
is watching the Jobs panel. Library jobs (embed, transcribe, face-index,
re-encode) are background maintenance; they can wait a few seconds. When
both try to touch the GPU at once the film render stalls and the user
notices.

This module exposes a tiny coordination primitive:

- `interactive_busy` is a module-level `threading.Event`. When set, every
  library job pauses its per-item loop at the top until the event clears.
- `enter_interactive()` / `leave_interactive()` are reference-counted so
  two concurrent film pipelines both block library work until both have
  released the lane.
- `wait_for_interactive_idle(cancel_event)` is the hook library jobs call
  before each item - it polls the event and the cancel event so a user
  can still cancel a paused library job.
- `interactive()` is a context manager wrapper for use at the orchestrator
  stage-run boundary.

The semantics are intentionally minimal: this is a cooperative pause, not
a lock. Library jobs that have already handed work to a background pool
(e.g. a running ffmpeg sub-process) will finish that item before pausing;
the lane only guarantees "no new library item will start while the film
pipeline is running".
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

logger = logging.getLogger("studiolite.library.gpu_lane")


# Module-level singletons. The event is the fast-path check library jobs
# consult every iteration; the refcount lets nested or concurrent film
# pipelines share the "busy" state without racing on clear().
interactive_busy: threading.Event = threading.Event()
_interactive_refcount: int = 0
_refcount_lock: threading.Lock = threading.Lock()


def enter_interactive() -> None:
    """Signal that an interactive (filmmaker) stage is about to run.

    Increments the shared refcount and sets `interactive_busy`. Safe to
    call from any thread. Nested / concurrent callers all stack; the
    event only clears once every caller has paired with
    `leave_interactive()`.
    """
    global _interactive_refcount
    with _refcount_lock:
        _interactive_refcount += 1
        interactive_busy.set()
        logger.debug("enter_interactive: refcount=%d", _interactive_refcount)


def leave_interactive() -> None:
    """Pair with `enter_interactive()`. Clears the event when the last
    caller leaves. Never decrements below zero - a stray leave is logged
    and swallowed so a bug in one caller can't permanently wedge the
    library lane."""
    global _interactive_refcount
    with _refcount_lock:
        if _interactive_refcount <= 0:
            logger.warning("leave_interactive called with refcount=%d; "
                           "ignoring", _interactive_refcount)
            _interactive_refcount = 0
            interactive_busy.clear()
            return
        _interactive_refcount -= 1
        if _interactive_refcount == 0:
            interactive_busy.clear()
        logger.debug("leave_interactive: refcount=%d", _interactive_refcount)


def wait_for_interactive_idle(cancel_event: Optional[threading.Event] = None,
                              poll_interval: float = 0.5) -> bool:
    """Block until `interactive_busy` clears or `cancel_event` fires.

    Called at the top of every library per-item loop iteration. Returns
    True if the lane is idle and it's ok to proceed, False if the caller
    was cancelled while waiting (so the library job can exit cleanly).

    The fast path is cheap - a single `is_set()` on the event. Only
    paused jobs enter the poll loop.
    """
    if not interactive_busy.is_set():
        return True
    while interactive_busy.is_set():
        if cancel_event is not None and cancel_event.is_set():
            return False
        # Prefer waiting on the cancel event (if given) so a cancel wakes
        # us immediately; fall back to a plain sleep otherwise.
        if cancel_event is not None:
            if cancel_event.wait(timeout=poll_interval):
                return False
        else:
            # Use the event's own wait so a clear() wakes us right away.
            # `wait` returns True if the event is set; we actually want to
            # notice when it CLEARS, so just sleep-poll.
            interactive_busy.wait(timeout=poll_interval)
            # Fall through and re-check the loop condition.
    # Lane cleared while we were waiting.
    if cancel_event is not None and cancel_event.is_set():
        return False
    return True


@contextmanager
def interactive() -> Iterator[None]:
    """Context manager form of enter/leave. Use around any interactive
    GPU work that should take priority over library jobs:

        with interactive():
            runner(project)

    `leave_interactive()` runs even if the body raises, so a crashed
    stage doesn't leave the library lane permanently blocked.
    """
    enter_interactive()
    try:
        yield
    finally:
        leave_interactive()


__all__ = [
    "interactive_busy",
    "enter_interactive",
    "leave_interactive",
    "wait_for_interactive_idle",
    "interactive",
]
