"""Tests for the GPU-lane coordination primitive.

Covers the four invariants the MVP relies on:

1. A library job pauses when `enter_interactive()` is called.
2. It resumes once `leave_interactive()` fires.
3. Two nested `enter_interactive()` calls require two `leave_interactive()`
   calls before the library work resumes (reference counting).
4. A cancelled library job exits cleanly even while paused.

The tests use real threads and short sleeps - the coordination primitive
is itself tiny so there's no value in mocking it; we want to exercise the
actual event + poll loop.
"""
from __future__ import annotations

import threading
import time

import pytest

from library import gpu_lane
from library.gpu_lane import (
    enter_interactive,
    interactive,
    interactive_busy,
    leave_interactive,
    wait_for_interactive_idle,
)


# How long we're willing to wait for a worker thread to observe a state
# change. Generous so a loaded CI box doesn't flake. The coordinator uses
# ~5ms sleeps so even 1s is thousands of chances to observe.
_WAIT_S = 2.0


@pytest.fixture(autouse=True)
def _reset_lane():
    """Each test gets a clean lane. The module-level singletons would
    otherwise leak state across tests - a failed test could wedge the
    lane for every test that followed."""
    gpu_lane._interactive_refcount = 0
    interactive_busy.clear()
    yield
    gpu_lane._interactive_refcount = 0
    interactive_busy.clear()


class _FakeLibraryJob(threading.Thread):
    """Stand-in for EmbedJob/TranscribeJob/etc. Loops over a fake list
    and calls `wait_for_interactive_idle` at the top of each iteration
    just like the real jobs do. Records which items it processed and
    whether the final exit was clean or cancelled."""

    def __init__(self, items: int = 5, poll_interval: float = 0.005):
        super().__init__(daemon=True, name="fake-library-job")
        self._items = items
        self._poll = poll_interval
        self._cancel = threading.Event()
        self.processed: list[int] = []
        self.exited_cleanly: bool = False
        self.cancelled: bool = False
        self.paused_count: int = 0

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        for i in range(self._items):
            if interactive_busy.is_set():
                self.paused_count += 1
            ok = wait_for_interactive_idle(self._cancel, poll_interval=self._poll)
            if not ok:
                self.cancelled = True
                return
            if self._cancel.is_set():
                self.cancelled = True
                return
            self.processed.append(i)
            # Simulate per-item work so the thread doesn't finish before
            # the test has a chance to toggle the lane.
            time.sleep(self._poll)
        self.exited_cleanly = True


def _wait_for(predicate, timeout: float = _WAIT_S) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_enter_sets_event_and_refcount_increments():
    """Smoke test for the primitive itself."""
    assert not interactive_busy.is_set()
    enter_interactive()
    try:
        assert interactive_busy.is_set()
        assert gpu_lane._interactive_refcount == 1
    finally:
        leave_interactive()
    assert not interactive_busy.is_set()
    assert gpu_lane._interactive_refcount == 0


def test_library_job_pauses_when_enter_interactive_called():
    """A library job that starts work mid-stream should block the moment
    the lane goes busy."""
    enter_interactive()
    try:
        job = _FakeLibraryJob(items=5)
        job.start()
        # Give the job a beat to spin into its loop and discover the lane
        # is busy. It should NOT make progress.
        time.sleep(0.05)
        assert job.processed == [], (
            "job processed items while lane was busy: " + str(job.processed)
        )
        assert job.is_alive(), "job exited instead of pausing"
    finally:
        leave_interactive()
    # Now that we've left the lane the job should drain and finish.
    assert _wait_for(lambda: not job.is_alive()), "job did not resume"
    assert job.exited_cleanly is True
    assert job.processed == list(range(5))
    # It saw at least one iteration where the lane was busy.
    assert job.paused_count >= 1


def test_library_job_resumes_after_leave_interactive():
    """Variant of the previous: here we start a job BEFORE entering the
    lane, let it process a few items, then enter -> leave and watch it
    finish."""
    job = _FakeLibraryJob(items=10, poll_interval=0.01)
    job.start()

    # Let it get some work done first.
    assert _wait_for(lambda: len(job.processed) >= 1)

    enter_interactive()
    try:
        # Snapshot progress and wait a beat; it should stall.
        snapshot = len(job.processed)
        time.sleep(0.1)
        assert len(job.processed) == snapshot, (
            "job kept processing while lane was busy; progressed from "
            f"{snapshot} to {len(job.processed)}"
        )
    finally:
        leave_interactive()

    # After leaving, it finishes.
    assert _wait_for(lambda: not job.is_alive())
    assert job.exited_cleanly is True
    assert job.processed == list(range(10))


def test_nested_enters_require_matching_leaves_before_resume():
    """Reference counting: two concurrent film pipelines both block
    library work until both finish."""
    job = _FakeLibraryJob(items=4, poll_interval=0.01)
    job.start()
    # Wait for the job to make some progress before we touch the lane.
    assert _wait_for(lambda: len(job.processed) >= 1)

    enter_interactive()  # pipeline A
    enter_interactive()  # pipeline B
    try:
        assert gpu_lane._interactive_refcount == 2
        snapshot = len(job.processed)
        time.sleep(0.1)
        assert len(job.processed) == snapshot

        # Only one leave: the lane must stay busy.
        leave_interactive()
        assert interactive_busy.is_set(), "lane cleared after only one leave"
        assert gpu_lane._interactive_refcount == 1
        snapshot = len(job.processed)
        time.sleep(0.1)
        assert len(job.processed) == snapshot, (
            "job resumed after only one of two leaves"
        )
    finally:
        leave_interactive()  # last pipeline leaves

    assert not interactive_busy.is_set()
    assert _wait_for(lambda: not job.is_alive())
    assert job.exited_cleanly is True
    assert job.processed == list(range(4))


def test_cancelled_job_exits_cleanly_while_paused():
    """A cancel event fires while the job is sitting in the paused poll
    loop. The job should notice and exit without processing more items."""
    enter_interactive()
    try:
        job = _FakeLibraryJob(items=5, poll_interval=0.01)
        job.start()
        # Let it park in the paused loop.
        time.sleep(0.05)
        assert job.processed == [], "job processed while paused"
        assert job.is_alive(), "job exited instead of pausing"

        # Fire cancel while the lane is still busy.
        job.cancel()
        # It should exit even though the lane is still held.
        assert _wait_for(lambda: not job.is_alive()), (
            "job did not exit after cancel while paused"
        )
        assert job.cancelled is True
        assert job.exited_cleanly is False
        assert job.processed == []
    finally:
        leave_interactive()


def test_context_manager_pairs_enter_and_leave():
    """`interactive()` is the context-manager wrapper used by the
    orchestrator. Must set on entry, clear on exit, and clear even if
    the body raises."""
    assert not interactive_busy.is_set()
    with interactive():
        assert interactive_busy.is_set()
        assert gpu_lane._interactive_refcount == 1
    assert not interactive_busy.is_set()
    assert gpu_lane._interactive_refcount == 0

    with pytest.raises(RuntimeError):
        with interactive():
            assert interactive_busy.is_set()
            raise RuntimeError("boom")
    assert not interactive_busy.is_set()
    assert gpu_lane._interactive_refcount == 0


def test_wait_returns_true_immediately_when_lane_is_idle():
    """Fast path: no event, no cancel -> immediate True."""
    cancel = threading.Event()
    t0 = time.time()
    assert wait_for_interactive_idle(cancel, poll_interval=1.0) is True
    assert time.time() - t0 < 0.05, "fast path should be sub-50ms"


def test_wait_returns_false_when_cancel_fires_before_lane_clears():
    cancel = threading.Event()
    enter_interactive()
    try:
        def _fire_cancel():
            time.sleep(0.05)
            cancel.set()
        threading.Thread(target=_fire_cancel, daemon=True).start()
        assert wait_for_interactive_idle(cancel, poll_interval=0.01) is False
    finally:
        leave_interactive()


def test_leave_without_enter_does_not_underflow():
    """A stray leave shouldn't put the refcount negative or wedge the
    lane in a weird state."""
    # Pre-condition: lane idle.
    assert gpu_lane._interactive_refcount == 0
    leave_interactive()  # no matching enter
    assert gpu_lane._interactive_refcount == 0
    assert not interactive_busy.is_set()
    # And a subsequent enter/leave still works.
    enter_interactive()
    assert interactive_busy.is_set()
    leave_interactive()
    assert not interactive_busy.is_set()
