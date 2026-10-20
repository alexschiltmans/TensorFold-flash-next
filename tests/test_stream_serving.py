"""Concurrent requests against a model with streamed experts: serialized by the server, never interleaved.

The expert stream allows one forward at a time (``ExpertStream._enter``: layers in order, one thread per forward).
Here the lane server's fake engine (``tests/lane_fakes.py``) walks that same check through every layer of every
prefill and round, so an interleaved forward would raise inside the engine. No model and no GPU work: the check
runs on a stand-in holding only the order state.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("mlx.core")

from tensorfold.families.qwen4_exp.expert_stream import ExpertStream  # noqa: E402
from tests.test_lane_server import FakeTokenizer, expected_reply, make_app  # noqa: E402
from tests.lane_fakes import FakeEngine  # noqa: E402

LAYERS = 6


class OrderGuard:
    """``ExpertStream``'s per-layer order check on a stand-in with only its order state (no arenas, no reads)."""

    def __init__(self, layers: int = LAYERS, layer_delay: float = 0.0005) -> None:
        self.state = SimpleNamespace(index={i: i for i in range(layers)}, arenas=[None] * layers, _next=None,
                                     _owner=None, flush=lambda: None)
        self.layers, self.layer_delay = layers, layer_delay
        self.forwards = 0
        self.errors: list[BaseException] = []

    def forward(self) -> None:
        try:
            for i in range(self.layers):
                ExpertStream._enter(self.state, i)
                time.sleep(self.layer_delay)          # a layer takes time: room for another forward to cut in
        except RuntimeError as err:
            self.errors.append(err)
            raise
        self.forwards += 1


class StreamedFakeEngine(FakeEngine):
    """The fake engine, every prefill and round a full guarded forward."""

    guard: OrderGuard

    def prefill(self, stream: Any, **kwargs: Any) -> list[Any]:
        self.guard.forward()
        return super().prefill(stream, **kwargs)

    def prefill_prefix(self, prompt_ids: Any, **kwargs: Any) -> list[Any]:
        self.guard.forward()
        return super().prefill_prefix(prompt_ids, **kwargs)

    def _forward_rows(self, rows: list[list[int]], cache: list[Any]) -> tuple[Any, Any]:
        self.guard.forward()
        return super()._forward_rows(rows, cache)


def streamed_app(guard: OrderGuard) -> Any:
    def factory(*args: Any, **kwargs: Any) -> StreamedFakeEngine:
        engine = StreamedFakeEngine(*args, **kwargs)
        engine.guard = guard
        return engine

    return make_app(lanes=1, engine_factory=factory)


def test_the_guard_catches_two_forwards_at_once() -> None:
    """Control: without the server's serialization, two concurrent forwards trip the stream's check."""

    guard = OrderGuard(layer_delay=0.002)
    start = threading.Barrier(2)

    def run() -> None:
        start.wait()
        try:
            guard.forward()
        except RuntimeError:
            pass

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert guard.errors and "one forward at a time" in str(guard.errors[0])


def test_a_title_request_and_a_turn_at_once_are_serialized_not_interleaved() -> None:
    guard = OrderGuard()
    app = streamed_app(guard)
    try:
        title = [{"role": "system", "content": "Write a ~5 word title for the next user message."},
                 {"role": "user", "content": "<user>fix the failing test</user>"}]
        turn = [{"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "fix the failing test"}]
        results: dict[str, dict[str, Any]] = {}
        deltas: list[str] = []

        def run(name: str, messages: list[dict[str, Any]], limit: int, sink: Any = None) -> None:
            results[name] = app.chat(messages, max_tokens=limit, on_delta=sink)

        background = threading.Thread(target=run, args=("title", title, 40, deltas.append))
        background.start()
        deadline = time.perf_counter() + 5
        while not deltas and time.perf_counter() < deadline:      # the title is decoding when the turn arrives
            time.sleep(0.001)
        assert deltas
        run("turn", turn, 16)
        background.join(timeout=30)
        assert not background.is_alive()
        assert guard.errors == [] and app.scheduler.failed_rounds == 0
        assert guard.forwards > 0
        assert results["title"]["content"] == expected_reply(app, title, 40)[1] == "".join(deltas)
        assert results["turn"]["content"] == expected_reply(app, turn, 16)[1]
    finally:
        app.close()


def test_many_concurrent_requests_are_serialized() -> None:
    guard = OrderGuard(layer_delay=0.0002)
    app = streamed_app(guard)
    try:
        prompts = [[{"role": "user", "content": f"request number {i} " * (1 + i % 3)}] for i in range(6)]
        prompts.append([{"role": "system", "content": "Write a ~5 word title for the next user message."},
                        {"role": "user", "content": "<user>request number 0</user>"}])
        results: dict[int, dict[str, Any]] = {}
        go = threading.Barrier(len(prompts))

        def run(i: int) -> None:
            go.wait()
            results[i] = app.chat(prompts[i], max_tokens=10)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert all(not t.is_alive() for t in threads)
        assert guard.errors == [] and app.scheduler.failed_rounds == 0
        for i, messages in enumerate(prompts):
            assert results[i]["content"] == expected_reply(app, messages, 10)[1], i
        assert max(r.streams for r in app.engine.round_stats) == 1          # one sequence a round, always
    finally:
        app.close()


def test_the_app_refuses_more_than_one_lane_for_a_streamed_model() -> None:
    from tensorfold.server.app import ChatApp

    model = SimpleNamespace()
    model.__dict__["expert_stream"] = object()
    with pytest.raises(ValueError, match="one lane"):
        ChatApp(model, FakeTokenizer(), served_name="x", lanes=2, engine_factory=FakeEngine)
