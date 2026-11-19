# SPDX-License-Identifier: Apache-2.0
"""Ordered tile mapping must preserve feathering, padding and repeat requests."""

import pytest
import torch
from nndeploy.e_dit.algorithms.vsr.blending import tiled_restore_rect
from nndeploy.e_dit.algorithms.vsr.parallel import ParallelVSRRestorer


@pytest.mark.parametrize("shape", [(1, 3, 5, 11, 17), (1, 3, 2, 3, 4)])
def test_ordered_map_matches_serial(shape):
    frames = torch.randn(shape)
    calls = []

    def restore(x):
        return x.square() * 0.2 + x.mean()

    def ordered(windows):
        # Simulate reverse completion while exposing the original result order.
        windows = list(windows)
        calls.extend(x.shape for x in windows)
        results = {i: restore(windows[i]) for i in reversed(range(len(windows)))}
        yield from (results[i] for i in range(len(windows)))

    kwargs = {
        "tile_t": 4,
        "tile_h": 6,
        "tile_w": 8,
        "t_overlap": 2,
        "s_overlap": 2,
        "show_progress": False,
    }
    expected = tiled_restore_rect(frames, restore, **kwargs)
    for _ in range(2):
        actual = tiled_restore_rect(
            frames, restore, restore_windows_fn=ordered, **kwargs
        )
        assert torch.equal(actual, expected)
    assert calls
    assert all(tuple(shape) == (1, 3, 4, 6, 8) for shape in calls)


@pytest.mark.parametrize(
    "devices", [["cuda:0"], ["cuda:0", "cuda:0"], ["cpu", "cuda:0"]]
)
def test_invalid_devices_fail_before_loading(devices):
    with pytest.raises(ValueError):
        ParallelVSRRestorer(devices)


def test_closing_consumed_iterator_keeps_pool_usable(monkeypatch):
    from types import SimpleNamespace

    model = ParallelVSRRestorer.__new__(ParallelVSRRestorer)
    model.closed = False
    model.workers = []
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model.local = SimpleNamespace(restore_window=lambda x: x + 1)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    for _ in range(2):
        results = model.restore_windows([torch.zeros(1)])
        assert next(results).item() == 1
        results.close()
        assert not model.closed


@pytest.mark.parametrize("count", [1, 2, 3, 5])
def test_partial_batches_preserve_order_and_acknowledge(monkeypatch, count):
    from types import SimpleNamespace

    class Connection:
        output = None
        requests = 0

        def send(self, message):
            if isinstance(message, str):
                assert message == "consumed"
                assert self.output is not None
                self.output = None
            else:
                assert self.output is None, "More than one tile in flight"
                command, value = message
                assert command == "restore"
                self.output = value + 10
                self.requests += 1

        def poll(self, timeout):
            return self.output is not None

        def recv(self):
            return "result", self.output

    connection = Connection()
    model = ParallelVSRRestorer.__new__(ParallelVSRRestorer)
    model.closed = False
    model.workers = [(connection, None)]
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model.local = SimpleNamespace(restore_window=lambda x: x + 10)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    for _ in range(2):
        results = list(model.restore_windows(torch.tensor([i]) for i in range(count)))
        assert [x.item() for x in results] == list(range(10, 10 + count))
        assert connection.output is None
    assert connection.requests == 2 * (count // 2)


def test_between_batch_cancellation_preserves_pool(monkeypatch):
    from types import SimpleNamespace

    model = ParallelVSRRestorer.__new__(ParallelVSRRestorer)
    model.closed = False
    model.workers = []
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model.local = SimpleNamespace(restore_window=lambda x: x + 1)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)

    def windows():
        yield torch.zeros(1)
        raise TimeoutError("request expired")

    results = model.restore_windows(windows())
    assert next(results).item() == 1
    with pytest.raises(TimeoutError):
        next(results)
    assert not model.closed
    recovered = list(model.restore_windows([torch.zeros(1)]))
    assert len(recovered) == 1 and recovered[0].item() == 1
