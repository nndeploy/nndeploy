"""Typed NumPy packets preserve ownership and array layout across replacement."""
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import numpy as np
import pytest
from nndeploy import _nndeploy_internal as native


@pytest.mark.parametrize('shape', [[], [3], [2, 3, 1], [1, 2, 3, 4]])
@pytest.mark.parametrize('setter', ['edge', 'node'])
def test_explicit_numpy_edge_preserves_shape_across_multiple_writes(shape, setter):
    edge = native.dag.Edge('array')
    edge.set_type(np.ndarray)
    node = native.dag.Node('producer', [], [edge])
    publish = edge.set if setter == 'edge' else node.set_output_data
    for index in range(2):
        value = np.full(shape, index + .5, dtype=np.float16)
        assert publish(value)
        result = edge.get(None)
        assert result.shape == tuple(shape)
        assert result.dtype == value.dtype
        np.testing.assert_array_equal(result, value)


def test_numpy_edge_retains_strided_readonly_array_until_last_reader_releases_it():
    edge = native.dag.Edge('array')
    edge.set_type(np.ndarray)
    value = np.arange(24, dtype=np.int64).reshape(2, 3, 4)[:, ::-1, ::2]
    value.setflags(write=False)
    original = value.copy()
    reference = weakref.ref(value)
    assert edge.set(value)
    result = edge.get(None)
    del value
    gc.collect()
    assert reference() is not None
    assert result.flags.writeable is False
    assert result.strides[1] < 0
    edge.set('replacement')
    np.testing.assert_array_equal(result, original)
    del result
    gc.collect()
    assert reference() is None


def test_numpy_declaration_survives_blocked_pipeline_publication():
    edge = native.dag.Edge('arrays')
    edge.set_type(np.ndarray)
    edge.set_parallel_type(native.base.ParallelType.Pipeline)
    edge.set_queue_max_size(1)
    edge.set_queue_overflow_policy(native.base.QueueOverflowPolicy.AllBackpressure)
    edge.mark_graph_output()
    edge.construct()
    edge.set(np.full((3,), 0, dtype=np.float16))

    def produce():
        for index in range(1, 8):
            assert edge.set(np.full((3,), index, dtype=np.float16))

    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            writer = pool.submit(produce)
            with pytest.raises(TimeoutError):
                writer.result(timeout=.05)
            reader = pool.submit(lambda: [edge.get_graph_output() for _ in range(8)])
            values = reader.result(timeout=5)
            writer.result(timeout=5)
            assert all(value.shape == (3,) and value.dtype == np.float16 for value in values)
            assert [value.tolist() for value in values] == [[index] * 3 for index in range(8)]
        finally:
            edge.request_terminate()
