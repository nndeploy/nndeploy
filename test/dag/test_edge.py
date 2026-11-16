"""Basic public Edge operations without machine-specific files or model SDKs."""
import numpy as np
import pytest

from nndeploy import _nndeploy_internal as native
from nndeploy.dag import Edge
from nndeploy.device import Tensor


def test_edge_empty_and_scalar_values():
    edge = Edge('test_edge')
    assert edge.get_name() == 'test_edge'
    assert edge.get(None) is None
    assert edge.get_graph_output() is None
    assert edge.set(1)
    assert edge.get(None) == 1
    assert edge.get_graph_output() == 1


def test_edge_tensor_and_created_buffer():
    edge = Edge('values')
    values = np.arange(4, dtype=np.int32)
    assert edge.set(Tensor.from_numpy(values))
    np.testing.assert_array_equal(np.asarray(edge.get(None)), values)
    host = native.device.get_default_host_device()
    buffer = edge.create(host, native.device.BufferDesc(16))
    np.asarray(buffer)[:] = 7
    assert edge.notify_written(buffer)
    np.testing.assert_array_equal(np.asarray(edge.get(None)), np.full(16, 7, dtype=np.uint8))


def test_edge_legacy_numpy_image():
    edge = Edge('image')
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    status = edge.set(image)
    if status == native.base.StatusCode.ErrorNotSupport:
        pytest.skip('Native OpenCV support is disabled')
    assert status
    np.testing.assert_array_equal(edge.get(None), image)
