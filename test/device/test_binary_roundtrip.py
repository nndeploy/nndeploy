import struct

import numpy as np

from nndeploy import _nndeploy_internal as native
from nndeploy.device import Buffer, Device, Tensor, create_tensor_from_numpy


def test_buffer_copy_modify_and_binary_roundtrip():
    payload = bytes([0, 255, 1, 128])
    wire = struct.pack("=Q", len(payload)) + payload
    host = Device("cpu")
    src, dst = Buffer(host, 1), Buffer(host, len(payload))
    assert src.deserialize(wire)
    assert src.copy_to(dst)
    assert dst.serialize() == wire
    assert not dst.deserialize(wire[:-1])
    assert native.device.Buffer.serialize(dst) == wire
    assert dst.just_modify(2)
    assert native.device.Buffer.get_size(dst) == 2
    assert not dst.just_modify(5)


def test_tensor_binary_roundtrip():
    values = np.array([[0, 255], [128, 1]], dtype=np.uint8)
    tensor = create_tensor_from_numpy(values)
    restored = Tensor()
    assert restored.deserialize(tensor.serialize())
    np.testing.assert_array_equal(np.asarray(restored), values)


def test_tensor_contiguity_uses_byte_strides():
    tensor = create_tensor_from_numpy(np.zeros((2, 3, 5), dtype=np.float32))
    desc = tensor.get_desc()
    desc.stride_ = [60, 20, 4]
    assert Tensor(desc).is_continue()
    desc.stride_ = [64, 20, 4]
    assert not native.device.Tensor(desc).is_continue()
