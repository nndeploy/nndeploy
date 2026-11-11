import gc
import mmap
import sys
import unittest
import weakref

import numpy as np

from nndeploy import _nndeploy_internal as native
from nndeploy.device import Buffer, Tensor, create_tensor_from_numpy


class NumpyInterop(unittest.TestCase):
    def test_input_layout_and_readonly(self):
        image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        readonly = image.copy()
        readonly.flags.writeable = False
        for source in (image[:, ::2], image[::-1], image.T, readonly,
                       np.broadcast_to(image[:1], (4, 3, 4))):
            for factory in (Tensor.from_numpy, Buffer.from_numpy):
                with self.subTest(shape=source.shape, factory=factory):
                    result = factory(source)
                    if isinstance(result, Buffer):
                        actual = result.to_numpy(source.dtype).reshape(source.shape)
                    else:
                        actual = np.asarray(result)
                    np.testing.assert_array_equal(actual, source)
                    original = source.copy()
                    np.asarray(result).flat[0] = 17
                    np.testing.assert_array_equal(source, original)

    def test_zero_copy_owner_follows_native_shallow_copies(self):
        for factory in (Tensor.from_numpy, create_tensor_from_numpy, Buffer.from_numpy):
            with self.subTest(factory=factory):
                source = np.arange(8, dtype=np.uint8)
                owner = weakref.ref(source)
                first = factory(source)
                shallow = type(first)(first)
                np.asarray(shallow)[2] = 42
                self.assertEqual(source[2], 42)
                del source, first
                gc.collect()
                self.assertIsNotNone(owner())
                self.assertEqual(np.asarray(shallow)[2], 42)
                del shallow
                gc.collect()
                self.assertIsNone(owner())

    def test_resizable_buffer_exporter_stays_pinned(self):
        data = bytearray(b'abc')
        tensor = native.device.Tensor.from_numpy(data, native.base.DeviceType())
        with self.assertRaises(BufferError):
            data.extend(b'def')
        del tensor
        data.extend(b'def')

    def test_buffer_view_survives_tensor_clear(self):
        tensor = Tensor.from_numpy(np.arange(6, dtype=np.uint8))
        buffer = tensor.get_buffer()
        tensor.clear()
        del tensor
        gc.collect()
        self.assertEqual(buffer.get_size(), 6)
        np.testing.assert_array_equal(np.asarray(buffer), np.arange(6))

    def test_tensor_dtype_and_fill(self):
        for dtype in (np.uint8, np.int16, np.uint64, np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                source = np.zeros(8, dtype=dtype)
                tensor = Tensor.from_numpy(source)
                self.assertEqual(np.asarray(tensor).dtype, source.dtype)
                value = 2**63 + 1 if dtype is np.uint64 else 3
                self.assertTrue(tensor.set(value))
                np.testing.assert_array_equal(source, np.full_like(source, value))

    def test_buffer_dtype_and_public_type(self):
        source = np.array([0, 1], dtype=np.uint8)
        buffer = Buffer.from_numpy(source)
        self.assertIsInstance(buffer, Buffer)
        self.assertEqual(buffer.get_size(), 2)
        for dtype in ('>u2', '<u2'):
            result = buffer.to_numpy(np.dtype(dtype))
            self.assertEqual(result.dtype, np.dtype(dtype))
            np.testing.assert_array_equal(result, source.view(dtype))
        dates = np.arange(3).astype("datetime64[D]")
        np.testing.assert_array_equal(Buffer.from_numpy(dates).to_numpy(dates.dtype), dates)
        for dtype in (object, [('payload', object)]):
            with self.assertRaises(TypeError):
                Buffer.from_numpy(np.empty(1, dtype=dtype))
            with self.assertRaises(TypeError):
                buffer.to_numpy(dtype)

    def test_tensor_rejects_unavailable_device(self):
        device = native.base.DeviceType(native.base.DeviceTypeCode.notsupport)
        with self.assertRaises(ValueError):
            native.device.Tensor.from_numpy(np.ones(1), device)

    def test_non_host_upload_and_failure_cleanup(self):
        # Exercise device dispatch on CPU-only builders without CUDA hardware.
        host = native.device.get_default_host_device()
        code = native.base.DeviceTypeCode.notsupport
        target = native.base.DeviceType(code)

        class UploadDevice(native.device.Device):
            allocations = 0
            fail = False

            def to_buffer_desc(self, desc, config):
                return host.to_buffer_desc(desc, config)

            def allocate(self, desc):
                self.allocations += 1
                return host.allocate(desc)

            def deallocate(self, ptr):
                self.allocations -= 1
                host.deallocate(ptr)

            def upload(self, src, dst, size, stream):
                if self.fail:
                    return native.base.Status(native.base.StatusCode.ErrorInvalidValue)
                return host.copy(src, dst, size)

            def download(self, src, dst, size, stream):
                return host.copy(src, dst, size)

        device = UploadDevice(target)

        class Architecture(native.device.Architecture):
            def get_device(self, device_id):
                return device

        architecture = Architecture(code)
        native.device.register_architecture(code, architecture)
        self.addCleanup(native.device.destory_architecture, code)
        source = np.arange(12, dtype=np.uint16)[::-3]
        expected = source.copy()
        result = Tensor.from_numpy(source, target)
        source.fill(0)
        output = Tensor.from_numpy(np.zeros_like(expected))
        self.assertTrue(result.copy_to(output))
        np.testing.assert_array_equal(np.asarray(output), expected)
        self.assertTrue(result.set(13))
        self.assertTrue(result.copy_to(output))
        np.testing.assert_array_equal(np.asarray(output), np.full_like(expected, 13))
        del result
        self.assertEqual(device.allocations, 0)

        device.fail = True
        with self.assertRaises(RuntimeError):
            Tensor.from_numpy(source, target)
        self.assertEqual(device.allocations, 0)

    def test_bounds_of_large_dimensions(self):
        if sys.maxsize <= 2**32:
            self.skipTest("requires a 64-bit address space")
        length = 2**31 + 3
        try:
            mapping = mmap.mmap(-1, length)
        except (OSError, OverflowError):
            self.skipTest("cannot reserve a 2 GiB virtual mapping")
        with mapping:
            source = np.ndarray((length,), dtype=np.uint8, buffer=mapping)
            with self.assertRaises(ValueError):
                Tensor.from_numpy(source)
            buffer = Buffer.from_numpy(source)
            self.assertEqual(buffer.get_size(), length)
            self.assertEqual(memoryview(buffer).shape, (length,))
            del buffer, source

    def test_tensor_validates_all_byte_strides(self):
        source = np.arange(8, dtype=np.uint8)
        tensor = Tensor.from_numpy(source)
        desc = tensor.get_desc()
        desc.shape_ = [2, 2]
        for strides in ([2, 2000], [2], [2, sys.maxsize * 2 + 1]):
            desc.stride_ = strides
            self.assertFalse(tensor.just_modify(desc))
            np.testing.assert_array_equal(np.asarray(tensor), source)
        desc.stride_ = [1, 4]
        self.assertTrue(tensor.just_modify(desc))
        np.testing.assert_array_equal(np.asarray(tensor), [[0, 4], [1, 5]])
        self.assertTrue(tensor.set(9))
        np.testing.assert_array_equal(source, [9, 9, 2, 3, 9, 9, 6, 7])
