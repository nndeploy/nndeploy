import numpy as np
from unittest import TestCase

from nndeploy.device import create_tensor_from_numpy
from nndeploy.op import functional


def test_cpu_gather_negative_axis_and_indices():
    for index_type in (np.int32, np.int64):
        values = np.arange(12, dtype=np.int16).reshape(3, 4)
        indices = np.array([[3, -2], [1, 0]], dtype=index_type)
        output = functional.gather(create_tensor_from_numpy(values),
                                   create_tensor_from_numpy(indices), axis=-1)
        np.testing.assert_array_equal(np.asarray(output), np.take(values, indices, axis=-1))
        with TestCase().assertRaises(RuntimeError):
            functional.gather(create_tensor_from_numpy(values),
                              create_tensor_from_numpy(np.array([4], dtype=index_type)), axis=1)


def test_cpu_reshape_materializes_independent_output():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = functional.reshape(create_tensor_from_numpy(values),
                                create_tensor_from_numpy(np.array([2, -1], dtype=np.int64)))
    np.testing.assert_array_equal(np.asarray(output), values.reshape(2, 6))
