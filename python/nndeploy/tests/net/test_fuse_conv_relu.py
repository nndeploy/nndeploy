import unittest
import numpy as np
import torch
import nndeploy

from nndeploy.test_utils import createTensorFromNumpy
from nndeploy.net import build_model


class ConvReluNet(nndeploy.net.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = nndeploy.op.Conv(32, 3, [3, 3])
        self.relu1 = nndeploy.op.Relu()

    @build_model
    def construct(self):
        data_type = nndeploy._C.base.DataType()
        data_type.code_ = nndeploy._C.base.DataTypeCode.kDataTypeCodeFp
        data = nndeploy._C.op.makeInput(
            self.model_desc,
            "input",
            data_type,
            [1,3,32,32]
        )
        data = self.conv1(data)
        data = self.relu1(data)
        return data


test_net = ConvReluNet()
test_net.construct()


class TestFuseConvRelu(unittest.TestCase):

    def test_fuse(self):

        pass
