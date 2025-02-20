
import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.ir
from nndeploy.op import Op, OpCreator, register_op_creator, create_op


# python3 nndeploy/test/op/test_op.py

class MyOp(_C.op.Op):
    def __init__(self):
        super().__init__()
        self.test = "test"

    def run(self):
        print("MyOp run")
        return nndeploy.base.Status()

    def __str__(self):
        return f"name: {self.get_name()}, op_type: {self.get_op_type()}, device_type: {self.get_device_type()}, test: {self.test}"


class MyOpCreator(_C.op.OpCreator):
    def __init__(self):
        super().__init__()

    def create_op(self, device_type: nndeploy.base.DeviceType, name: str, op_type: nndeploy.ir.OpType, inputs: list[str], outputs: list[str]):
        if op_type == nndeploy.ir.OpType.kOpTypeNone and device_type.code_ == nndeploy.base.DeviceTypeCode.cpu:
            op = MyOp()
            op.set_device_type(device_type)
            op.set_name(name) 
            op.set_op_type(op_type)
            op.set_all_input_name(inputs)
            op.set_all_output_name(outputs)
            return op
        else:
            return None


if __name__ == "__main__":
    print("Op")
    op = Op()
    print(op)
    print("--------------------------------")
    creator = MyOpCreator()
    register_op_creator(nndeploy.base.DeviceTypeCode.cpu, nndeploy.ir.OpType.kOpTypeNone, creator)
    op = create_op(nndeploy.base.DeviceType(nndeploy.base.DeviceTypeCode.cpu), "kOpTypeNone", nndeploy.ir.OpType.kOpTypeNone, [], [], None)
    print(op)
    print("--------------------------------")