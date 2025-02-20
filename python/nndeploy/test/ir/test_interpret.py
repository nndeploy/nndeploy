import nndeploy._nndeploy_internal as _C

import nndeploy.base

from nndeploy.ir import OpType, register_op_param_creator, create_op_param, OpDesc, ValueDesc, ModelDesc
from nndeploy.ir import Interpret, InterpretCreator, register_interpret_creator, create_interpret


# python3 nndeploy/test/ir/test_interpret.py


class MyInterpret(_C.ir.Interpret):
    def __init__(self):
        super().__init__()
    def interpret(self, model_value, input=[]):
        print("interpret")
        return _C.base.Status()
    

class MyInterpretCreator(_C.ir.InterpretCreator):
    def __init__(self):
        super().__init__()
    def create_interpret(self, model_type: nndeploy.base.ModelType, model_desc=None, is_external=False) -> _C.ir.Interpret:
        if model_type == nndeploy.base.ModelType.TorchPth:
            return MyInterpret()
        else:
            return MyInterpret()


if __name__ == "__main__":
    interpret = create_interpret(nndeploy.base.ModelType.Default)
    print(interpret)
    
    creator = MyInterpretCreator()
    register_interpret_creator(nndeploy.base.ModelType.TorchPth, creator)
    test_interpret = MyInterpret()
    print(type(test_interpret))

    # create_interpret(nndeploy.base.ModelType.TorchPth)
    pytorch_interpret = create_interpret(nndeploy.base.ModelType.TorchPth)
    print(type(pytorch_interpret))