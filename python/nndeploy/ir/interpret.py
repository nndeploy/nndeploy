
import nndeploy._nndeploy_internal as _C

import numpy as np
import nndeploy.base
# import nndeploy.device

# from .op_param import name_to_op_type, op_type_to_name, string_to_op_type, string_to_op_type
# from .op_param import OpType, OpParamCreator, register_op_param_creator, create_op_param, OpParam

# from .ir import OpDesc, ValueDesc, ModelDesc


class Interpret(_C.ir.Interpret):
    def __init__(self):
        super().__init__()
        
    def interpret(self, model_value, input=[]):
        raise NotImplementedError("base class Interpret does not implement interpret method")
    
    def dump(self, file_path):
        with open(file_path, "w") as oss:
            super().dump(oss)
    
    def save_model(self, structure_stream, st_ptr):
        return super().save_model(structure_stream, st_ptr)
    
    def save_model_to_file(self, structure_file_path, weight_file_path):
        return super().save_model_to_file(structure_file_path, weight_file_path)
    
    def get_model_desc(self):
        return super().get_model_desc()


class InterpretCreator(_C.ir.InterpretCreator):
    def __init__(self):
        super().__init__()
        
    def createInterpret(self, type):
        raise NotImplementedError("base class Interpret does not implement interpret method")
    
  
def register_interpret_creator(type, creator):
    return _C.ir.register_interpret_creator(type, creator)


def createInterpret(type):
    return _C.ir.createInterpret(type)


class MyInterpret(_C.ir.Interpret):
    def __init__(self):
        super().__init__()
    def interpret(self, model_value, input=[]):
        print("interpret")
        return _C.base.Status()
    
class MyInterpretCreator(_C.ir.InterpretCreator):
    def __init__(self):
        super().__init__()
    def createInterpret(self, model_type: nndeploy.base.ModelType) -> _C.ir.Interpret:
        import ctypes
        if model_type == nndeploy.base.ModelType.TorchPth:
            print("create torch interpret")
            # value = _C.ir.DefaultInterpret.__new__(_C.ir.DefaultInterpret)
            value = _C.ir.DefaultInterpret()
            print(value)
            return MyInterpret()
            return value
            # value.interpret = MyInterpret.interpret
            # 获取Python对象的内存地址
            # address = id(value)
            # 将地址转换为指针
            # ptr = ctypes.cast(address, ctypes.POINTER(ctypes.c_void_p))
            ptr = _C.ir.convert_to_interpret_ptr(value)
            print(ptr)
            # return ptr
        else:
            return MyInterpret()

if __name__ == "__main__":
    interpret = createInterpret(nndeploy.base.ModelType.Default)
    print(interpret)
    
    # try:
    #     print(createInterpret(nndeploy.base.ModelType.Invalid))
    # except RuntimeError as e:
    #     print(f"catch error: {e}")

    
    
    creator = MyInterpretCreator()
    print("hello")
    register_interpret_creator(nndeploy.base.ModelType.TorchPth, creator)
    print("hello")
    test_interpret = MyInterpret()
    print(type(test_interpret))

    # createInterpret(nndeploy.base.ModelType.TorchPth)
    pytorch_interpret = createInterpret(nndeploy.base.ModelType.TorchPth)
    print("hello")
    print(type(pytorch_interpret))


