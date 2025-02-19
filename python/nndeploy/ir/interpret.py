
import nndeploy._nndeploy_internal as _C


import nndeploy.base
# import nndeploy.device

# import op_param

# import ir


# python3 nndeploy/ir/interpret.py


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
        
    def create_interpret_cpp(self, type, model_desc=None, is_external=False):
        raise NotImplementedError("base class Interpret does not implement interpret method")
    
    def create_interpret(self, type, model_desc=None, is_external=False):
        raise NotImplementedError("base class Interpret does not implement interpret method")
    
  
def register_interpret_creator(type, creator):
    return _C.ir.register_interpret_creator(type, creator)


def create_interpret(type):
    return _C.ir.create_interpret(type)


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


