
import nndeploy._nndeploy_internal as _C


import nndeploy.base
import nndeploy.device   

from .op_param import OpType, register_op_param_creator, create_op_param
from .ir import OpDesc, ValueDesc, ModelDesc


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
    
    def create_interpret(self, type, model_desc=None, is_external=False):
        # must be self.op
        raise NotImplementedError("base class Interpret must implement create_interpret method")
    
    def create_interpret_shared_ptr(self, type, model_desc=None, is_external=False):
        raise NotImplementedError("base class Interpret don't implement create_interpret_shared_ptr method")
    
  
def register_interpret_creator(type, creator):
    return _C.ir.register_interpret_creator(type, creator)


def create_interpret(type, model_desc=None, is_external=False):
    return _C.ir.create_interpret(type, model_desc, is_external)


