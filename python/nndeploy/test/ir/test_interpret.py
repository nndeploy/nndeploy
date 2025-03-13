import nndeploy._nndeploy_internal as _C

import nndeploy.base

from nndeploy.ir import OpType, register_op_param_creator, create_op_param, OpDesc, ValueDesc, ModelDesc
from nndeploy.ir import Interpret, InterpretCreator, register_interpret_creator, create_interpret


# python3 nndeploy/test/ir/test_interpret.py
# 创建一个全局字典保存所有创建的解释器实例
_interpret_instances = {}

class MyInterpret(_C.ir.Interpret):  # 正确继承自C++的Interpret类
    def __init__(self):
        super().__init__()
    
    def interpret(self, model_value, input=[]):
        print("interpret")
        return _C.base.Status()  # 返回正确的Status类型
    

class MyInterpretCreator(_C.ir.InterpretCreator):  # 正确继承自C++的InterpretCreator类
    def __init__(self):
        super().__init__()
    
    def create_interpret(self, model_type: nndeploy.base.ModelType, model_desc=None, is_external=False):
        print("create_interpret")
        # 创建实例并保存到全局字典中
        temp = MyInterpret()
        # 生成唯一ID并存储实例
        instance_id = id(temp)
        _interpret_instances[instance_id] = temp
        print(f"Created instance: {temp}, id: {instance_id}")
        return temp
      
      
    def create_interpret_shared_ptr(self, model_type: nndeploy.base.ModelType, model_desc=None, is_external=False):
        print("create_interpret_shared_ptr")
        temp = MyInterpret()  # 返回正确类型的对象
        print(temp)
        print(type(temp))
        print("--------------------------------")
        return temp


# 保持创建器对象的全局引用
_creators = {}

if __name__ == "__main__":
    print("--------------------------------")
    interpret = create_interpret(nndeploy.base.ModelType.Default)
    print(interpret)
    print("--------------------------------")
    creator = MyInterpretCreator()
    _creators[nndeploy.base.ModelType.TorchPth] = creator
    register_interpret_creator(nndeploy.base.ModelType.TorchPth, creator)
    test_interpret = MyInterpret()
    test_interpret = creator.create_interpret(nndeploy.base.ModelType.TorchPth)
    print("类型:", type(test_interpret))
    print("父类:", type(test_interpret).__bases__)
    print("所有父类:", type(test_interpret).__mro__)
    
    print("--------------------------------")
    print(nndeploy.base.ModelType.TorchPth)
    pytorch_interpret = create_interpret(nndeploy.base.ModelType.TorchPth)
    print(nndeploy.base.ModelType.TorchPth)
    print(type(pytorch_interpret)) 