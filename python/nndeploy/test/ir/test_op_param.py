import nndeploy._nndeploy_internal as _C

from nndeploy.ir import OpType, register_op_param_creator, create_op_param


# python3 nndeploy/test/ir/test_op_param.py


if __name__ == "__main__":
    print(OpType.from_name("Net"))
    print(OpType.from_name("Add"))
    print(OpType.from_name("Mul"))
    
    try:
        print(OpType.from_name("InvalidOpType"))
    except ValueError as e:
        print(f"catch error: {e}")

    class MyOpParamCreator(_C.ir.OpParamCreator):
        def __init__(self):
            super().__init__()

        def create_op_param(self, op_type: _C.ir.OpType):
            if op_type == _C.ir.OpType.BatchNormalization:
                return _C.base.Param()
            elif op_type == _C.ir.OpType.Net:
                return _C.ir.BatchNormalizationParam()
            else:
                raise ValueError(f"not supported op type: {op_type}")
    
    creator = MyOpParamCreator()
    register_op_param_creator(OpType.Net, creator)
    bn_param = create_op_param(OpType.Net)
    print(type(bn_param))