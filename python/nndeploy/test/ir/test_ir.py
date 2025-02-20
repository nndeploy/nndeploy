import nndeploy._nndeploy_internal as _C

import nndeploy.base

from nndeploy.ir import OpType, register_op_param_creator, create_op_param, OpDesc, ValueDesc, ModelDesc


# python3 nndeploy/test/ir/test_ir.py


if __name__ == "__main__":
    model = ModelDesc()
    model.name = "test_conv"
    model.inputs = [ValueDesc("input", nndeploy.base.DataType(nndeploy.base.DataTypeCode.Fp, 32, 1), [1, 3, 224, 224])]
    model.outputs = [ValueDesc("output", nndeploy.base.DataType(nndeploy.base.DataTypeCode.Fp, 32, 1), [1, 64, 112, 112])]
    
    conv1 = OpDesc("Conv", OpType.Conv)
    conv1.inputs = ["input"]
    conv1.outputs = ["conv1_out"]
    model.op_descs.append(conv1)
    
    conv2 = OpDesc("Conv", OpType.Conv) 
    conv2.inputs = ["conv1_out"]
    conv2.outputs = ["output"]
    model.op_descs.append(conv2)
    
    print(model)
    