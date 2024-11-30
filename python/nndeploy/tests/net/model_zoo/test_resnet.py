import nndeploy._nndeploy_internal as _C
from nndeploy.base import DeviceType


def get_net():
    _C.nndeployFrameworkInit()
    onnx_interpret = _C.ir.createInterpret(_C.base.ModelType.kModelTypeOnnx)
    assert onnx_interpret != None

    onnx_interpret.interpret(
        [
            "/data/sjx/code/nndeploy_resource/nndeploy/model_zoo/classfication/resnet50-v1-7.sim.onnx"
        ]
    )
    onnx_interpret.saveModelToFile("resnet50.json", "resnet50.safetensors")

    default_interpret = _C.ir.createInterpret(_C.base.ModelType.kModelTypeDefault)
    default_interpret.interpret(["resnet50.json", "resnet50.safetensors"])

    md = default_interpret.getModelDesc()
    assert md != None

    net = _C.net.Net()
    net.setModelDesc(md)

    device = DeviceType("cpu", 0)
    net.setDeviceType(device)

    net.init()
    net.dump("resnet50.dot")
    # net.deinit()

    _C.nndeployFrameworkDeinit()
    return net


net = get_net()

inputs = net.getAllInput()
print(inputs[0].shape)

# net.prerun()
net.run()
# net.postrun()
