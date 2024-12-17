import nndeploy._nndeploy_internal as _C
from nndeploy.base import DeviceType
import argparse
from PIL import Image
import numpy as np
from nndeploy.test_utils import createTensorFromNumpy, createNumpyFromTensor

import onnxruntime as ort

# from pytorch.test.distributed._tensor.test_optimizers import input_fn

def get_net(args):
    _C.nndeployFrameworkInit()
    if args.model_type == "onnx":
        interpret = _C.ir.createInterpret(_C.base.ModelType.kModelTypeOnnx)
        assert interpret != None

        interpret.interpret(args.model_path)
        interpret.saveModelToFile("resnet50.json", "resnet50.safetensors")

    elif args.model_type == "default":
        default_interpret = _C.ir.createInterpret(_C.base.ModelType.kModelTypeDefault)
        assert default_interpret != None
        interpret.interpret(args.model_path)

    md = interpret.getModelDesc()
    assert md != None

    net = _C.net.Net()
    net.setModelDesc(md)

    device = DeviceType(args.device, 0)
    net.setDeviceType(device)
    # net.enableOpt(False)
    net.init()
    if args.dump_net_path != None:
        net.dump(args.dump_net_path)

    return net


def read_img(img_path):
    image = Image.open(img_path)
    image = image.resize((224, 224))
    image_array = np.array(image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 归一化
    image_array = (image_array / 255.0 - mean) / std
    print(image_array)
    # 增加批次维度
    image_array = np.expand_dims(image_array, axis=0)
    
    # 从HWC转换为NCHW
    image_array = np.transpose(image_array, (0, 3, 1, 2))
    # print(image_array.shape)
    image_array = image_array.astype(np.float32)
    return image_array


def clsidx_to_label(idx):
    labels = {}
    label_path = "imagenet1000_clsidx2label.txt"
    with open(label_path, "r") as file:
        for line in file:
            index, label = line.strip().split(":")
            labels[int(index)] = label
    return [labels[index] for index in idx]


def predict(net, img, args):
    input_tensor = createTensorFromNumpy(img, args.device)
    input_array = createNumpyFromTensor(input_tensor)
    diff_input = input_array - img
    print("nndeploy输入:", input_array)
    print("onnx输入:", img)
    print("nndeploy和onnx输入的差异:", diff_input)

    input_map = {"data": createTensorFromNumpy(img, args.device)}
    net.setInputs(input_map)
    
    # input_array = np.array()
    # inputs = net.getAllInput()
    # for input in inputs:
    #     if input.getName() == "data":
    #         input_array = createNumpyFromTensor(input)
    #         diff_input = input_array - img
    #         print("nndeploy和onnx输入的差异:", diff_input)
    # print(input_array.shape)
    net.preRun()
    net.run()
    net.postRun()

    output = net.getAllOutput()[0]
    output_array = createNumpyFromTensor(output)
    print(output_array.shape)
    # print("nndeploy推理结果")
    # print(output_array)
    top_5_indices = np.argsort(output_array)[0][::-1][:5]
    infer_result = clsidx_to_label(top_5_indices)
    print("top5 classes:")
    for i in infer_result:
        print(i)
    return output_array, input_array


def main(args):
    net = get_net(args)
    img = read_img(args.image_path)
    # session = ort.InferenceSession((args.model_path)[0],providers=[ 'CPUExecutionProvider'])

    
    # input_name = session.get_inputs()[0].name
    # output_name = session.get_outputs()[0].name

    input_data = img

    # 4. 运行推理
    # 将输入数据作为字典传递给 run 方法
    # outputs = session.run(None, {input_name: input_data})
    # print(outputs[0].shape)
    # top_5_indices = np.argsort(outputs[0])[0][::-1][:5]
    
    # infer_result = clsidx_to_label(top_5_indices)
    # print("top5 classes:")
    # for i in infer_result:
    #     print(i)

 
    # print("onnx推理结果:", outputs[0])
    nndeploy_result, nndeploy_input = predict(net, img, args)

    # 计算nndeploy和onnx推理结果的差异
    # diff = nndeploy_result - outputs[0]
    # print("nndeploy和onnx推理结果的差异:", diff)

    # diff_input = nndeploy_input - input_data
    # print("nndeploy和onnx输入的差异:", diff_input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Inference Arguments")

    # model_type: supports 'onnx' or 'default'
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["onnx", "default"],
        default="default",
        help="Type of the model (onnx or default)",
    )

    # model_path: a list of model paths
    parser.add_argument(
        "--model_path", type=str, nargs="+", help="List of paths to the model files"
    )

    # device: supports 'cpu' or 'Ascend'
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "ascendcl"],
        default="cpu",
        help="Running device (cpu or ascendcl)",
    )

    parser.add_argument(
        "--dump_net_path",
        type=str,
        default="resnet50.dot",
        help="Running device (cpu or Ascend)",
    )

    parser.add_argument("--image_path", type=str, help="Image to be classify")

    # Parse the arguments
    args = parser.parse_args()

    main(args)
