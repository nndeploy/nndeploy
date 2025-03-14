import nndeploy._nndeploy_internal as _C
from nndeploy.base import DeviceType
import argparse
from PIL import Image
import numpy as np
from nndeploy.test.test_util import create_tensor_from_numpy, create_numpy_from_tensor

import onnxruntime as ort


def get_net(args):
    _C.nndeployFrameworkInit()
    if args.model_type == "onnx":
        interpret = _C.ir.createInterpret(_C.base.ModelType.kModelTypeOnnx)
        assert interpret != None

        interpret.interpret(args.model_path)
        interpret.save_model_to_file("resnet50.json", "resnet50.safetensors")

    elif args.model_type == "default":
        default_interpret = _C.ir.createInterpret(_C.base.ModelType.kModelTypeDefault)
        assert default_interpret != None
        interpret.interpret(args.model_path)

    md = interpret.get_model_desc()
    assert md != None

    net = _C.net.Net()
    net.setModelDesc(md)

    device = DeviceType(args.device, 0)
    net.setDeviceType(device)

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

    # 增加批次维度
    image_array = np.expand_dims(image_array, axis=0)

    # 从HWC转换为NCHW
    image_array = np.transpose(image_array, (0, 3, 1, 2))
    print(image_array.shape)
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
    input_map = {"data": create_tensor_from_numpy(img, args.device)}
    net.setInputs(input_map)
    net.preRun()
    net.run()
    net.postRun()

    output = net.getAllOutput()[0]
    output_array = create_numpy_from_tensor(output)
    print("nndeploy推理结果")
    print(output_array)
    top_5_indices = np.argsort(output_array)[0][::-1][:5]
    infer_result = clsidx_to_label(top_5_indices)
    print("top5 classes:")
    for i in infer_result:
        print(i)


def main(args):
    net = get_net(args)
    img = read_img(args.image_path)
    session = ort.InferenceSession(
        (args.model_path)[0], providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_data = img

    # 4. 运行推理
    # 将输入数据作为字典传递给 run 方法
    outputs = session.run(None, {input_name: input_data})

    top_5_indices = np.argsort(outputs[0])[0][::-1][:5]

    infer_result = clsidx_to_label(top_5_indices)
    print("top5 classes:")
    for i in infer_result:
        print(i)

    print("onnx推理结果:", outputs[0])
    predict(net, img, args)


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
