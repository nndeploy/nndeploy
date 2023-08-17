
#include "flag.h"

namespace nndeploy {
namespace demo {

DEFINE_bool(usage, false, "usage");

DEFINE_string(name, "", "pipeline name");

DEFINE_string(inference_type, "", "inference_type");

DEFINE_string(device_type, "", "device_type");

DEFINE_string(model_type, "", "model_type");

DEFINE_bool(is_path, true, "is_path");

DEFINE_string(model_value, "", "model_value");

DEFINE_string(encrypt_type, "", "encrypt_type");

DEFINE_string(input_type, "", "input_type");

DEFINE_string(input_path, "", "input_path");

DEFINE_string(output_path, "", "output_path");

void showUsage() {
  std::cout << "Usage: " << std::endl;
  std::cout << "  --name: pipeline name, eg: yolo_v5_v6_v8" << std::endl;
  std::cout << "  --inference_type: inference_type, eg: kInferenceTypeOpenVino"
            << std::endl;
  std::cout << "  --device_type: device_type, eg: kDeviceTypeCodeX86:0"
            << std::endl;
  std::cout << "  --model_type: model_type, eg: kModelTypeOnnx" << std::endl;
  std::cout << "  --is_path: is_path" << std::endl;
  std::cout << "  --model_value: model_value, eg: "
               "path/nndeploy_resource/detect/yolo/yolov5s.onnx"
            << std::endl;
  std::cout << "  --encrypt_type: encrypt_type, eg: kEncryptTypeNone"
            << std::endl;
  std::cout << "  --input_type: input_type, eg: kInputTypeImage" << std::endl;
  std::cout << "  --input_path: input_path, eg: "
               "path/nndeploy_resource/detect/input.jpg"
            << std::endl;
  std::cout << "  --output_path: output_path, eg: "
               "path/nndeploy_resource/detect/output/output.jpg"
            << std::endl;
}

std::string getName() { return FLAGS_name; }

base::InferenceType getInferenceType() {
  return base::stringToInferenceType(FLAGS_inference_type);
}

base::DeviceType getDeviceType() {
  return base::stringToDeviceType(FLAGS_device_type);
}

base::ModelType getModelType() {
  return base::stringToModelType(FLAGS_model_type);
}

bool isPath() { return FLAGS_is_path; }

std::vector<std::string> getModelValue() {
  std::vector<std::string> model_value;
  std::string model_value_str = FLAGS_model_value;
  std::string::size_type pos1, pos2;
  pos2 = model_value_str.find(",");
  pos1 = 0;
  while (std::string::npos != pos2) {
    model_value.push_back(model_value_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = model_value_str.find(",", pos1);
  }
  model_value.push_back(model_value_str.substr(pos1));
  return model_value;
}

base::EncryptType getEncryptType() {
  return base::stringToEncryptType(FLAGS_encrypt_type);
}

InputType getInputType() {
  if (FLAGS_input_type == "kInputTypeImage") {
    return kInputTypeImage;
  } else if (FLAGS_input_type == "kInputTypeVideo") {
    return kInputTypeVideo;
  } else if (FLAGS_input_type == "kInputTypeCamera") {
    return kInputTypeCamera;
  } else if (FLAGS_input_type == "kDeviceTypeOther") {
    return kDeviceTypeOther;
  } else {
    return kInputTypeImage;
  }
}
std::string getInputPath() { return FLAGS_input_path; }
std::string getOutputPath() { return FLAGS_output_path; }

}  // namespace demo
}  // namespace nndeploy
