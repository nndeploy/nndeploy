
#include "flag.h"

namespace nndeploy {
namespace demo {

DEFINE_bool(usage, false, "usage");

DEFINE_string(name, "", "graph name");

DEFINE_string(inference_type, "", "inference_type");

DEFINE_string(device_type, "", "device_type");

DEFINE_string(model_type, "", "model_type");

DEFINE_bool(is_path, true, "is_path");

DEFINE_string(model_value, "", "model_value");

DEFINE_string(encrypt_type, "", "encrypt_type");

DEFINE_string(license, "", "license");

DEFINE_string(codec_type, "", "codec_type");

DEFINE_string(codec_flag, "", "codec_flag");

DEFINE_string(input_path, "", "input_path");

DEFINE_string(output_path, "", "output_path");

DEFINE_int32(num_thread, 4, "num_thread");

DEFINE_int32(gpu_tune_kernel, -1, "gpu_tune_kernel");

DEFINE_string(share_memory_mode, "", "share_memory_mode");

DEFINE_string(precision_type, "", "precision_type");

DEFINE_string(power_type, "", "power_type");

DEFINE_string(parallel_type, "", "parallel_type");

DEFINE_string(cache_path, "", "cache_path");

DEFINE_string(library_path, "", "library_path");

void showUsage() {
  std::cout << "Usage: " << std::endl;
  std::cout << "  --name: graph name, eg: yolo_v5_v6_v8" << std::endl;
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
  std::cout << "  --license: license, eg: path/to/lincese or license string"
            << std::endl;
  std::cout << "  --codec_type: codec_type, eg: kCodecTypeOpenCV" << std::endl;
  std::cout << "  --codec_flag: codec_type, eg: kCodecFlagImage" << std::endl;
  std::cout << "  --input_path: input_path, eg: "
               "path/nndeploy_resource/detect/input.jpg"
            << std::endl;
  std::cout << "  --output_path: output_path, eg: "
               "path/nndeploy_resource/detect/output/output.jpg"
            << std::endl;
  std::cout << "  --num_thread: num_thread, eg: 4" << std::endl;
  std::cout << "  --gpu_tune_kernel: gpu_tune_kernel, eg: 1" << std::endl;
  std::cout
      << "  --share_memory_mode: share_memory_mode, eg: kShareMemoryTypeNoShare"
      << std::endl;
  std::cout << "  --precision_type: precision_type, eg: kPrecisionTypeFp32"
            << std::endl;
  std::cout << "  --power_type: power_type, eg: kPowerTypeNormal" << std::endl;
  std::cout << "  --parallel_type: parallel_type, eg: kParallelTypeSequential"
            << std::endl;
  std::cout << "  --cache_path: cache_path, eg: "
               "path/to/model_0.trt"
            << std::endl;
  std::cout << "  --library_path: library_path, eg: "
               "path/to/opencl.so"
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
    model_value.emplace_back(model_value_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = model_value_str.find(",", pos1);
  }
  model_value.emplace_back(model_value_str.substr(pos1));
  return model_value;
}

base::EncryptType getEncryptType() {
  return base::stringToEncryptType(FLAGS_encrypt_type);
}

std::string getLicense() { return FLAGS_license; }

std::string getInputPath() { return FLAGS_input_path; }
std::string getOutputPath() { return FLAGS_output_path; }

int getNumThread() { return FLAGS_num_thread; }
int getGpuTuneKernel() { return FLAGS_gpu_tune_kernel; }
base::ShareMemoryType getShareMemoryType() {
  return base::stringToShareMemoryType(FLAGS_share_memory_mode);
}
base::PrecisionType getPrecisionType() {
  return base::stringToPrecisionType(FLAGS_precision_type);
}
base::PowerType getPowerType() {
  return base::stringToPowerType(FLAGS_power_type);
}
base::ParallelType getParallelType() {
  return base::stringToParallelType(FLAGS_parallel_type);
}
base::CodecType getCodecType() {
  return base::stringToCodecType(FLAGS_codec_type);
}
base::CodecFlag getCodecFlag() {
  return base::stringToCodecFlag(FLAGS_codec_flag);
}
std::vector<std::string> getCachePath() {
  std::vector<std::string> cache_path;
  std::string cache_path_str = FLAGS_cache_path;
  std::string::size_type pos1, pos2;
  pos2 = cache_path_str.find(",");
  pos1 = 0;
  while (std::string::npos != pos2) {
    cache_path.emplace_back(cache_path_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = cache_path_str.find(",", pos1);
  }
  cache_path.emplace_back(cache_path_str.substr(pos1));
  return cache_path;
}
std::vector<std::string> getLibraryPath() {
  std::vector<std::string> library_path;
  std::string library_path_str = FLAGS_library_path;
  std::string::size_type pos1, pos2;
  pos2 = library_path_str.find(",");
  pos1 = 0;
  while (std::string::npos != pos2) {
    library_path.emplace_back(library_path_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = library_path_str.find(",", pos1);
  }
  library_path.emplace_back(library_path_str.substr(pos1));
  return library_path;
}

std::vector<std::string> getAllFileFromDir(std::string dir_path) {
  std::vector<std::string> allFile = {};
  if (nndeploy::base::isDirectory(dir_path)) {
    nndeploy::base::glob(dir_path, "", allFile);
  }
  return allFile;
}

}  // namespace demo
}  // namespace nndeploy
