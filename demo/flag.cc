
#include "flag.h"

#include "nndeploy/base/dlopen.h"

namespace nndeploy {
namespace demo {

DEFINE_bool(usage, false, "usage");

DEFINE_bool(remove_in_out_node, false, "remove_in_out_node");
DEFINE_string(task_id, "", "task_id");
DEFINE_string(json_file, "", "json_file");
DEFINE_string(plugin, "", "plugin");
DEFINE_string(node_param, "", "node_param");
DEFINE_bool(dump, false, "dump");
DEFINE_bool(debug, false, "debug");
DEFINE_bool(time_profile, true, "time_profile");

DEFINE_string(name, "", "graph name");

DEFINE_string(inference_type, "", "inference_type");

DEFINE_string(device_type, "", "device_type");

DEFINE_string(model_type, "", "model_type");

DEFINE_bool(is_path, true, "is_path");

DEFINE_string(model_value, "", "model_value");

DEFINE_string(classifier_model_value, "", "classifier_model_value");

DEFINE_string(detector_model_value, "", "detector_model_value");

DEFINE_string(recognizer_model_value, "", "recognizer_model_value");

DEFINE_string(character_txt_value, "", "character_txt_value");

DEFINE_string(encrypt_type, "", "encrypt_type");

DEFINE_string(license, "", "license");

DEFINE_string(codec_type, "", "codec_type");

DEFINE_string(codec_flag, "", "codec_flag");

DEFINE_string(config_path, "", "config_path");

DEFINE_string(input_path, "", "input_path");

DEFINE_string(output_path, "", "output_path");

DEFINE_int32(num_thread, 4, "num_thread");

DEFINE_int32(gpu_tune_kernel, -1, "gpu_tune_kernel");

DEFINE_string(share_memory_mode, "", "share_memory_mode");

DEFINE_string(precision_type, "", "precision_type");

DEFINE_string(power_type, "", "power_type");

DEFINE_string(parallel_type, "kParallelTypeSequential", "parallel_type");

DEFINE_string(cache_path, "", "cache_path");

DEFINE_string(library_path, "", "library_path");

DEFINE_string(model_inputs, "", "model_inputs");

DEFINE_string(model_outputs, "", "model_outputs");

DEFINE_string(classifier_model_inputs, "", "classifier_model_inputs");
DEFINE_string(recognizer_model_inputs, "", "recognizer_model_inputs");
DEFINE_string(detector_model_inputs, "", "detector_model_inputs");

DEFINE_string(classifier_model_outputs, "", "classifier_model_outputs");
DEFINE_string(recognizer_model_outputs, "", "recognizer_model_outputs");
DEFINE_string(detector_model_outputs, "", "detector_model_outputs");

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
  std::cout << "  --recognizer_model_value: recognizer_model_value, eg: "
               "path/nndeploy_resource/detect/yolo/yolov5s.onnx"
            << std::endl;
  std::cout << "  --classifier_model_value: classifier_model_value, eg: "
               "path/nndeploy_resource/detect/yolo/yolov5s.onnx"
            << std::endl;
  std::cout << "  --detector_model_value: detector_model_value, eg: "
               "path/nndeploy_resource/detect/yolo/yolov5s.onnx"
            << std::endl;
  std::cout << "  --encrypt_type: encrypt_type, eg: kEncryptTypeNone"
            << std::endl;
  std::cout << "  --license: license, eg: path/to/lincese or license string"
            << std::endl;
  std::cout << "  --codec_type: codec_type, eg: kCodecTypeOpenCV" << std::endl;
  std::cout << "  --codec_flag: codec_type, eg: kCodecFlagImage" << std::endl;
  std::cout << "  --config_path: config_path, eg: "
               "path/nndeploy_resource/llm/config.jpg"
            << std::endl;
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
  std::cout << "  --model_inputs: model_inputs, eg: "
               "input_0,input_1"
            << std::endl;
  std::cout << "  --model_outputs: model_outputs, eg: "
               "output_0,output_1"
            << std::endl;
}

std::string getName() { return FLAGS_name; }

bool removeInOutNode() { return FLAGS_remove_in_out_node; }
std::string getTaskId() { return FLAGS_task_id; }
std::string getJsonFile() { return FLAGS_json_file; }
std::vector<std::string> getPlugin() {
  std::vector<std::string> plugin;
  std::string plugin_str = FLAGS_plugin;
  if (plugin_str != "") {
    std::string::size_type pos1, pos2;
    pos2 = plugin_str.find(",");
    pos1 = 0;
    while (std::string::npos != pos2) {
      plugin.emplace_back(plugin_str.substr(pos1, pos2 - pos1));
      pos1 = pos2 + 1;
      pos2 = plugin_str.find(",", pos1);
    }
    plugin.emplace_back(plugin_str.substr(pos1));
  }
  return plugin;
}
bool loadPlugin() {
  std::vector<std::string> plugin = getPlugin();
  for (const auto& plugin_item : plugin) {
    // NNDEPLOY_LOGI("load plugin: %s", plugin_item.c_str());
    bool success = base::loadLibraryFromPath(plugin_item, true);
    if (!success) {
      NNDEPLOY_LOGE("load plugin failed: %s", plugin_item.c_str());
      return false;
    }
  }
  return true;
}
std::map<std::string, std::map<std::string, std::string>> getNodeParam() {
  std::map<std::string, std::map<std::string, std::string>> node_param;
  std::string node_param_str = FLAGS_node_param;

  if (node_param_str.empty()) {
    return node_param;
  }

  // 先按逗号分割每个节点参数项
  std::vector<std::string> param_items;
  std::string::size_type pos1, pos2;
  pos2 = node_param_str.find(",");
  pos1 = 0;

  while (std::string::npos != pos2) {
    param_items.emplace_back(node_param_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = node_param_str.find(",", pos1);
  }
  param_items.emplace_back(node_param_str.substr(pos1));

  // 再按冒号分割每个参数项：node_name:param_key:param_value
  for (const auto& param_item : param_items) {
    std::string::size_type colon1 = param_item.find(":");
    if (colon1 != std::string::npos) {
      std::string::size_type colon2 = param_item.find(":", colon1 + 1);
      if (colon2 != std::string::npos) {
        std::string node_name = param_item.substr(0, colon1);
        std::string param_key =
            param_item.substr(colon1 + 1, colon2 - colon1 - 1);
        std::string param_value = param_item.substr(colon2 + 1);

        node_param[node_name][param_key] = param_value;
      }
    }
  }

  return node_param;
}
bool dump() { return FLAGS_dump; }
bool debug() { return FLAGS_debug; }
bool timeProfile() { return FLAGS_time_profile; }

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

std::string getConfigPath() { return FLAGS_config_path; }

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

std::vector<std::string> getModelInputs() {
  std::vector<std::string> model_inputs;
  std::string model_inputs_str = FLAGS_model_inputs;
  std::string::size_type pos1, pos2;
  pos2 = model_inputs_str.find(",");
  pos1 = 0;
  while (std::string::npos != pos2) {
    model_inputs.emplace_back(model_inputs_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = model_inputs_str.find(",", pos1);
  }
  model_inputs.emplace_back(model_inputs_str.substr(pos1));
  return model_inputs;
}

std::vector<std::string> getModelOutputs() {
  std::vector<std::string> model_outputs;
  std::string model_outputs_str = FLAGS_model_outputs;
  std::string::size_type pos1, pos2;
  pos2 = model_outputs_str.find(",");
  pos1 = 0;
  while (std::string::npos != pos2) {
    model_outputs.emplace_back(model_outputs_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = model_outputs_str.find(",", pos1);
  }
  model_outputs.emplace_back(model_outputs_str.substr(pos1));
  return model_outputs;
}

std::vector<std::string> getClassifierModelValue() {
  std::vector<std::string> model_value;
  std::string model_value_str = FLAGS_classifier_model_value;
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

std::vector<std::string> getRecognizerModelValue() {
  std::vector<std::string> model_value;
  std::string model_value_str = FLAGS_recognizer_model_value;
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

std::string getCharacterTxtValue() { return FLAGS_character_txt_value; }

std::vector<std::string> getDetectorModelValue() {
  std::vector<std::string> model_value;
  std::string model_value_str = FLAGS_detector_model_value;
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

std::vector<std::string> getClassifierModelInputs() {
  std::vector<std::string> model_inputs;
  std::string model_inputs_str = FLAGS_classifier_model_inputs;
  std::string::size_type pos1, pos2;
  pos2 = model_inputs_str.find(",");
  pos1 = 0;
  while (std::string::npos != pos2) {
    model_inputs.emplace_back(model_inputs_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = model_inputs_str.find(",", pos1);
  }
  model_inputs.emplace_back(model_inputs_str.substr(pos1));
  return model_inputs;
}

std::vector<std::string> getClassifierModelOutputs() {
  std::vector<std::string> model_outputs;
  std::string model_outputs_str = FLAGS_classifier_model_outputs;
  std::string::size_type pos1, pos2;
  pos2 = model_outputs_str.find(",");
  pos1 = 0;
  while (std::string::npos != pos2) {
    model_outputs.emplace_back(model_outputs_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = model_outputs_str.find(",", pos1);
  }
  model_outputs.emplace_back(model_outputs_str.substr(pos1));
  return model_outputs;
}

std::vector<std::string> getRecognizerModelInputs() {
  std::vector<std::string> model_inputs;
  std::string model_inputs_str = FLAGS_recognizer_model_inputs;
  std::string::size_type pos1, pos2;
  pos2 = model_inputs_str.find(",");
  pos1 = 0;
  while (std::string::npos != pos2) {
    model_inputs.emplace_back(model_inputs_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = model_inputs_str.find(",", pos1);
  }
  model_inputs.emplace_back(model_inputs_str.substr(pos1));
  return model_inputs;
}

std::vector<std::string> getRecognizerModelOutputs() {
  std::vector<std::string> model_outputs;
  std::string model_outputs_str = FLAGS_recognizer_model_outputs;
  std::string::size_type pos1, pos2;
  pos2 = model_outputs_str.find(",");
  pos1 = 0;
  while (std::string::npos != pos2) {
    model_outputs.emplace_back(model_outputs_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = model_outputs_str.find(",", pos1);
  }
  model_outputs.emplace_back(model_outputs_str.substr(pos1));
  return model_outputs;
}

std::vector<std::string> getDetectorModelInputs() {
  std::vector<std::string> model_inputs;
  std::string model_inputs_str = FLAGS_detector_model_inputs;
  std::string::size_type pos1, pos2;
  pos2 = model_inputs_str.find(",");
  pos1 = 0;
  while (std::string::npos != pos2) {
    model_inputs.emplace_back(model_inputs_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = model_inputs_str.find(",", pos1);
  }
  model_inputs.emplace_back(model_inputs_str.substr(pos1));
  return model_inputs;
}

std::vector<std::string> getDetectorModelOutputs() {
  std::vector<std::string> model_outputs;
  std::string model_outputs_str = FLAGS_detector_model_outputs;
  std::string::size_type pos1, pos2;
  pos2 = model_outputs_str.find(",");
  pos1 = 0;
  while (std::string::npos != pos2) {
    model_outputs.emplace_back(model_outputs_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = model_outputs_str.find(",", pos1);
  }
  model_outputs.emplace_back(model_outputs_str.substr(pos1));
  return model_outputs;
}

}  // namespace demo
}  // namespace nndeploy
