/**
 * nndeploy Onnx Converter Demo:
 * Implementation of onnx model convert to default model format(JSON +
 * safetensors)
 */

// #include <experimental/filesystem>
#include "gflags/gflags.h"
#include "nndeploy/ir/default_interpret.h"
#include "nndeploy/ir/interpret.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/net/net.h"

using namespace nndeploy;

/**
 * @brief Construct a new define string object
 * @note
 * --src_model
 *  "path/to/src_model"
 */
DEFINE_string(src_model, "", "src_model");

/**
 * @brief Construct a new define string object
 * @note
 * --dst_model
 *  "path/to/dst_model.json,path/to/dst_model.safetensors"
 */
DEFINE_string(dst_model, "", "dst_model");

/**
 * @brief Construct a new define string object
 * @note
 * --input_shape
 *  "input_0:1,3,224,224;input_1:1,1,112,112"
 */
DEFINE_string(input_shape, "", "input_shape");

/**
 * @brief Construct a new define string object
 * @note
 * --input_data_type
 *  "input_0:kDataTypeCodeFp,32,1;input_1:kDataTypeCodeFp,32,1"
 */
DEFINE_string(input_data_type, "", "input_data_type");

std::string getSrcModel() { return FLAGS_src_model; }

std::vector<std::string> getDstModel() {
  std::vector<std::string> dst_model;
  std::string dst_model_str = FLAGS_dst_model;
  if (dst_model_str.empty()) {
    dst_model.emplace_back(FLAGS_src_model + ".json");
    dst_model.emplace_back(FLAGS_src_model + ".safetensors");
    return dst_model;
  }
  std::string::size_type pos1, pos2;
  pos2 = dst_model_str.find(",");
  pos1 = 0;
  while (std::string::npos != pos2) {
    dst_model.emplace_back(dst_model_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = dst_model_str.find(",", pos1);
  }
  dst_model.emplace_back(dst_model_str.substr(pos1));
  return dst_model;
}

std::map<std::string, std::vector<int32_t>> getInputShape() {
  std::map<std::string, std::vector<int32_t>> input_shape;
  std::string input_shape_str = FLAGS_input_shape;
  std::string::size_type pos1, pos2;
  pos2 = input_shape_str.find(";");
  pos1 = 0;
  while (std::string::npos != pos2) {
    std::string shape_str = input_shape_str.substr(pos1, pos2 - pos1);
    std::string::size_type pos3 = shape_str.find(":");
    std::string name = shape_str.substr(0, pos3);
    std::string dims_str = shape_str.substr(pos3 + 1);

    std::vector<int32_t> dims;
    std::string::size_type pos4, pos5;
    pos5 = dims_str.find(",");
    pos4 = 0;
    while (std::string::npos != pos5) {
      dims.push_back(std::stoi(dims_str.substr(pos4, pos5 - pos4)));
      pos4 = pos5 + 1;
      pos5 = dims_str.find(",", pos4);
    }
    dims.push_back(std::stoi(dims_str.substr(pos4)));

    input_shape[name] = dims;
    pos1 = pos2 + 1;
    pos2 = input_shape_str.find(";", pos1);
  }
  if (pos1 < input_shape_str.length()) {
    std::string shape_str = input_shape_str.substr(pos1);
    std::string::size_type pos3 = shape_str.find(":");
    std::string name = shape_str.substr(0, pos3);
    std::string dims_str = shape_str.substr(pos3 + 1);

    std::vector<int32_t> dims;
    std::string::size_type pos4, pos5;
    pos5 = dims_str.find(",");
    pos4 = 0;
    while (std::string::npos != pos5) {
      dims.push_back(std::stoi(dims_str.substr(pos4, pos5 - pos4)));
      pos4 = pos5 + 1;
      pos5 = dims_str.find(",", pos4);
    }
    dims.push_back(std::stoi(dims_str.substr(pos4)));

    input_shape[name] = dims;
  }
  return input_shape;
}

std::map<std::string, base::DataType> getInputDataType() {
  std::map<std::string, base::DataType> input_data_type;
  std::string input_data_type_str = FLAGS_input_data_type;
  std::string::size_type pos1, pos2;
  pos2 = input_data_type_str.find(";");
  pos1 = 0;
  while (std::string::npos != pos2) {
    std::string data_type_str = input_data_type_str.substr(pos1, pos2 - pos1);
    std::string::size_type pos3 = data_type_str.find(":");
    std::string name = data_type_str.substr(0, pos3);
    std::string data_type_str = data_type_str.substr(pos3 + 1);
    input_data_type[name] = base::stringToDataType(data_type_str);
    pos1 = pos2 + 1;
    pos2 = input_data_type_str.find(";", pos1);
  }
  if (pos1 < input_data_type_str.length()) {
    std::string data_type_str = input_data_type_str.substr(pos1);
    std::string::size_type pos3 = data_type_str.find(":");
    std::string name = data_type_str.substr(0, pos3);
    std::string data_type_str = data_type_str.substr(pos3 + 1);
    input_data_type[name] = base::stringToDataType(data_type_str);
  }
  return input_data_type;
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }

  auto onnx_interpret =
      std::shared_ptr<ir::Interpret>(ir::createInterpret(base::kModelTypeOnnx));
  if (onnx_interpret == nullptr) {
    NNDEPLOY_LOGE("ir::createInterpret failed.\n");
    return -1;
  }
  std::vector<std::string> src_model_value;
  src_model_value.push_back(getSrcModel());
  std::vector<ir::ValueDesc> input_value_desc;
  std::map<std::string, base::DataType> input_data_type = getInputDataType();
  for (const auto &item : input_data_type) {
    ir::ValueDesc value_desc(item.first, item.second);
    input_value_desc.push_back(value_desc);
  }
  base::Status status =
      onnx_interpret->interpret(src_model_value, input_value_desc);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("interpret failed\n");
    return -1;
  }
  std::vector<std::string> dst_model_value = getDstModel();
  status =
      onnx_interpret->saveModelToFile(dst_model_value[0], dst_model_value[1]);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("saveModelToFile failed\n");
    return -1;
  }

  return 0;
}
