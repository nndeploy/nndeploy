
#include "nndeploy/ir/ir.h"

namespace nndeploy {
namespace ir {

OpDesc::OpDesc() {}
OpDesc::OpDesc(OpType op_type) : op_type_(op_type) {
  op_param_ = createOpParam(op_type);
}
OpDesc::OpDesc(const std::string &name, OpType op_type)
    : name_(name), op_type_(op_type) {
  op_param_ = createOpParam(op_type);
}
OpDesc::OpDesc(const std::string &name, OpType op_type,
               std::shared_ptr<base::Param> op_param)
    : name_(name), op_type_(op_type) {
  if (op_param != nullptr) {
    op_param_ = op_param->copy();
  } else {
    op_param_ = createOpParam(op_type);
  }
}
OpDesc::OpDesc(const std::string &name, OpType op_type,
               std::initializer_list<std::string> inputs,
               std::initializer_list<std::string> outputs)
    : name_(name), op_type_(op_type), inputs_(inputs), outputs_(outputs) {
  op_param_ = createOpParam(op_type);
}
OpDesc::OpDesc(const std::string &name, OpType op_type,
               std::initializer_list<std::string> inputs,
               std::initializer_list<std::string> outputs,
               std::shared_ptr<base::Param> op_param)
    : name_(name), op_type_(op_type), inputs_(inputs), outputs_(outputs) {
  if (op_param != nullptr) {
    op_param_ = op_param->copy();
  } else {
    op_param_ = createOpParam(op_type);
  }
}
OpDesc::OpDesc(const std::string &name, OpType op_type,
               std::vector<std::string> &inputs,
               std::vector<std::string> &outputs)
    : name_(name), op_type_(op_type), inputs_(inputs), outputs_(outputs) {
  op_param_ = createOpParam(op_type);
}
OpDesc::OpDesc(const std::string &name, OpType op_type,
               std::vector<std::string> &inputs,
               std::vector<std::string> &outputs,
               std::shared_ptr<base::Param> op_param)
    : name_(name), op_type_(op_type), inputs_(inputs), outputs_(outputs) {
  if (op_param != nullptr) {
    op_param_ = op_param->copy();
  } else {
    op_param_ = createOpParam(op_type);
  }
}

OpDesc::~OpDesc() {}

/**
 * @brief
 *
 * @param stream
 * @return base::Status
 * @note
 * op类型,op的名称,输入个数,输出个数,输入名称0,...,输入名称n,输出名称0,...,输出名称n,参数0,参数1,...参数n
 */
base::Status OpDesc::serialize(std::ostream &stream) const {
  // 写入算子类型
  stream << opTypeToString(op_type_) << ",";

  // 写入算子名称
  stream << name_ << ",";

  // 写入输入个数
  stream << (int)inputs_.size() << ",";

  // 写入输出个数
  stream << (int)outputs_.size() << ",";

  // 写入输入名称
  for (const auto &input : inputs_) {
    stream << input << ",";
  }

  // 写入输出名称
  for (const auto &output : outputs_) {
    stream << output << ",";
  }

  // 写入参数
  if (op_param_ != nullptr) {
    op_param_->serialize(stream);
  }

  // 写入换行符
  // stream << std::endl;

  return base::kStatusCodeOk;
}
/**
 * @brief
 *
 * @param stream
 * @return base::Status
 * @note
 * op类型,op的名称,输入个数,输出个数,输入名称0,...,输入名称n,输出名称0,...,输出名称n,参数0,参数1,...参数n
 */
base::Status OpDesc::deserialize(const std::string &line) {
  std::istringstream iss(line);
  std::string token;

  // 读取op类型
  if (!std::getline(iss, token, ',')) return base::kStatusCodeErrorInvalidValue;
  op_type_ = stringToOpType(token);

  // 读取op名称
  if (!std::getline(iss, name_, ',')) return base::kStatusCodeErrorInvalidValue;

  // 读取输入个数
  if (!std::getline(iss, token, ',')) return base::kStatusCodeErrorInvalidValue;
  int input_count = std::stoi(token);

  // 读取输出个数
  if (!std::getline(iss, token, ',')) return base::kStatusCodeErrorInvalidValue;
  int output_count = std::stoi(token);

  // 读取输入名称
  inputs_.clear();
  for (int i = 0; i < input_count; ++i) {
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    inputs_.push_back(token);
  }

  // 读取输出名称
  outputs_.clear();
  for (int i = 0; i < output_count; ++i) {
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    outputs_.push_back(token);
  }

  // 读取参数
  op_param_ = createOpParam(op_type_);
  if (op_param_ != nullptr) {
    // 最大值或者最小值为默认值
    std::string str;
    iss >> str;
    std::cout << str << std::endl;
    base::Status status = op_param_->deserialize(str);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(status, base::kStatusCodeOk, status,
                                 "op_param_->deserialize(iss) failed!");
  }

  return base::kStatusCodeOk;
}

ValueDesc::ValueDesc() {}
ValueDesc::ValueDesc(const std::string &name) : name_(name) {}
ValueDesc::ValueDesc(const std::string &name, base::DataType data_type)
    : name_(name), data_type_(data_type) {}
ValueDesc::ValueDesc(const std::string &name, base::DataType data_type,
                     base::IntVector shape)
    : name_(name), data_type_(data_type), shape_(shape) {}

ValueDesc::~ValueDesc() {
  name_.clear();
  shape_.clear();
}

/**
 * @brief
 *
 * @param stream
 * @return base::Status
 * @note 名称,[数据类型],[形状]
 */
base::Status ValueDesc::serialize(std::ostream &stream) const {
  stream << name_;

  if (data_type_.code_ != base::kDataTypeCodeNotSupport) {
    stream << "," << base::dataTypeToString(data_type_);
  }
  if (!shape_.empty()) {
    stream << ",[";
    for (size_t i = 0; i < shape_.size(); ++i) {
      if (i > 0) {
        stream << ",";
      }
      stream << shape_[i];
    }
    stream << "]";
  }

  return base::kStatusCodeOk;
}
/**
 * @brief
 *
 * @param stream
 * @return base::Status
 * @note 名称,[数据类型],[形状]
 */
base::Status ValueDesc::deserialize(const std::string &str) {
  std::string line(str);
  // 解析名称
  size_t pos = line.find(',');
  if (pos != std::string::npos) {
    name_ = line.substr(0, pos);
    line = line.substr(pos + 1);
  } else {
    name_ = line;
    return base::kStatusCodeOk;
  }

  // 解析数据类型
  pos = line.find(',');
  if (pos != std::string::npos) {
    std::string data_type_str = line.substr(0, pos);
    data_type_ = base::stringToDataType(data_type_str);
    line = line.substr(pos + 1);
  }

  // 解析形状
  if (line.front() == '[' && line.back() == ']') {
    line = line.substr(1, line.length() - 2);
    std::istringstream iss(line);
    std::string token;
    while (std::getline(iss, token, ',')) {
      shape_.push_back(std::stoi(token));
    }
  }

  return base::kStatusCodeOk;
}

ModelDesc::ModelDesc() {}
ModelDesc::~ModelDesc() {
  for (auto iter : weights_) {
    if (iter.second != nullptr) {
      delete iter.second;
    }
  }
}

/**
 * @brief
 *
 * @param oss
 * @return base::Status
 */
base::Status ModelDesc::dump(std::ostream &oss) {
  return serializeStructureToText(oss);
}

/**
 * @brief 序列化模型结构为文本
 *
 * @param output
 * @return base::Status
 * 打印模型结构,不包含权重,文件按行分割,行内使用“,”分割,可选项目[]标识,具体结构如下
 * ModelName: 模型名称
 * inputs_num: 输入个数
 * 名称,[数据类型],[形状]
 * 名称,[数据类型],[形状]
 * ...
 * 名称,[数据类型],[形状]
 * outputs_num: 输出个数
 * 名称,[数据类型],[形状]
 * 名称,[数据类型],[形状]
 * ...
 * 名称,[数据类型],[形状]
 * ops_num: op的个数
 * op类型,op的名称,输入个数,输出个数,输入名称0,...,输入名称n,输出名称0,...,输出名称n,参数0,参数1,...参数n
 * op类型,op的名称,输入个数,输出个数,输入名称0,...,输入名称n,输出名称0,...,输出名称n,参数0,参数1,...参数n
 * ...
 * op类型,op的名称,输入个数,输出个数,输入名称0,...,输入名称n,输出名称0,...,输出名称n,参数0,参数1,...参数n
 * weights: weights个数，weight名称0，weight名称1，...,weight名称n
 * values_num: value的个数（可选项）
 * 名称,[数据类型],[形状]
 * 名称,[数据类型],[形状]
 * ...
 * 名称,[数据类型],[形状]
 * blocks_num: block的个数（可选项，递归调用）
 */
base::Status ModelDesc::serializeStructureToText(std::ostream &stream) const {
  // 写入模型名称
  stream << "ModelName: " << name_ << std::endl;

  // 写入输入信息
  stream << "inputs_num: " << (int)inputs_.size() << std::endl;
  for (const auto &input : inputs_) {
    base::Status status = input->serialize(stream);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Failed to serialize input\n");
      return status;
    }
    stream << std::endl;
  }

  // 写入输出信息
  stream << "outputs_num: " << (int)outputs_.size() << std::endl;
  for (const auto &output : outputs_) {
    base::Status status = output->serialize(stream);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Failed to serialize output\n");
      return status;
    }
    stream << std::endl;
  }

  // 写入操作信息
  stream << "ops_num: " << (int)op_descs_.size() << std::endl;
  for (const auto &op : op_descs_) {
    base::Status status = op->serialize(stream);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Failed to serialize operation\n");
      return status;
    }
    stream << std::endl;
  }

  // 写入权重信息
  stream << "weights: " << (int)weights_.size();
  for (const auto &weight : weights_) {
    stream << "," << weight.first;
  }
  stream << std::endl;

  // 写入中间值信息（如果有）
  if (!values_.empty()) {
    stream << "values_num: " << (int)values_.size() << std::endl;
    for (const auto &value : values_) {
      base::Status status = value->serialize(stream);
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Failed to serialize value\n");
        return status;
      }
      stream << std::endl;
    }
  }

  // 写入块信息（如果有）
  if (!blocks_.empty()) {
    stream << "blocks_num: " << (int)blocks_.size() << std::endl;
    for (const auto &block : blocks_) {
      base::Status status = block->serializeStructureToText(stream);
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Failed to serialize block\n");
        return status;
      }
    }
  }

  return base::kStatusCodeOk;
}

/**
 * @brief
 * 从文本反序列化为模型结构，不包含weights和values，当input有值时，覆盖inputs_
 *
 * @param output
 * @return base::Status
 * 打印模型结构,不包含权重,文件按行分割,行内使用“,”分割,可选项目[]标识,具体结构如下
 * ModelName: 模型名称
 * inputs_num: 输入个数
 * 名称,[数据类型],[形状]
 * 名称,[数据类型],[形状]
 * ...
 * 名称,[数据类型],[形状]
 * outputs_num: 输出个数
 * 名称,[数据类型],[形状]
 * 名称,[数据类型],[形状]
 * ...
 * 名称,[数据类型],[形状]
 * ops_num: op的个数
 * op类型,op的名称,输入个数,输出个数,输入名称0,...,输入名称n,输出名称0,...,输出名称n,参数0,参数1,...参数n
 * op类型,op的名称,输入个数,输出个数,输入名称0,...,输入名称n,输出名称0,...,输出名称n,参数0,参数1,...参数n
 * ...
 * op类型,op的名称,输入个数,输出个数,输入名称0,...,输入名称n,输出名称0,...,输出名称n,参数0,参数1,...参数n
 * weights: weights个数，weight名称0，weight名称1，...,weight名称n
 * values_num: value的个数（可选项）
 * 名称,[数据类型],[形状]
 * 名称,[数据类型],[形状]
 * ...
 * 名称,[数据类型],[形状]
 * blocks_num: block的个数（可选项，递归调用）
 */
base::Status ModelDesc::deserializeStructureFromText(
    std::istream &stream, const std::vector<ValueDesc> &input) {
  std::string line;  // 每行信息
  while (std::getline(stream, line)) {
    std::istringstream iss(line);
    std::string key;
    if (!(iss >> key)) {
      continue;
    }

    if (key == "ModelName:") {
      iss >> name_;
    } else if (key == "inputs_num:") {
      int inputs_num;
      iss >> inputs_num;
      inputs_.clear();
      for (size_t i = 0; i < inputs_num; ++i) {
        std::getline(stream, line);
        std::shared_ptr<ValueDesc> input_desc = std::make_shared<ValueDesc>();
        base::Status status = input_desc->deserialize(line);
        if (status != base::kStatusCodeOk) {
          NNDEPLOY_LOGE("Failed to deserialize input\n");
          return status;
        }
        inputs_.emplace_back(input_desc);
      }
    } else if (key == "outputs_num:") {
      int outputs_num;
      iss >> outputs_num;
      outputs_.clear();
      for (size_t i = 0; i < outputs_num; ++i) {
        std::getline(stream, line);
        std::shared_ptr<ValueDesc> output_desc = std::make_shared<ValueDesc>();
        base::Status status = output_desc->deserialize(line);
        if (status != base::kStatusCodeOk) {
          NNDEPLOY_LOGE("Failed to deserialize output\n");
          return status;
        }
        outputs_.emplace_back(output_desc);
      }
    } else if (key == "ops_num:") {
      int ops_num;
      iss >> ops_num;
      op_descs_.clear();
      for (size_t i = 0; i < ops_num; ++i) {
        std::getline(stream, line);
        std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>();
        base::Status status = op_desc->deserialize(line);
        if (status != base::kStatusCodeOk) {
          NNDEPLOY_LOGE("Failed to deserialize op\n");
          return status;
        }
        op_descs_.emplace_back(op_desc);
      }
    } else if (key == "blocks_num:") {
      int blocks_num;
      iss >> blocks_num;
      blocks_.clear();
      for (size_t i = 0; i < blocks_num; ++i) {
        std::shared_ptr<ModelDesc> block = std::make_shared<ModelDesc>();
        base::Status status =
            block->deserializeStructureFromText(stream, input);
        if (status != base::kStatusCodeOk) {
          NNDEPLOY_LOGE("Failed to deserialize block\n");
          return status;
        }
        blocks_.emplace_back(block);
      }
    }
  }

  if (!input.empty()) {
    for (auto &new_input : input) {
      for (auto &existing_input : inputs_) {
        if (existing_input->name_ == new_input.name_) {
          existing_input->name_ = new_input.name_;
          existing_input->data_type_ = new_input.data_type_;
          existing_input->shape_ = new_input.shape_;
          break;
        }
      }
    }
  }
  return base::kStatusCodeOk;
}

// 序列化模型权重为二进制文件
base::Status ModelDesc::serializeWeightsToBinary(std::ostream &stream) const {
  size_t weights_size = weights_.size();
  NNDEPLOY_LOGE("weights_size = %d\n", (int)weights_size);
  if (!stream.write(reinterpret_cast<const char *>(&weights_size),
                    sizeof(weights_size))) {
    return base::kStatusCodeErrorIO;
  }
  for (auto &weight : weights_) {
    NNDEPLOY_LOGE("weight->getName() = %s\n", weight.second->getName().c_str());
    base::Status status = weight.second->serialize(stream);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(status, base::kStatusCodeOk, status,
                                 "weight.second->serialize(stream) failed!\n");
  }
  for (auto &block : blocks_) {
    base::Status status = block->serializeWeightsToBinary(stream);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(
        status, base::kStatusCodeOk, status,
        "block->serializeWeightsToBinary(stream) failed!\n");
  }
  return base::kStatusCodeOk;
}
// 从二进制文件反序列化为模型权重
base::Status ModelDesc::deserializeWeightsFromBinary(std::istream &stream) {
  NNDEPLOY_LOGE("hello world\n");
  size_t weights_size;
  NNDEPLOY_LOGE("hello world\n");
  if (!stream.read(reinterpret_cast<char *>(&weights_size),
                   sizeof(weights_size))) {
    return base::kStatusCodeErrorIO;
  }
  NNDEPLOY_LOGE("weights_size = %d\n", (int)weights_size);
  NNDEPLOY_LOGE("hello world\n");
  weights_.clear();
  for (size_t i = 0; i < weights_size; ++i) {
    device::Tensor *weight = new device::Tensor();
    NNDEPLOY_LOGE("hello world\n");
    base::Status status = weight->deserialize(stream);
    if (status != base::kStatusCodeOk) {
      delete weight;
      NNDEPLOY_LOGE("weight->deserialize(stream) failed!\n");
      return status;
    }
    NNDEPLOY_LOGE("weight->getName() = %s\n", weight->getName().c_str());
    weights_[weight->getName()] = weight;
    NNDEPLOY_LOGE("hello world\n");
  }
  for (auto &block : blocks_) {
    base::Status status = block->deserializeWeightsFromBinary(stream);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("block->deserializeWeightsFromBinary(stream) failed!\n");
      return status;
    }
  }
  return base::kStatusCodeOk;
}

}  // namespace ir
}  // namespace nndeploy