
#include "nndeploy/op/ir.h"

namespace nndeploy {
namespace op {

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

ValueDesc::ValueDesc() {}
ValueDesc::ValueDesc(const std::string &name) : name_(name) {}
ValueDesc::ValueDesc(const std::string &name, base::DataType data_type)
    : name_(name), data_type_(data_type) {}
ValueDesc::ValueDesc(const std::string &name, base::DataType data_type,
                     base::IntVector shape)
    : name_(name), data_type_(data_type), shape_(shape) {}

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
 * @note - 打印模型结构，不包含权重，文件按行分割，行内使用“,”分割，具体结构如下
 * ModelName: 模型名称
 * inputs num: 输入个数
 * inputs[0]: 第一个输入的名称，[数据类型]，[形状]
 * inputs[1]: 第二个输入的名称，[数据类型]，[形状]
 * ...
 * inputs[n]: 第n个输入的名称，[数据类型]，[形状]
 */
base::Status ModelDesc::dump(std::ostream &oss) {
  oss << "ModelDesc: " << std::endl;
  oss << "  name: " << name_ << std::endl;
  oss << "  inputs: " << std::endl;
  for (auto iter : inputs_) {
    oss << "    " << iter->name_;
    oss << std::endl;
  }
  oss << "  outputs: " << std::endl;
  for (auto iter : outputs_) {
    oss << "    " << iter->name_;
    oss << std::endl;
  }
  oss << "  op_descs: " << std::endl;
  for (auto iter : op_descs_) {
    oss << "    " << iter->name_ << " : " << iter->op_type_ << std::endl;
  }
  oss << "  weights: " << std::endl;
  for (auto iter : weights_) {
    oss << "    " << iter.first;
    oss << std::endl;
  }
  if (values_.size() > 0) {
    oss << "  values_: " << std::endl;
    for (auto iter : values_) {
      oss << "    " << iter->name_;
      oss << std::endl;
    }
  }
  if (blocks_.size() > 0) {
    oss << "  blocks_: " << std::endl;
    for (auto iter : blocks_) {
      oss << "    " << iter->name_;
      oss << std::endl;
    }
  }

  return base::kStatusCodeOk;
}

// base::Status ModelDesc::serialize(std::ostream &output) const {
//   // 序列化模型名称
//   base::writeString(output, name_);

//   // 序列化算子描述列表
//   base::writeVector(output, op_descs_, [this](std::ostream &os, const
//   std::shared_ptr<OpDesc> &op_desc) {
//     return this->serializeOpDesc(op_desc, os);
//   });

//   // 序列化权重
//   base::writeMap(output, weights_, [this](std::ostream &os, const
//   std::pair<std::string, device::Tensor*> &weight) {
//     base::writeString(os, weight.first);
//     return this->serializeTensor(weight.second, os);
//   });

//   // 序列化输入描述
//   base::writeVector(output, inputs_, [this](std::ostream &os, const
//   std::shared_ptr<ValueDesc> &value_desc) {
//     return this->serializeValueDesc(value_desc, os);
//   });

//   // 序列化输出描述
//   base::writeVector(output, outputs_, [this](std::ostream &os, const
//   std::shared_ptr<ValueDesc> &value_desc) {
//     return this->serializeValueDesc(value_desc, os);
//   });

//   // 序列化中间值描述
//   base::writeVector(output, values_, [this](std::ostream &os, const
//   std::shared_ptr<ValueDesc> &value_desc) {
//     return this->serializeValueDesc(value_desc, os);
//   });

//   // 序列化子模型块
//   base::writeVector(output, blocks_, [](std::ostream &os, const
//   std::shared_ptr<ModelDesc> &block) {
//     return block->serialize(os);
//   });

//   return base::kStatusCodeOk;
// }

// base::Status ModelDesc::deserialize(std::istream &input) {
//   // 反序列化模型名称
//   base::readString(input, name_);

//   // 反序列化算子描述列表
//   base::readVector(input, op_descs_, [this](std::istream &is,
//   std::shared_ptr<OpDesc> &op_desc) {
//     return this->deserializeOpDesc(is, op_desc);
//   });

//   // 反序列化权重
//   base::readMap(input, weights_, [this](std::istream &is,
//   std::pair<std::string, device::Tensor*> &weight) {
//     base::readString(is, weight.first);
//     return this->deserializeTensor(is, weight.second);
//   });

//   // 反序列化输入描述
//   base::readVector(input, inputs_, [this](std::istream &is,
//   std::shared_ptr<ValueDesc> &value_desc) {
//     return this->deserializeValueDesc(is, value_desc);
//   });

//   // 反序列化输出描述
//   base::readVector(input, outputs_, [this](std::istream &is,
//   std::shared_ptr<ValueDesc> &value_desc) {
//     return this->deserializeValueDesc(is, value_desc);
//   });

//   // 反序列化中间值描述
//   base::readVector(input, values_, [this](std::istream &is,
//   std::shared_ptr<ValueDesc> &value_desc) {
//     return this->deserializeValueDesc(is, value_desc);
//   });

//   // 反序列化子模型块
//   base::readVector(input, blocks_, [](std::istream &is,
//   std::shared_ptr<ModelDesc> &block) {
//     block = std::make_shared<ModelDesc>();
//     return block->deserialize(is);
//   });

//   return base::kStatusCodeOk;
// }

std::map<OpType, std::shared_ptr<OpParamCreator>> &
getGlobalOpParamCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<OpType, std::shared_ptr<OpParamCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(new std::map<OpType, std::shared_ptr<OpParamCreator>>);
  });
  return *creators;
}

std::shared_ptr<base::Param> createOpParam(OpType type) {
  std::shared_ptr<base::Param> temp;
  auto &creater_map = getGlobalOpParamCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createOpParam(type);
  }
  return temp;
}

// Concat 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeConcat, ConcatParam);

// Conv 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeConv, ConvParam);

// MaxPool 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeMaxPool, MaxPoolParam);

// Reshape 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeReshape, ReshapeParam);

// Resize 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeResize, ResizeParam);

// Softmax 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeSoftmax, SoftmaxParam);

// Split 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeSplit, SplitParam);

// Tranpose 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeTranspose, TransposeParam);

// TODO: @Leonisux:
// 补充llama的算子的参数的注册
// RMSNorm 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeRMSNorm, RMSNormParam);

}  // namespace op
}  // namespace nndeploy