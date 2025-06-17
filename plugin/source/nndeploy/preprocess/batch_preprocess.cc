
#include "nndeploy/preprocess/batch_preprocess.h"

#include "nndeploy/preprocess/util.h"

namespace nndeploy {
namespace preprocess {

base::Status BatchPreprocess::setNodeKey(const std::string &key) {
  base::Status status = this->deinit();
  if (status != base::kStatusCodeOk) {
    return status;
  }
  node_key_ = key;
  dag::NodeDesc desc(node_key_, "inner_preprocess_node",
                     {"inner_preprocess_node.input"},
                     {"inner_preprocess_node.output"});
  node_ = this->createNode(desc);
  if (!node_) {
    NNDEPLOY_LOGE("Node creation failed for node_key: %s\n", node_key_.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  return base::kStatusCodeOk;
}

base::Status BatchPreprocess::setDataFormat(base::DataFormat data_format) {
  data_format_ = data_format;
  return base::kStatusCodeOk;
}
base::DataFormat BatchPreprocess::getDataFormat() { return data_format_; }

base::Status BatchPreprocess::setParam(base::Param *param) {
  if (node_) {
    node_->setParam(param);
  }
  return base::kStatusCodeOk;
}
base::Status BatchPreprocess::setParamSharedPtr(
    std::shared_ptr<base::Param> param) {
  if (node_) {
    node_->setParamSharedPtr(param);
  }
  return base::kStatusCodeOk;
}
base::Param *BatchPreprocess::getParam() {
  if (node_) {
    return node_->getParam();
  }
  return nullptr;
}
std::shared_ptr<base::Param> BatchPreprocess::getParamSharedPtr() {
  if (node_) {
    return node_->getParamSharedPtr();
  }
  return nullptr;
}

// base::Status BatchPreprocess::make() {
//   return base::kStatusCodeOk;
// }

base::Status BatchPreprocess::run() {
  std::vector<cv::Mat> *input_data =
      inputs_[0]->getGraphOutputAny<std::vector<cv::Mat>>();
  int batch_size = input_data->size();
  device::Tensor *dst_tensor = nullptr;
  for (int i = 0; i < batch_size; i++) {
    dag::Edge *input = node_->getInput();
    input->set((*input_data)[i]);
    node_->run();
    dag::Edge *output = node_->getOutput();
    device::Tensor *single_tensor = output->getTensor(node_);
    if (single_tensor == nullptr) {
      NNDEPLOY_LOGE("single_tensor is nullptr");
      return base::kStatusCodeErrorInvalidParam;
    }
    device::Device *device = single_tensor->getDevice();
    device::TensorDesc desc = single_tensor->getDesc();
    if (i == 0) {
      if (data_format_ == base::kDataFormatNDCHW ||
          data_format_ == base::kDataFormatNDHWC) {
        // 在这里，`desc.shape_`是一个表示张量形状的向量。`insert`函数用于在向量的指定位置插入一个元素。
        // `desc.shape_.begin() + 1`表示在向量的第二个位置插入元素。
        // `batch_size`是要插入的元素，表示批处理的大小。
        // 这行代码的作用是将批处理大小插入到张量形状的第二个位置，从而调整张量的形状以适应批处理。
        desc.shape_.insert(desc.shape_.begin() + 1, batch_size);
        desc.data_format_ = data_format_;
      } else {
        desc.shape_[0] = batch_size;
      }
      dst_tensor = outputs_[0]->create(device, desc);
    }
    void *single_data = single_tensor->getData();
    void *data = ((char *)dst_tensor->getData()) + single_tensor->getSize() * i;
    device->copy(data, single_data, single_tensor->getSize());
  }
  outputs_[0]->notifyWritten(dst_tensor);
  return base::kStatusCodeOk;
}

base::Status BatchPreprocess::serialize(
    rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
  base::Status status = dag::CompositeNode::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  std::string data_format_str = base::dataFormatToString(data_format_);
  json.AddMember("data_format_",
                 rapidjson::Value(data_format_str.c_str(),
                                  data_format_str.length(), allocator),
                 allocator);
  json.AddMember("node_key_",
                 rapidjson::Value(node_key_.c_str(), node_key_.length(),
                                  allocator),
                 allocator);
  // if (node_) {
  //   status = node_->serialize(json, allocator);
  //   if (status != base::kStatusCodeOk) {
  //     return status;
  //   }
  // }
  return base::kStatusCodeOk;
}

// std::string BatchPreprocess::serialize() {
//   rapidjson::Document doc;
//   doc.SetObject();
//   this->serialize(doc, doc.GetAllocator());
//   rapidjson::StringBuffer buffer;
//   rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
//   doc.Accept(writer);
//   std::string json_str = buffer.GetString();
//   if (node_ == nullptr) {
//     return json_str;
//   }
//   json_str[json_str.length() - 1] = ',';
//   json_str += "\"node_\": ";
//   json_str += node_->serialize();
//   json_str += "}";
//   return json_str;
// }

base::Status BatchPreprocess::deserialize(rapidjson::Value &json) {
  base::Status status = dag::CompositeNode::deserialize(json);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  if (json.HasMember("data_format_") && json["data_format_"].IsString()) {
    std::string data_format_str = json["data_format_"].GetString();
    data_format_ = base::stringToDataFormat(data_format_str);
    if (data_format_ == base::kDataFormatNotSupport) {
      NNDEPLOY_LOGE("Invalid data format: %s", data_format_str.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
  }
  if (json.HasMember("node_key_") && json["node_key_"].IsString()) {
    std::string node_key_str = json["node_key_"].GetString();
    this->setNodeKey(node_key_str);
  }
  // if (node_) {
  //   status = node_->deserialize(json);
  //   if (status != base::kStatusCodeOk) {
  //     NNDEPLOY_LOGE("Node deserialization failed");
  //     return status;
  //   }
  // }
  return base::kStatusCodeOk;
}

// base::Status BatchPreprocess::deserialize(const std::string &json_str) {
//   rapidjson::Document document;
//   if (document.Parse(json_str.c_str()).HasParseError()) {
//     NNDEPLOY_LOGE("parse json string failed\n");
//     return base::kStatusCodeErrorInvalidParam;
//   }
//   rapidjson::Value &json = document;
//   base::Status status = this->deserialize(json);
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("deserialize failed\n");
//     return status;
//   }
//   if (json.HasMember("node_") && json["node_"].IsObject()) {
//     rapidjson::Value &node_json = json["node_"];
//     rapidjson::StringBuffer buffer;
//     rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
//     node_json.Accept(writer);
//     std::string node_json_str = buffer.GetString();
//     // dag::NodeDesc node_desc;
//     // status = node_desc.deserialize(node_json_str);
//     // if (status != base::kStatusCodeOk) {
//     //   return status;
//     // }
//     // Node *node = this->createNode(node_desc);
//     // if (node == nullptr) {
//     //   NNDEPLOY_LOGE("create node failed\n");
//     //   return base::kStatusCodeErrorInvalidValue;
//     // }
//     base::Status status = node_->deserialize(node_json_str);
//     if (status != base::kStatusCodeOk) {
//       NNDEPLOY_LOGE("deserialize node failed\n");
//       return status;
//     }
//     // node_ = node;
//   }
//   return base::kStatusCodeOk;
// }

REGISTER_NODE("nndeploy::preprocess::BatchPreprocess", BatchPreprocess);

}  // namespace preprocess
}  // namespace nndeploy
