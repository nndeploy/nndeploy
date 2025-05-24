#ifndef _NNDEPLOY_CODEC_CODEC_H_
#define _NNDEPLOY_CODEC_CODEC_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace codec {

class NNDEPLOY_CC_API DecodeNode : public dag::Node {
 public:
  DecodeNode(const std::string &name) : dag::Node(name) {
    node_type_ = dag::NodeType::kNodeTypeInput;
  }
  DecodeNode(const std::string &name, std::vector<dag::Edge *> inputs,
             std::vector<dag::Edge *> outputs)
      : dag::Node(name) {
    node_type_ = dag::NodeType::kNodeTypeInput;
    if (inputs.size() > 0) {
      NNDEPLOY_LOGE("DecodeNode not support inputs");
      constructed_ = false;
      return;
    }
    if (outputs.size() > 1) {
      NNDEPLOY_LOGE("DecodeNode only support one output");
      constructed_ = false;
      return;
    }
    outputs_ = outputs;
  }
  DecodeNode(const std::string &name, base::CodecFlag flag) : dag::Node(name) {
    flag_ = flag;
    node_type_ = dag::NodeType::kNodeTypeInput;
    // this->setOutputTypeInfo<cv::Mat>();
  }
  DecodeNode(const std::string &name, std::vector<dag::Edge *> inputs,
             std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : dag::Node(name) {
    flag_ = flag;
    node_type_ = dag::NodeType::kNodeTypeInput;
    // this->setOutputTypeInfo<cv::Mat>();
    if (inputs.size() > 0) {
      NNDEPLOY_LOGE("DecodeNode not support inputs");
      constructed_ = false;
      return;
    }
    if (outputs.size() > 1) {
      NNDEPLOY_LOGE("DecodeNode only support one output");
      constructed_ = false;
      return;
    }
    outputs_ = outputs;
  }
  virtual ~DecodeNode() {}

  base::Status setCodecFlag(base::CodecFlag flag) {
    flag_ = flag;
    return base::kStatusCodeOk;
  }
  base::CodecFlag getCodecFlag() { return flag_; }
  virtual base::Status setPath(const std::string &path) = 0;

  void setSize(int size) {
    if (size > 0) {
      size_ = size;
    }
  }
  int getSize() { return size_; }

  double getFps() { return fps_; };
  int getWidth() { return width_; };
  int getHeight() { return height_; };

  virtual base::EdgeUpdateFlag updateInput() {
    if (index_ < size_) {
      return base::kEdgeUpdateFlagComplete;
    } else {
      if (size_ == 0) {
        return base::kEdgeUpdateFlagComplete;
      } else {
        return base::kEdgeUpdateFlagTerminate;
      }
    }
  }

  virtual base::Status run() = 0;

  virtual base::Status serialize(
      rapidjson::Value &json,
      rapidjson::Document::AllocatorType &allocator) const {
    base::Status status = dag::Node::serialize(json, allocator);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    std::string flag_str = base::codecFlagToString(flag_);
    json.AddMember("flag_", rapidjson::Value(flag_str.c_str(), allocator),
                   allocator);
    return status;
  }

  virtual base::Status deserialize(rapidjson::Value &json) {
    base::Status status = dag::Node::deserialize(json);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    if (json.HasMember("flag_") && json["flag_"].IsString()) {
      flag_ = base::stringToCodecFlag(json["flag_"].GetString());
    }
    return status;
  }

 protected:
  base::CodecFlag flag_ = base::kCodecFlagImage;
  std::string path_ = "";
  bool path_changed_ = false;
  int size_ = 0;
  double fps_ = 0.0;
  int width_ = 0;
  int height_ = 0;
  int index_ = 0;
};

class NNDEPLOY_CC_API EncodeNode : public dag::Node {
 public:
  EncodeNode(const std::string &name) : dag::Node(name) {
    node_type_ = dag::NodeType::kNodeTypeOutput;
    // this->setInputTypeInfo<cv::Mat>();
  }
  EncodeNode(const std::string &name, std::vector<dag::Edge *> inputs,
             std::vector<dag::Edge *> outputs)
      : dag::Node(name) {
    // this->setInputTypeInfo<cv::Mat>();
    node_type_ = dag::NodeType::kNodeTypeOutput;
    if (inputs.size() > 1) {
      NNDEPLOY_LOGE("EncodeNode only support one input");
      constructed_ = false;
      return;
    }
    inputs_ = inputs;
    if (outputs.size() > 0) {
      NNDEPLOY_LOGE("EncodeNode not support outputs");
      constructed_ = false;
      return;
    }
  }
  EncodeNode(const std::string &name, base::CodecFlag flag) : dag::Node(name) {
    flag_ = flag;
    node_type_ = dag::NodeType::kNodeTypeOutput;
    // this->setInputTypeInfo<cv::Mat>();
  }
  EncodeNode(const std::string &name, std::vector<dag::Edge *> inputs,
             std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : dag::Node(name) {
    flag_ = flag;
    // this->setInputTypeInfo<cv::Mat>();
    node_type_ = dag::NodeType::kNodeTypeOutput;
    if (inputs.size() > 1) {
      NNDEPLOY_LOGE("EncodeNode only support one input");
      constructed_ = false;
      return;
    }
    inputs_ = inputs;
    if (outputs.size() > 0) {
      NNDEPLOY_LOGE("EncodeNode not support outputs");
      constructed_ = false;
      return;
    }
  }
  virtual ~EncodeNode() {};

  base::Status setCodecFlag(base::CodecFlag flag) {
    flag_ = flag;
    return base::kStatusCodeOk;
  }
  base::CodecFlag getCodecFlag() { return flag_; }
  // virtual base::Status setPath(const std::string &path) {
  //   path_ = path;
  //   path_changed_ = true;
  //   return base::kStatusCodeOk;
  // }
  // virtual base::Status setRefPath(const std::string &ref_path) {
  //   ref_path_ = ref_path;
  //   path_changed_ = true;
  //   return base::kStatusCodeOk;
  // }
  virtual base::Status setPath(const std::string &path) = 0;
  virtual base::Status setRefPath(const std::string &ref_path) = 0;
  void setFourcc(const std::string &fourcc) { fourcc_ = fourcc; }
  void setFps(double fps) { fps_ = fps; };
  void setWidth(int width) { width_ = width; };
  void setHeight(int height) { height_ = height; };

  int getIndex() { return index_; };

  virtual base::Status run() = 0;

  virtual base::Status serialize(
      rapidjson::Value &json,
      rapidjson::Document::AllocatorType &allocator) const {
    base::Status status = dag::Node::serialize(json, allocator);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    std::string flag_str = base::codecFlagToString(flag_);
    json.AddMember("flag_", rapidjson::Value(flag_str.c_str(), allocator),
                   allocator);
    return status;
  }

  virtual base::Status deserialize(rapidjson::Value &json) {
    base::Status status = dag::Node::deserialize(json);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    if (json.HasMember("flag_") && json["flag_"].IsString()) {
      flag_ = base::stringToCodecFlag(json["flag_"].GetString());
    }
    return status;
  }

 protected:
  base::CodecFlag flag_ = base::kCodecFlagImage;
  std::string path_ = "";
  bool path_changed_ = false;
  std::string ref_path_ = "";
  std::string fourcc_ = "MJPG";
  double fps_ = 0.0;
  int width_ = 0;
  int height_ = 0;
  int index_ = 0;
};

using createDecodeNodeFunc = std::function<DecodeNode *(
    base::CodecFlag flag, const std::string &name, dag::Edge *output)>;
using createDecodeNodeSharedPtrFunc = std::function<std::shared_ptr<DecodeNode>(
    base::CodecFlag flag, const std::string &name, dag::Edge *output)>;

std::map<base::CodecType, createDecodeNodeFunc> &
getGlobalCreateDecodeNodeFuncMap();

std::map<base::CodecType, createDecodeNodeSharedPtrFunc> &
getGlobalCreateDecodeNodeSharedPtrFuncMap();

class TypeCreatelDecodeNodeRegister {
 public:
  explicit TypeCreatelDecodeNodeRegister(base::CodecType type,
                                         createDecodeNodeFunc func) {
    getGlobalCreateDecodeNodeFuncMap()[type] = func;
  }
};

class TypeCreatelDecodeNodeSharedPtrRegister {
 public:
  explicit TypeCreatelDecodeNodeSharedPtrRegister(
      base::CodecType type, createDecodeNodeSharedPtrFunc func) {
    getGlobalCreateDecodeNodeSharedPtrFuncMap()[type] = func;
  }
};
extern NNDEPLOY_CC_API DecodeNode *createDecodeNode(base::CodecType type,
                                                    base::CodecFlag flag,
                                                    const std::string &name,
                                                    dag::Edge *output);

extern NNDEPLOY_CC_API std::shared_ptr<DecodeNode> createDecodeNodeSharedPtr(
    base::CodecType type, base::CodecFlag flag, const std::string &name,
    dag::Edge *output);

using createEncodeNodeFunc = std::function<EncodeNode *(
    base::CodecFlag flag, const std::string &name, dag::Edge *input)>;
using createEncodeNodeSharedPtrFunc = std::function<std::shared_ptr<EncodeNode>(
    base::CodecFlag flag, const std::string &name, dag::Edge *input)>;

std::map<base::CodecType, createEncodeNodeFunc> &
getGlobalCreateEncodeNodeFuncMap();

std::map<base::CodecType, createEncodeNodeSharedPtrFunc> &
getGlobalCreateEncodeNodeSharedPtrFuncMap();

class TypeCreatelEncodeNodeRegister {
 public:
  explicit TypeCreatelEncodeNodeRegister(base::CodecType type,
                                         createEncodeNodeFunc func) {
    getGlobalCreateEncodeNodeFuncMap()[type] = func;
  }
};

class TypeCreatelEncodeNodeSharedPtrRegister {
 public:
  explicit TypeCreatelEncodeNodeSharedPtrRegister(
      base::CodecType type, createEncodeNodeSharedPtrFunc func) {
    getGlobalCreateEncodeNodeSharedPtrFuncMap()[type] = func;
  }
};

extern NNDEPLOY_CC_API EncodeNode *createEncodeNode(base::CodecType type,
                                                    base::CodecFlag flag,
                                                    const std::string &name,
                                                    dag::Edge *input);

extern NNDEPLOY_CC_API std::shared_ptr<EncodeNode> createEncodeNodeSharedPtr(
    base::CodecType type, base::CodecFlag flag, const std::string &name,
    dag::Edge *input);

}  // namespace codec
}  // namespace nndeploy

#endif /* _NNDEPLOY_CODEC_CODEC_H_ */
