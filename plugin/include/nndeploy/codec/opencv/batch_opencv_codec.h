#ifndef _NNDEPLOY_CODEC_BATCH_CODEC_H_
#define _NNDEPLOY_CODEC_BATCH_CODEC_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/dag/composite_node.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace codec {

class NNDEPLOY_CC_API BatchOpenCvDecode : public dag::CompositeNode {
 public:
  BatchOpenCvDecode(const std::string &name) : dag::CompositeNode(name) {
    key_ = "nndeploy::codec::BatchOpenCvDecode";
    node_type_ = dag::NodeType::kNodeTypeInput;
    this->setOutputTypeInfo<std::vector<cv::Mat>>();
  }
  BatchOpenCvDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs)
      : dag::CompositeNode(name, inputs, outputs) {
    key_ = "nndeploy::codec::BatchOpenCvDecode";
    node_type_ = dag::NodeType::kNodeTypeInput;
    this->setOutputTypeInfo<std::vector<cv::Mat>>();
  }
  virtual ~BatchOpenCvDecode() {}

  void setBatchSize(int batch_size_) { batch_size_ = batch_size_; }
  base::Status setNodeKey(const std::string &key) {
    node_key_ = key;
    return base::kStatusCodeOk;
  }

  base::Status setCodecFlag(base::CodecFlag flag) {
    if (node_) {
      return node_->setCodecFlag(flag);
    }
    return base::kStatusCodeErrorNullParam;
  }
  base::CodecFlag getCodecFlag() {
    if (node_) {
      return node_->getCodecFlag();
    }
    return base::kCodecFlagImage;  // 默认值
  }
  void setPath(const std::string &path) {
    if (node_) {
      node_->setPath(path);
    }
  }

  void setSize(int size) {
    if (node_) {
      node_->setSize(size);
    }
  }
  int getSize() {
    if (node_) {
      iter_size_ = NNDEPLOY_UP_DIV((int)(node_->getSize()), batch_size_);
    }
    return iter_size_;
  }

  double getFps() {
    if (node_) {
      return node_->getFps();
    }
    return 0.0;  // 默认值
  }
  int getWidth() {
    if (node_) {
      return node_->getWidth();
    }
    return 0;  // 默认值
  }
  int getHeight() {
    if (node_) {
      return node_->getHeight();
    }
    return 0;  // 默认值
  }

  virtual base::EdgeUpdateFlag updateInput() {
    if (index_ < iter_size_) {
      return base::kEdgeUpdateFlagComplete;
    } else {
      if (iter_size_ == 0) {
        return base::kEdgeUpdateFlagComplete;
      } else {
        return base::kEdgeUpdateFlagTerminate;
      }
    }
  }

  virtual base::Status make() {
    std::vector<std::string> input_names = this->getInputNames();
    std::vector<std::string> output_names = this->getRealOutputsName();
    dag::NodeDesc desc(node_key_, "inner_codec_node", input_names,
                       output_names);
    node_ = (DecodeNode *)this->createNode(desc);
    if (!node_) {
      NNDEPLOY_LOGE("Node creation failed for node_key: %s\n",
                    node_key_.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
    // if (node_->getInputTypeInfo() != this->getInputTypeInfo() ||
    //     node_->getOutputTypeInfo() != this->getOutputTypeInfo()) {
    //   NNDEPLOY_LOGE(
    //       "Type mismatch: Node input/output types do not match
    //       BatchPreprocess " "types.\n");
    //   return base::kStatusCodeErrorInvalidParam;
    // }
    return base::kStatusCodeOk;
  }

  virtual base::Status init() {
    index_ = 0;
    if (node_) {
      return node_->init();
    }
    return base::kStatusCodeErrorNullParam;
  }

  virtual base::Status deinit() {
    if (node_) {
      return node_->deinit();
    }
    return base::kStatusCodeErrorNullParam;
  }

  virtual base::Status run() {
    auto results = new std::vector<cv::Mat>();
    if (index_ >= iter_size_) {
      outputs_[0]->setAny(results, false);
      return base::kStatusCodeOk;
    }
    for (int i = 0; i < batch_size_; i++) {
      cv::Mat *single = nullptr;
      if (index_ * batch_size_ + i < node_->getSize()) {
        node_->run();
        dag::Edge *output = node_->getOutput();
        cv::Mat *single = output->getCvMat(node_);
        if (single == nullptr) {
          NNDEPLOY_LOGE("single_tensor is nullptr");
          return base::kStatusCodeErrorInvalidParam;
        }
      }
      cv::Mat res = single->clone();
      results->push_back(res);
    }
    outputs_[0]->setAny(results, false);
    index_++;
    return base::kStatusCodeOk;
  }

 private:
  int batch_size_ = 1;
  int index_ = 0;
  int iter_size_ = 1;
  std::string node_key_ = "";
  DecodeNode *node_ = nullptr;
};

class NNDEPLOY_CC_API BatchOpenCvEncode : public dag::CompositeNode {
 public:
  BatchOpenCvEncode(const std::string &name) : dag::CompositeNode(name) {
    key_ = "nndeploy::codec::BatchOpenCvEncode";
    node_type_ = dag::NodeType::kNodeTypeOutput;
    this->setInputTypeInfo<std::vector<cv::Mat>>();
  }
  BatchOpenCvEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs)
      : dag::CompositeNode(name, inputs, outputs) {
    key_ = "nndeploy::codec::BatchOpenCvEncode";
    node_type_ = dag::NodeType::kNodeTypeOutput;
    this->setInputTypeInfo<std::vector<cv::Mat>>();
  }
  virtual ~BatchOpenCvEncode() {}

  base::Status setNodeKey(const std::string &key) {
    node_key_ = key;
    return base::kStatusCodeOk;
  }

  base::Status setCodecFlag(base::CodecFlag flag) {
    if (node_) {
      return node_->setCodecFlag(flag);
    }
    return base::kStatusCodeErrorNullParam;
  }
  base::CodecFlag getCodecFlag() {
    if (node_) {
      return node_->getCodecFlag();
    }
    return base::kCodecFlagImage;  // 默认值
  }
  void setPath(const std::string &path) {
    if (node_) {
      node_->setPath(path);
    }
  }
  void setRefPath(const std::string &ref_path) {
    if (node_) {
      node_->setRefPath(ref_path);
    }
  }
  void setFourcc(const std::string &fourcc) {
    if (node_) {
      node_->setFourcc(fourcc);
    }
  }
  void setFps(double fps) {
    if (node_) {
      node_->setFps(fps);
    }
  }
  void setWidth(int width) {
    if (node_) {
      node_->setWidth(width);
    }
  }
  void setHeight(int height) {
    if (node_) {
      node_->setHeight(height);
    }
  }
  int getIndex() {
    if (node_) {
      return node_->getIndex();
    }
    return 0;  // 默认值
  }

  virtual base::Status make() {
    dag::NodeDesc desc(node_key_, "inner_codec_node", {"inner_codec_node.input"},
                       {});
    node_ = (EncodeNode *)this->createNode(desc);
    if (!node_) {
      NNDEPLOY_LOGE("Node creation failed for node_key: %s\n",
                    node_key_.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
    if (node_->getInputTypeInfo() != this->getInputTypeInfo() ||
        node_->getOutputTypeInfo() != this->getOutputTypeInfo()) {
      NNDEPLOY_LOGE(
          "Type mismatch: Node input/output types do not match BatchPreprocess "
          "types.\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    dag::Edge *output = node_->getOutput();
    std::vector<cv::Mat> *results = output->getAny<std::vector<cv::Mat>>(node_);
    if (results == nullptr) {
      NNDEPLOY_LOGE("results is nullptr");
      return base::kStatusCodeErrorInvalidParam;
    }
    dag::Edge *input = node_->getInput();
    for (int i = 0; i < results->size(); i++) {
      input->setAny((*results)[i]);
      node_->run();
    }
    return base::kStatusCodeOk;
  }

 private:
  std::string node_key_ = "";
  EncodeNode *node_ = nullptr;
};

}  // namespace codec
}  // namespace nndeploy

#endif /* _NNDEPLOY_CODEC_BATCH_CODEC_H_ */
