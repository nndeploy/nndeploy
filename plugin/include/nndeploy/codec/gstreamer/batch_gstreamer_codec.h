
#ifndef _NNDEPLOY_CODEC_GSTREAMER_BATCH_CODEC_H_
#define _NNDEPLOY_CODEC_GSTREAMER_BATCH_CODEC_H_

#include "nndeploy/codec/gstreamer/gstreamer_codec.h"

namespace nndeploy {
namespace codec {

class NNDEPLOY_CC_API BatchGStreamerDecode : public dag::CompositeNode {
 public:
  BatchGStreamerDecode(const std::string &name) : dag::CompositeNode(name) {
    key_ = "nndeploy::codec::BatchGStreamerDecode";
    node_type_ = dag::NodeType::kNodeTypeInput;
    this->setOutputTypeInfo<std::vector<cv::Mat>>();
    desc_ =
        "BatchGStreamerDecode node for decoding batches of images/videos using "
        "GStreamer";
  }
  BatchGStreamerDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs)
      : dag::CompositeNode(name, inputs, outputs) {
    key_ = "nndeploy::codec::BatchGStreamerDecode";
    node_type_ = dag::NodeType::kNodeTypeInput;
    this->setOutputTypeInfo<std::vector<cv::Mat>>();
    desc_ =
        "BatchGStreamerDecode node for decoding batches of images/videos using "
        "GStreamer";
  }
  virtual ~BatchGStreamerDecode() {
    if (this->getInitialized()) {
      this->deinit();
      this->setInitializedFlag(false);
    }
  }

  void setBatchSize(int batch_size) { batch_size_ = batch_size; }
  base::Status setNodeKey(const std::string &key) {
    base::Status status = base::kStatusCodeOk;
    status = this->deinit();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("deinit failed");
      return status;
    }
    node_key_ = key;
    std::vector<std::string> input_names = this->getInputNames();
    std::vector<std::string> output_names = this->getRealOutputsName();
    dag::NodeDesc desc(node_key_, "inner_decode_node", input_names,
                       output_names);
    node_ = (Decode *)this->createNode(desc);
    if (!node_) {
      NNDEPLOY_LOGE("Node creation failed for node_key: %s\n",
                    node_key_.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
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
    return base::kCodecFlagImage;
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
      size_ = NNDEPLOY_UP_DIV((int)(node_->getSize()), batch_size_);
    }
    return size_;
  }

  double getFps() {
    if (node_) {
      return node_->getFps();
    }
    return 0.0;
  }
  int getWidth() {
    if (node_) {
      return node_->getWidth();
    }
    return 0;
  }
  int getHeight() {
    if (node_) {
      return node_->getHeight();
    }
    return 0;
  }

  int getLoopCount() {
    if (node_) {
      loop_count_ = NNDEPLOY_UP_DIV((int)(node_->getLoopCount()), batch_size_);
    }
    return loop_count_;
  }

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

  virtual base::Status run() {
    auto results = new std::vector<cv::Mat>();
    if (index_ >= size_) {
      outputs_[0]->set(results, false);
      return base::kStatusCodeOk;
    }
    for (int i = 0; i < batch_size_; i++) {
      cv::Mat *single = nullptr;
      if (index_ * batch_size_ + i < node_->getSize()) {
        node_->run();
        dag::Edge *output = node_->getOutput();
        single = output->getCvMat(node_);
        if (single == nullptr) {
          NNDEPLOY_LOGE("single_tensor is nullptr");
          return base::kStatusCodeErrorInvalidParam;
        }
      }
      if (single != nullptr && !single->empty()) {
        cv::Mat res(single->rows, single->cols, single->type());
        single->copyTo(res);
        results->push_back(res);
      }
    }
    outputs_[0]->set(results, false);
    index_++;
    return base::kStatusCodeOk;
  }

  virtual base::Status serialize(
      rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
    base::Status status = dag::CompositeNode::serialize(json, allocator);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    json.AddMember("batch_size_", batch_size_, allocator);
    json.AddMember(
        "node_key_",
        rapidjson::Value(node_key_.c_str(), node_key_.length(), allocator),
        allocator);
    return base::kStatusCodeOk;
  }
  virtual base::Status deserialize(rapidjson::Value &json) {
    base::Status status = dag::CompositeNode::deserialize(json);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    if (json.HasMember("batch_size_") && json["batch_size_"].IsInt()) {
      batch_size_ = json["batch_size_"].GetInt();
    }
    if (json.HasMember("node_key_") && json["node_key_"].IsString()) {
      std::string node_key_str = json["node_key_"].GetString();
      this->setNodeKey(node_key_str);
    }
    return base::kStatusCodeOk;
  }

 private:
  int batch_size_ = 1;
  int index_ = 0;
  int size_ = 0;
  std::string node_key_ = "";
  Decode *node_ = nullptr;
};

class NNDEPLOY_CC_API BatchGStreamerEncode : public dag::CompositeNode {
 public:
  BatchGStreamerEncode(const std::string &name) : dag::CompositeNode(name) {
    key_ = "nndeploy::codec::BatchGStreamerEncode";
    node_type_ = dag::NodeType::kNodeTypeOutput;
    this->setInputTypeInfo<std::vector<cv::Mat>>();
    desc_ =
        "BatchGStreamerEncode node for encoding batches of images/videos using "
        "GStreamer";
  }
  BatchGStreamerEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs)
      : dag::CompositeNode(name, inputs, outputs) {
    key_ = "nndeploy::codec::BatchGStreamerEncode";
    node_type_ = dag::NodeType::kNodeTypeOutput;
    this->setInputTypeInfo<std::vector<cv::Mat>>();
    desc_ =
        "BatchGStreamerEncode node for encoding batches of images/videos using "
        "GStreamer";
  }
  virtual ~BatchGStreamerEncode() {
    if (this->getInitialized()) {
      this->deinit();
      this->setInitializedFlag(false);
    }
  }

  base::Status setNodeKey(const std::string &key) {
    base::Status status = base::kStatusCodeOk;
    status = this->deinit();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("deinit failed");
      return status;
    }
    node_key_ = key;
    dag::NodeDesc desc(node_key_, "inner_codec_node",
                       {"inner_codec_node.input"}, {});
    node_ = (Encode *)this->createNode(desc);
    if (!node_) {
      NNDEPLOY_LOGE("Node creation failed for node_key: %s\n",
                    node_key_.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
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
    return base::kCodecFlagImage;
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
  void setSize(int size) {
    if (node_) {
      node_->setSize(size);
    }
  }
  int getSize() {
    if (node_) {
      return node_->getSize();
    }
    return 0;
  }
  int getIndex() {
    if (node_) {
      return node_->getIndex();
    }
    return 0;
  }

  virtual base::Status run() {
    std::vector<cv::Mat> *cv_mats =
        inputs_[0]->get<std::vector<cv::Mat>>(this);
    if (cv_mats == nullptr) {
      NNDEPLOY_LOGE("cv_mats is nullptr");
      return base::kStatusCodeErrorInvalidParam;
    }
    dag::Edge *node_input = node_->getInput();
    for (int i = 0; i < cv_mats->size(); i++) {
      node_input->set((*cv_mats)[i]);
      node_->run();
    }
    return base::kStatusCodeOk;
  }

  virtual base::Status serialize(
      rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
    base::Status status = dag::CompositeNode::serialize(json, allocator);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    json.AddMember(
        "node_key_",
        rapidjson::Value(node_key_.c_str(), node_key_.length(), allocator),
        allocator);
    return base::kStatusCodeOk;
  }
  virtual base::Status deserialize(rapidjson::Value &json) {
    base::Status status = dag::CompositeNode::deserialize(json);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    if (json.HasMember("node_key_") && json["node_key_"].IsString()) {
      std::string node_key_str = json["node_key_"].GetString();
      this->setNodeKey(node_key_str);
    }
    return base::kStatusCodeOk;
  }

 private:
  std::string node_key_ = "";
  Encode *node_ = nullptr;
};

}  // namespace codec
}  // namespace nndeploy

#endif /* _NNDEPLOY_CODEC_GSTREAMER_BATCH_CODEC_H_ */
