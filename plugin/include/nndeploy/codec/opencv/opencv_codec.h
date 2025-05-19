
#ifndef _NNDEPLOY_CODEC_OPENCV_CODEC_H_
#define _NNDEPLOY_CODEC_OPENCV_CODEC_H_

#include "nndeploy/codec/codec.h"

namespace nndeploy {
namespace codec {

class NNDEPLOY_CC_API OpenCvImageDecodeNode : public DecodeNode {
 public:
  OpenCvImageDecodeNode(const std::string &name) : DecodeNode(name) {
    key_ = "nndeploy::codec::OpenCvImageDecodeNode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvImageDecodeNode(const std::string &name, std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs)
      : DecodeNode(name, inputs, outputs) {
    key_ = "nndeploy::codec::OpenCvImageDecodeNode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvImageDecodeNode(const std::string &name, base::CodecFlag flag)
      : DecodeNode(name, flag) {
    key_ = "nndeploy::codec::OpenCvImageDecodeNode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvImageDecodeNode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : DecodeNode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvImageDecodeNode";
    this->setOutputTypeInfo<cv::Mat>();
  }

  virtual ~OpenCvImageDecodeNode() {};

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();
};

class NNDEPLOY_CC_API OpenCvImagesDecodeNode : public DecodeNode {
 public:
  OpenCvImagesDecodeNode(const std::string &name) : DecodeNode(name) {  
    key_ = "nndeploy::codec::OpenCvImagesDecodeNode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvImagesDecodeNode(const std::string &name, std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs)
      : DecodeNode(name, inputs, outputs) {
    key_ = "nndeploy::codec::OpenCvImagesDecodeNode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvImagesDecodeNode(const std::string &name, base::CodecFlag flag)
      : DecodeNode(name, flag) {
    key_ = "nndeploy::codec::OpenCvImagesDecodeNode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvImagesDecodeNode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : DecodeNode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvImagesDecodeNode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~OpenCvImagesDecodeNode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();

 private:
  base::Status callOnce();

 private:
  std::vector<std::string> images_;
};

class NNDEPLOY_CC_API OpenCvVedioDecodeNode : public DecodeNode {
 public:
  OpenCvVedioDecodeNode(const std::string &name) : DecodeNode(name) { 
    key_ = "nndeploy::codec::OpenCvVedioDecodeNode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvVedioDecodeNode(const std::string &name, std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs)
      : DecodeNode(name, inputs, outputs) {
    key_ = "nndeploy::codec::OpenCvVedioDecodeNode";
    this->setOutputTypeInfo<cv::Mat>();
  }

  OpenCvVedioDecodeNode(const std::string &name, base::CodecFlag flag)
      : DecodeNode(name, flag) {
    key_ = "nndeploy::codec::OpenCvVedioDecodeNode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvVedioDecodeNode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : DecodeNode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvVedioDecodeNode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~OpenCvVedioDecodeNode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();

 private:
  cv::VideoCapture *cap_ = nullptr;
};

class NNDEPLOY_CC_API OpenCvCameraDecodeNode : public DecodeNode {
 public:
  OpenCvCameraDecodeNode(const std::string &name) : DecodeNode(name) {    
    key_ = "nndeploy::codec::OpenCvCameraDecodeNode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvCameraDecodeNode(const std::string &name, std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs)
      : DecodeNode(name, inputs, outputs) {
    key_ = "nndeploy::codec::OpenCvCameraDecodeNode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvCameraDecodeNode(const std::string &name, base::CodecFlag flag)
      : DecodeNode(name, flag) {
    key_ = "nndeploy::codec::OpenCvCameraDecodeNode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvCameraDecodeNode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : DecodeNode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvCameraDecodeNode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~OpenCvCameraDecodeNode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();

 private:
  cv::VideoCapture *cap_ = nullptr;
};

class NNDEPLOY_CC_API OpenCvImageEncodeNode : public EncodeNode {
 public:
  OpenCvImageEncodeNode(const std::string &name) : EncodeNode(name) {
    key_ = "nndeploy::codec::OpenCvImageEncodeNode";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvImageEncodeNode(const std::string &name, std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs)
      : EncodeNode(name, inputs, outputs) {
    key_ = "nndeploy::codec::OpenCvImageEncodeNode";  
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvImageEncodeNode(const std::string &name, base::CodecFlag flag)
      : EncodeNode(name, flag) {
    key_ = "nndeploy::codec::OpenCvImageEncodeNode";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvImageEncodeNode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : EncodeNode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvImageEncodeNode";
    this->setInputTypeInfo<cv::Mat>();
  }
  virtual ~OpenCvImageEncodeNode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();
};

class NNDEPLOY_CC_API OpenCvImagesEncodeNode : public EncodeNode {
 public:
  OpenCvImagesEncodeNode(const std::string &name) : EncodeNode(name) {
    key_ = "nndeploy::codec::OpenCvImagesEncodeNode";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvImagesEncodeNode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs)
      : EncodeNode(name, inputs, outputs) {
    key_ = "nndeploy::codec::OpenCvImagesEncodeNode";
    this->setInputTypeInfo<cv::Mat>();
  }

  OpenCvImagesEncodeNode(const std::string &name, base::CodecFlag flag)
      : EncodeNode(name, flag) {
    key_ = "nndeploy::codec::OpenCvImagesEncodeNode";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvImagesEncodeNode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : EncodeNode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvImagesEncodeNode";
    this->setInputTypeInfo<cv::Mat>();
  }
  virtual ~OpenCvImagesEncodeNode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();
};

class NNDEPLOY_CC_API OpenCvVedioEncodeNode : public EncodeNode {
 public:
  OpenCvVedioEncodeNode(const std::string &name) : EncodeNode(name) {
    key_ = "nndeploy::codec::OpenCvVedioEncodeNode";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvVedioEncodeNode(const std::string &name, std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs)
      : EncodeNode(name, inputs, outputs) {
    key_ = "nndeploy::codec::OpenCvVedioEncodeNode";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvVedioEncodeNode(const std::string &name, base::CodecFlag flag)
      : EncodeNode(name, flag) {
    key_ = "nndeploy::codec::OpenCvVedioEncodeNode";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvVedioEncodeNode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : EncodeNode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvVedioEncodeNode";
    this->setInputTypeInfo<cv::Mat>();
  }
  virtual ~OpenCvVedioEncodeNode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();

 private:
  cv::VideoCapture *cap_ = nullptr;
  cv::VideoWriter *writer_ = nullptr;
};

class NNDEPLOY_CC_API OpenCvCameraEncodeNode : public EncodeNode {
 public:
  OpenCvCameraEncodeNode(const std::string &name) : EncodeNode(name) {
    key_ = "nndeploy::codec::OpenCvCameraEncodeNode";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvCameraEncodeNode(const std::string &name, std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs)
      : EncodeNode(name, inputs, outputs) {
    key_ = "nndeploy::codec::OpenCvCameraEncodeNode";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvCameraEncodeNode(const std::string &name, base::CodecFlag flag)
      : EncodeNode(name, flag) {
    key_ = "nndeploy::codec::OpenCvCameraEncodeNode";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvCameraEncodeNode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : EncodeNode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvCameraEncodeNode";
    this->setInputTypeInfo<cv::Mat>();
  }
  virtual ~OpenCvCameraEncodeNode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();
};

DecodeNode *createOpenCvDecodeNode(base::CodecFlag flag,
                                   const std::string &name, dag::Edge *output);

std::shared_ptr<DecodeNode> createOpenCvDecodeNodeSharedPtr(
    base::CodecFlag flag, const std::string &name, dag::Edge *output);

EncodeNode *createOpenCvEncodeNode(base::CodecFlag flag,
                                   const std::string &name, dag::Edge *input);

std::shared_ptr<EncodeNode> createOpenCvEncodeNodeSharedPtr(
    base::CodecFlag flag, const std::string &name, dag::Edge *input);

}  // namespace codec
}  // namespace nndeploy

#endif /* _NNDEPLOY_CODEC_OPENCV_CODEC_H_ */
