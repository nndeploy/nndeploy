
#ifndef _NNDEPLOY_CODEC_OPENCV_CODEC_H_
#define _NNDEPLOY_CODEC_OPENCV_CODEC_H_

#include "nndeploy/codec/codec.h"

namespace nndeploy {
namespace codec {

class NNDEPLOY_CC_API OpenCvImageDecode : public Decode {
 public:
  OpenCvImageDecode(const std::string &name) : Decode(name) {
    key_ = "nndeploy::codec::OpenCvImageDecode";
    desc_ = "Decode image using OpenCV, from image path to cv::Mat, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvImageDecode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs) {
    key_ = "nndeploy::codec::OpenCvImageDecode";
    desc_ = "Decode image using OpenCV, from image path to cv::Mat, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvImageDecode(const std::string &name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::OpenCvImageDecode";
    desc_ = "Decode image using OpenCV, from image path to cv::Mat, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvImageDecode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvImageDecode";
    desc_ = "Decode image using OpenCV, from image path to cv::Mat, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
  }

  virtual ~OpenCvImageDecode() {};

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status setPath(const std::string &path) override;
  virtual base::Status run();
};

class NNDEPLOY_CC_API OpenCvImagesDecode : public Decode {
 public:
  OpenCvImagesDecode(const std::string &name) : Decode(name) {
    key_ = "nndeploy::codec::OpenCvImagesDecode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvImagesDecode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs) {
    key_ = "nndeploy::codec::OpenCvImagesDecode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvImagesDecode(const std::string &name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::OpenCvImagesDecode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvImagesDecode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvImagesDecode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~OpenCvImagesDecode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status setPath(const std::string &path) override;
  virtual base::Status run();

 private:
  base::Status callOnce();

 private:
  std::vector<std::string> images_;
};

class NNDEPLOY_CC_API OpenCvVedioDecode : public Decode {
 public:
  OpenCvVedioDecode(const std::string &name) : Decode(name) {
    key_ = "nndeploy::codec::OpenCvVedioDecode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvVedioDecode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs) {
    key_ = "nndeploy::codec::OpenCvVedioDecode";
    this->setOutputTypeInfo<cv::Mat>();
  }

  OpenCvVedioDecode(const std::string &name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::OpenCvVedioDecode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvVedioDecode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvVedioDecode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~OpenCvVedioDecode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status setPath(const std::string &path) override;
  virtual base::Status run();

 private:
  cv::VideoCapture *cap_ = nullptr;
};

class NNDEPLOY_CC_API OpenCvCameraDecode : public Decode {
 public:
  OpenCvCameraDecode(const std::string &name) : Decode(name) {
    key_ = "nndeploy::codec::OpenCvCameraDecode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvCameraDecode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs) {
    key_ = "nndeploy::codec::OpenCvCameraDecode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvCameraDecode(const std::string &name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::OpenCvCameraDecode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvCameraDecode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvCameraDecode";
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~OpenCvCameraDecode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status setPath(const std::string &path) override;
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
  OpenCvImageEncodeNode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
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

  virtual base::Status setRefPath(const std::string &ref_path) override;
  virtual base::Status setPath(const std::string &path) override;

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

  virtual base::Status setRefPath(const std::string &ref_path) override;
  virtual base::Status setPath(const std::string &path) override;

  virtual base::Status run();
};

class NNDEPLOY_CC_API OpenCvVedioEncodeNode : public EncodeNode {
 public:
  OpenCvVedioEncodeNode(const std::string &name) : EncodeNode(name) {
    key_ = "nndeploy::codec::OpenCvVedioEncodeNode";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvVedioEncodeNode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
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

  virtual base::Status setRefPath(const std::string &ref_path) override;
  virtual base::Status setPath(const std::string &path) override;

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
  OpenCvCameraEncodeNode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
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

  virtual base::Status setRefPath(const std::string &ref_path) override;
  virtual base::Status setPath(const std::string &path) override;

  virtual base::Status run();
};

Decode *createOpenCvDecode(base::CodecFlag flag,
                                   const std::string &name, dag::Edge *output);

std::shared_ptr<Decode> createOpenCvDecodeSharedPtr(
    base::CodecFlag flag, const std::string &name, dag::Edge *output);

EncodeNode *createOpenCvEncodeNode(base::CodecFlag flag,
                                   const std::string &name, dag::Edge *input);

std::shared_ptr<EncodeNode> createOpenCvEncodeNodeSharedPtr(
    base::CodecFlag flag, const std::string &name, dag::Edge *input);

}  // namespace codec
}  // namespace nndeploy

#endif /* _NNDEPLOY_CODEC_OPENCV_CODEC_H_ */
