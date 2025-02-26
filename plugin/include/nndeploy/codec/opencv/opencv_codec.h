
#ifndef _NNDEPLOY_CODEC_OPENCV_CODEC_H_
#define _NNDEPLOY_CODEC_OPENCV_CODEC_H_

#include "nndeploy/codec/codec.h"

namespace nndeploy {
namespace codec {

class NNDEPLOY_CC_API OpenCvImageDecodeNode : public DecodeNode {
 public:
  OpenCvImageDecodeNode(base::CodecFlag flag, const std::string &name,
                        dag::Edge *output)
      : DecodeNode(flag, name, output) {};
  virtual ~OpenCvImageDecodeNode() {};

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();
};

class NNDEPLOY_CC_API OpenCvImagesDecodeNode : public DecodeNode {
 public:
  OpenCvImagesDecodeNode(base::CodecFlag flag, const std::string &name,
                         dag::Edge *output)
      : DecodeNode(flag, name, output) {}
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
  OpenCvVedioDecodeNode(base::CodecFlag flag, const std::string &name,
                        dag::Edge *output)
      : DecodeNode(flag, name, output) {}
  virtual ~OpenCvVedioDecodeNode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();

 private:
  cv::VideoCapture *cap_ = nullptr;
};

class NNDEPLOY_CC_API OpenCvCameraDecodeNode : public DecodeNode {
 public:
  OpenCvCameraDecodeNode(base::CodecFlag flag, const std::string &name,
                         dag::Edge *output)
      : DecodeNode(flag, name, output) {}
  virtual ~OpenCvCameraDecodeNode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();

 private:
  cv::VideoCapture *cap_ = nullptr;
};

class NNDEPLOY_CC_API OpenCvImageEncodeNode : public EncodeNode {
 public:
  OpenCvImageEncodeNode(base::CodecFlag flag, const std::string &name,
                        dag::Edge *input)
      : EncodeNode(flag, name, input) {}
  virtual ~OpenCvImageEncodeNode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();
};

class NNDEPLOY_CC_API OpenCvImagesEncodeNode : public EncodeNode {
 public:
  OpenCvImagesEncodeNode(base::CodecFlag flag, const std::string &name,
                         dag::Edge *input)
      : EncodeNode(flag, name, input) {}
  virtual ~OpenCvImagesEncodeNode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();
};

class NNDEPLOY_CC_API OpenCvVedioEncodeNode : public EncodeNode {
 public:
  OpenCvVedioEncodeNode(base::CodecFlag flag, const std::string &name,
                        dag::Edge *input)
      : EncodeNode(flag, name, input) {}
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
  OpenCvCameraEncodeNode(base::CodecFlag flag, const std::string &name,
                         dag::Edge *input)
      : EncodeNode(flag, name, input) {}
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
