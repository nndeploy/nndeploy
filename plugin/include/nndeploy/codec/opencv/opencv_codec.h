
#ifndef _NNDEPLOY_CODEC_OPENCV_CODEC_H_
#define _NNDEPLOY_CODEC_OPENCV_CODEC_H_

#include "nndeploy/codec/codec.h"

namespace nndeploy {
namespace codec {

class NNDEPLOY_CC_API OpenCvImageDecode : public Decode {
 public:
  OpenCvImageDecode(const std::string &name) : Decode(name, base::CodecFlag::kCodecFlagImage) {
    key_ = "nndeploy::codec::OpenCvImageDecode";
    desc_ = "Decode image using OpenCV, from image path to cv::Mat, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvImageDecode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs, base::CodecFlag::kCodecFlagImage) {
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
  OpenCvImagesDecode(const std::string &name) : Decode(name, base::CodecFlag::kCodecFlagImages) {
    key_ = "nndeploy::codec::OpenCvImagesDecode";
    desc_ = "Decode multiple images using OpenCV, from image paths to cv::Mat, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvImagesDecode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs, base::CodecFlag::kCodecFlagImages) {
    key_ = "nndeploy::codec::OpenCvImagesDecode";
    desc_ = "Decode multiple images using OpenCV, from image paths to cv::Mat, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvImagesDecode(const std::string &name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::OpenCvImagesDecode";
    desc_ = "Decode multiple images using OpenCV, from image paths to cv::Mat, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvImagesDecode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvImagesDecode";
    desc_ = "Decode multiple images using OpenCV, from image paths to cv::Mat, default color space is BGR";
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
  OpenCvVedioDecode(const std::string &name) : Decode(name, base::CodecFlag::kCodecFlagVideo) {
    key_ = "nndeploy::codec::OpenCvVedioDecode";
    desc_ = "Decode video using OpenCV, from video file to cv::Mat frames, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvVedioDecode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs, base::CodecFlag::kCodecFlagVideo) {
    key_ = "nndeploy::codec::OpenCvVedioDecode";
    desc_ = "Decode video using OpenCV, from video file to cv::Mat frames, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
  }

  OpenCvVedioDecode(const std::string &name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::OpenCvVedioDecode";
    desc_ = "Decode video using OpenCV, from video file to cv::Mat frames, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvVedioDecode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvVedioDecode";
    desc_ = "Decode video using OpenCV, from video file to cv::Mat frames, default color space is BGR";
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
  OpenCvCameraDecode(const std::string &name) : Decode(name, base::CodecFlag::kCodecFlagCamera) {
    key_ = "nndeploy::codec::OpenCvCameraDecode";
    desc_ = "Decode camera stream using OpenCV, from camera device to cv::Mat frames, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvCameraDecode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs, base::CodecFlag::kCodecFlagCamera) {
    key_ = "nndeploy::codec::OpenCvCameraDecode";
    desc_ = "Decode camera stream using OpenCV, from camera device to cv::Mat frames, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvCameraDecode(const std::string &name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::OpenCvCameraDecode";
    desc_ = "Decode camera stream using OpenCV, from camera device to cv::Mat frames, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
  }
  OpenCvCameraDecode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvCameraDecode";
    desc_ = "Decode camera stream using OpenCV, from camera device to cv::Mat frames, default color space is BGR";
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

class NNDEPLOY_CC_API OpenCvImageEncode : public Encode {
 public:
  OpenCvImageEncode(const std::string &name) : Encode(name, base::CodecFlag::kCodecFlagImage) {
    key_ = "nndeploy::codec::OpenCvImageEncode";
    desc_ = "Encode image using OpenCV, from cv::Mat to image file, supports common image formats";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvImageEncode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs)
      : Encode(name, inputs, outputs, base::CodecFlag::kCodecFlagImage) {
    key_ = "nndeploy::codec::OpenCvImageEncode";
    desc_ = "Encode image using OpenCV, from cv::Mat to image file, supports common image formats";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvImageEncode(const std::string &name, base::CodecFlag flag)
      : Encode(name, flag) {
    key_ = "nndeploy::codec::OpenCvImageEncode";
    desc_ = "Encode image using OpenCV, from cv::Mat to image file, supports common image formats";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvImageEncode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Encode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvImageEncode";
    desc_ = "Encode image using OpenCV, from cv::Mat to image file, supports common image formats";
    this->setInputTypeInfo<cv::Mat>();
  }
  virtual ~OpenCvImageEncode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status setRefPath(const std::string &ref_path) override;
  virtual base::Status setPath(const std::string &path) override;

  virtual base::Status run();
};

class NNDEPLOY_CC_API OpenCvImagesEncode : public Encode {
 public:
  OpenCvImagesEncode(const std::string &name) : Encode(name) {
    key_ = "nndeploy::codec::OpenCvImagesEncode";
    desc_ = "Encode multiple images using OpenCV, from cv::Mat to image files, supports common image formats";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvImagesEncode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs)
      : Encode(name, inputs, outputs, base::CodecFlag::kCodecFlagImages) {
    key_ = "nndeploy::codec::OpenCvImagesEncode";
    desc_ = "Encode multiple images using OpenCV, from cv::Mat to image files, supports common image formats";
    this->setInputTypeInfo<cv::Mat>();
  }

  OpenCvImagesEncode(const std::string &name, base::CodecFlag flag)
      : Encode(name, flag) {
    key_ = "nndeploy::codec::OpenCvImagesEncode";
    desc_ = "Encode multiple images using OpenCV, from cv::Mat to image files, supports common image formats";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvImagesEncode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Encode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvImagesEncode";
    desc_ = "Encode multiple images using OpenCV, from cv::Mat to image files, supports common image formats";
    this->setInputTypeInfo<cv::Mat>();
  }
  virtual ~OpenCvImagesEncode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status setRefPath(const std::string &ref_path) override;
  virtual base::Status setPath(const std::string &path) override;

  virtual base::Status run();
};

class NNDEPLOY_CC_API OpenCvVedioEncode : public Encode {
 public:
  OpenCvVedioEncode(const std::string &name) : Encode(name, base::CodecFlag::kCodecFlagVideo) {
    key_ = "nndeploy::codec::OpenCvVedioEncode";
    desc_ = "Encode video using OpenCV, from cv::Mat frames to video file, supports common video formats";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvVedioEncode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs)
      : Encode(name, inputs, outputs, base::CodecFlag::kCodecFlagVideo) {
    key_ = "nndeploy::codec::OpenCvVedioEncode";
    desc_ = "Encode video using OpenCV, from cv::Mat frames to video file, supports common video formats";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvVedioEncode(const std::string &name, base::CodecFlag flag)
      : Encode(name, flag) {
    key_ = "nndeploy::codec::OpenCvVedioEncode";
    desc_ = "Encode video using OpenCV, from cv::Mat frames to video file, supports common video formats";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvVedioEncode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Encode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvVedioEncode";
    desc_ = "Encode video using OpenCV, from cv::Mat frames to video file, supports common video formats";
    this->setInputTypeInfo<cv::Mat>();
  }
  virtual ~OpenCvVedioEncode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status setRefPath(const std::string &ref_path) override;
  virtual base::Status setPath(const std::string &path) override;

  virtual base::Status run();

 private:
  cv::VideoCapture *cap_ = nullptr;
  cv::VideoWriter *writer_ = nullptr;
};

class NNDEPLOY_CC_API OpenCvCameraEncode : public Encode {
 public:
  OpenCvCameraEncode(const std::string &name) : Encode(name, base::CodecFlag::kCodecFlagCamera) {
    key_ = "nndeploy::codec::OpenCvCameraEncode";
    desc_ = "Encode camera stream using OpenCV, from cv::Mat frames to video output, supports common video formats";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvCameraEncode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs)
      : Encode(name, inputs, outputs, base::CodecFlag::kCodecFlagCamera) {
    key_ = "nndeploy::codec::OpenCvCameraEncode";
    desc_ = "Encode camera stream using OpenCV, from cv::Mat frames to video output, supports common video formats";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvCameraEncode(const std::string &name, base::CodecFlag flag)
      : Encode(name, flag) {
    key_ = "nndeploy::codec::OpenCvCameraEncode";
    desc_ = "Encode camera stream using OpenCV, from cv::Mat frames to video output, supports common video formats";
    this->setInputTypeInfo<cv::Mat>();
  }
  OpenCvCameraEncode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Encode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::OpenCvCameraEncode";
    desc_ = "Encode camera stream using OpenCV, from cv::Mat frames to video output, supports common video formats";
    this->setInputTypeInfo<cv::Mat>();
  }
  virtual ~OpenCvCameraEncode() {}

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status setRefPath(const std::string &ref_path) override;
  virtual base::Status setPath(const std::string &path) override;

  virtual base::Status run();
};

extern NNDEPLOY_CC_API Decode *createOpenCvDecode(base::CodecFlag flag,
                                   const std::string &name, dag::Edge *output);

extern NNDEPLOY_CC_API std::shared_ptr<Decode> createOpenCvDecodeSharedPtr(
    base::CodecFlag flag, const std::string &name, dag::Edge *output);

extern NNDEPLOY_CC_API Encode *createOpenCvEncode(base::CodecFlag flag,
                                   const std::string &name, dag::Edge *input);

extern NNDEPLOY_CC_API std::shared_ptr<Encode> createOpenCvEncodeSharedPtr(
    base::CodecFlag flag, const std::string &name, dag::Edge *input);

}  // namespace codec
}  // namespace nndeploy

#endif /* _NNDEPLOY_CODEC_OPENCV_CODEC_H_ */
