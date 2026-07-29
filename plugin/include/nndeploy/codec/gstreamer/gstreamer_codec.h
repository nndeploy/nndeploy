
#ifndef _NNDEPLOY_CODEC_GSTREAMER_CODEC_H_
#define _NNDEPLOY_CODEC_GSTREAMER_CODEC_H_

#include "nndeploy/codec/codec.h"

#include <gst/gst.h>
#include <gst/app/app.h>

namespace nndeploy {
namespace codec {

class NNDEPLOY_CC_API GStreamerImageDecode : public Decode {
 public:
  GStreamerImageDecode(const std::string &name)
      : Decode(name, base::CodecFlag::kCodecFlagImage) {
    key_ = "nndeploy::codec::GStreamerImageDecode";
    desc_ =
        "Decode image using GStreamer, from image path to cv::Mat, default "
        "color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
  }
  GStreamerImageDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs, base::CodecFlag::kCodecFlagImage) {
    key_ = "nndeploy::codec::GStreamerImageDecode";
    desc_ =
        "Decode image using GStreamer, from image path to cv::Mat, default "
        "color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
  }
  GStreamerImageDecode(const std::string &name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::GStreamerImageDecode";
    desc_ =
        "Decode image using GStreamer, from image path to cv::Mat, default "
        "color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
  }
  GStreamerImageDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::GStreamerImageDecode";
    desc_ =
        "Decode image using GStreamer, from image path to cv::Mat, default "
        "color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
  }

  virtual ~GStreamerImageDecode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setPath(const std::string &path) override;
  virtual base::Status run();

 private:
  GstElement *pipeline_ = nullptr;
  GstElement *source_ = nullptr;
  GstElement *decode_ = nullptr;
  GstElement *convert_ = nullptr;
  GstElement *sink_ = nullptr;
  GstSample *sample_ = nullptr;
  std::mutex sample_mutex_;
  cv::Mat last_mat_;
};

class NNDEPLOY_CC_API GStreamerImagesDecode : public Decode {
 public:
  GStreamerImagesDecode(const std::string &name)
      : Decode(name, base::CodecFlag::kCodecFlagImages) {
    key_ = "nndeploy::codec::GStreamerImagesDecode";
    desc_ =
        "Decode multiple images using GStreamer, from image paths to cv::Mat, "
        "default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeDir);
  }
  GStreamerImagesDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                     std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs, base::CodecFlag::kCodecFlagImages) {
    key_ = "nndeploy::codec::GStreamerImagesDecode";
    desc_ =
        "Decode multiple images using GStreamer, from image paths to cv::Mat, "
        "default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeDir);
  }
  GStreamerImagesDecode(const std::string &name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::GStreamerImagesDecode";
    desc_ =
        "Decode multiple images using GStreamer, from image paths to cv::Mat, "
        "default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
  }
  GStreamerImagesDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                     std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::GStreamerImagesDecode";
    desc_ =
        "Decode multiple images using GStreamer, from image paths to cv::Mat, "
        "default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
  }
  virtual ~GStreamerImagesDecode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setPath(const std::string &path) override;
  virtual base::Status run();

 private:
  std::vector<std::string> images_;
  GstElement *pipeline_ = nullptr;
  GstSample *sample_ = nullptr;
  std::mutex sample_mutex_;
};

class NNDEPLOY_CC_API GStreamerVideoDecode : public Decode {
 public:
  GStreamerVideoDecode(const std::string &name)
      : Decode(name, base::CodecFlag::kCodecFlagVideo) {
    key_ = "nndeploy::codec::GStreamerVideoDecode";
    desc_ =
        "Decode video using GStreamer, from video file to cv::Mat frames, "
        "default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
  }
  GStreamerVideoDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs, base::CodecFlag::kCodecFlagVideo) {
    key_ = "nndeploy::codec::GStreamerVideoDecode";
    desc_ =
        "Decode video using GStreamer, from video file to cv::Mat frames, "
        "default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
  }

  GStreamerVideoDecode(const std::string &name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::GStreamerVideoDecode";
    desc_ =
        "Decode video using GStreamer, from video file to cv::Mat frames, "
        "default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
  }
  GStreamerVideoDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::GStreamerVideoDecode";
    desc_ =
        "Decode video using GStreamer, from video file to cv::Mat frames, "
        "default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
  }
  virtual ~GStreamerVideoDecode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setPath(const std::string &path) override;
  virtual base::Status run();

 private:
  GstElement *pipeline_ = nullptr;
  GstSample *sample_ = nullptr;
  std::mutex sample_mutex_;
  GstBus *bus_ = nullptr;
  gint width_gs_ = 0;
  gint height_gs_ = 0;
};

class NNDEPLOY_CC_API GStreamerCameraDecode : public Decode {
 public:
  GStreamerCameraDecode(const std::string &name)
      : Decode(name, base::CodecFlag::kCodecFlagCamera) {
    key_ = "nndeploy::codec::GStreamerCameraDecode";
    desc_ =
        "Decode camera stream using GStreamer, from camera device to cv::Mat "
        "frames, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
  }
  GStreamerCameraDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                     std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs, base::CodecFlag::kCodecFlagCamera) {
    key_ = "nndeploy::codec::GStreamerCameraDecode";
    desc_ =
        "Decode camera stream using GStreamer, from camera device to cv::Mat "
        "frames, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
  }
  GStreamerCameraDecode(const std::string &name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::GStreamerCameraDecode";
    desc_ =
        "Decode camera stream using GStreamer, from camera device to cv::Mat "
        "frames, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
  }
  GStreamerCameraDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                     std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::GStreamerCameraDecode";
    desc_ =
        "Decode camera stream using GStreamer, from camera device to cv::Mat "
        "frames, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
  }
  virtual ~GStreamerCameraDecode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setPath(const std::string &path) override;
  virtual base::Status run();

 private:
  GstElement *pipeline_ = nullptr;
  GstSample *sample_ = nullptr;
  std::mutex sample_mutex_;
  GstBus *bus_ = nullptr;
};

class NNDEPLOY_CC_API GStreamerImageEncode : public Encode {
 public:
  GStreamerImageEncode(const std::string &name)
      : Encode(name, base::CodecFlag::kCodecFlagImage) {
    key_ = "nndeploy::codec::GStreamerImageEncode";
    desc_ =
        "Encode image using GStreamer, from cv::Mat to image file, supports "
        "common image formats (JPEG, PNG, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
    path_ = "resources/images/output.jpg";
  }
  GStreamerImageEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs)
      : Encode(name, inputs, outputs, base::CodecFlag::kCodecFlagImage) {
    key_ = "nndeploy::codec::GStreamerImageEncode";
    desc_ =
        "Encode image using GStreamer, from cv::Mat to image file, supports "
        "common image formats (JPEG, PNG, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
    path_ = "resources/images/output.jpg";
  }
  GStreamerImageEncode(const std::string &name, base::CodecFlag flag)
      : Encode(name, flag) {
    key_ = "nndeploy::codec::GStreamerImageEncode";
    desc_ =
        "Encode image using GStreamer, from cv::Mat to image file, supports "
        "common image formats (JPEG, PNG, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
    path_ = "resources/images/output.jpg";
  }
  GStreamerImageEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Encode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::GStreamerImageEncode";
    desc_ =
        "Encode image using GStreamer, from cv::Mat to image file, supports "
        "common image formats (JPEG, PNG, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
    path_ = "resources/images/output.jpg";
  }
  virtual ~GStreamerImageEncode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setRefPath(const std::string &ref_path) override;
  virtual base::Status setPath(const std::string &path) override;
  virtual base::Status run();

 private:
  GstElement *pipeline_ = nullptr;
  GstElement *source_ = nullptr;
  GstElement *convert_ = nullptr;
  GstElement *encode_ = nullptr;
  GstElement *sink_ = nullptr;
  GstBuffer *pending_buffer_ = nullptr;
  std::mutex buffer_mutex_;
  std::condition_variable buffer_cv_;
  bool eos_ = false;
};

class NNDEPLOY_CC_API GStreamerImagesEncode : public Encode {
 public:
  GStreamerImagesEncode(const std::string &name) : Encode(name) {
    key_ = "nndeploy::codec::GStreamerImagesEncode";
    desc_ =
        "Encode multiple images using GStreamer, from cv::Mat to image files, "
        "supports common image formats (JPEG, PNG, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeDir);
    path_ = "resources/images";
  }
  GStreamerImagesEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                     std::vector<dag::Edge *> outputs)
      : Encode(name, inputs, outputs, base::CodecFlag::kCodecFlagImages) {
    key_ = "nndeploy::codec::GStreamerImagesEncode";
    desc_ =
        "Encode multiple images using GStreamer, from cv::Mat to image files, "
        "supports common image formats (JPEG, PNG, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeDir);
    path_ = "resources/images";
  }

  GStreamerImagesEncode(const std::string &name, base::CodecFlag flag)
      : Encode(name, flag) {
    key_ = "nndeploy::codec::GStreamerImagesEncode";
    desc_ =
        "Encode multiple images using GStreamer, from cv::Mat to image files, "
        "supports common image formats (JPEG, PNG, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeDir);
    path_ = "resources/images";
  }
  GStreamerImagesEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                     std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Encode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::GStreamerImagesEncode";
    desc_ =
        "Encode multiple images using GStreamer, from cv::Mat to image files, "
        "supports common image formats (JPEG, PNG, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeDir);
    path_ = "resources/images";
  }
  virtual ~GStreamerImagesEncode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setRefPath(const std::string &ref_path) override;
  virtual base::Status setPath(const std::string &path) override;
  virtual base::Status run();

 private:
  int index_gs_ = 0;
};

class NNDEPLOY_CC_API GStreamerVideoEncode : public Encode {
 public:
  GStreamerVideoEncode(const std::string &name)
      : Encode(name, base::CodecFlag::kCodecFlagVideo) {
    key_ = "nndeploy::codec::GStreamerVideoEncode";
    desc_ =
        "Encode video using GStreamer, from cv::Mat frames to video file, "
        "supports common video formats (MP4, MKV, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
    path_ = "resources/videos/output.mp4";
  }
  GStreamerVideoEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs)
      : Encode(name, inputs, outputs, base::CodecFlag::kCodecFlagVideo) {
    key_ = "nndeploy::codec::GStreamerVideoEncode";
    desc_ =
        "Encode video using GStreamer, from cv::Mat frames to video file, "
        "supports common video formats (MP4, MKV, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
    path_ = "resources/videos/output.mp4";
  }
  GStreamerVideoEncode(const std::string &name, base::CodecFlag flag)
      : Encode(name, flag) {
    key_ = "nndeploy::codec::GStreamerVideoEncode";
    desc_ =
        "Encode video using GStreamer, from cv::Mat frames to video file, "
        "supports common video formats (MP4, MKV, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
    path_ = "resources/videos/output.mp4";
  }
  GStreamerVideoEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Encode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::GStreamerVideoEncode";
    desc_ =
        "Encode video using GStreamer, from cv::Mat frames to video file, "
        "supports common video formats (MP4, MKV, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
    path_ = "resources/videos/output.mp4";
  }
  virtual ~GStreamerVideoEncode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setRefPath(const std::string &ref_path) override;
  virtual base::Status setPath(const std::string &path) override;
  virtual base::Status run();

 private:
  GstElement *pipeline_ = nullptr;
  GstElement *source_ = nullptr;
  GstElement *convert_ = nullptr;
  GstElement *encode_ = nullptr;
  GstElement *mux_ = nullptr;
  GstElement *sink_ = nullptr;
  GstBuffer *pending_buffer_ = nullptr;
  std::mutex buffer_mutex_;
  std::condition_variable buffer_cv_;
  bool eos_ = false;
};

class NNDEPLOY_CC_API GStreamerCameraEncode : public Encode {
 public:
  GStreamerCameraEncode(const std::string &name)
      : Encode(name, base::CodecFlag::kCodecFlagCamera) {
    key_ = "nndeploy::codec::GStreamerCameraEncode";
    desc_ =
        "Encode camera stream using GStreamer, from cv::Mat frames to video "
        "output, supports common video formats (MP4, MKV, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "resources/videos/camera_out.mp4";
  }
  GStreamerCameraEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                     std::vector<dag::Edge *> outputs)
      : Encode(name, inputs, outputs, base::CodecFlag::kCodecFlagCamera) {
    key_ = "nndeploy::codec::GStreamerCameraEncode";
    desc_ =
        "Encode camera stream using GStreamer, from cv::Mat frames to video "
        "output, supports common video formats (MP4, MKV, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "resources/videos/camera_out.mp4";
  }
  GStreamerCameraEncode(const std::string &name, base::CodecFlag flag)
      : Encode(name, flag) {
    key_ = "nndeploy::codec::GStreamerCameraEncode";
    desc_ =
        "Encode camera stream using GStreamer, from cv::Mat frames to video "
        "output, supports common video formats (MP4, MKV, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "resources/videos/camera_out.mp4";
  }
  GStreamerCameraEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                     std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Encode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::GStreamerCameraEncode";
    desc_ =
        "Encode camera stream using GStreamer, from cv::Mat frames to video "
        "output, supports common video formats (MP4, MKV, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "resources/videos/camera_out.mp4";
  }
  virtual ~GStreamerCameraEncode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setRefPath(const std::string &ref_path) override;
  virtual base::Status setPath(const std::string &path) override;
  virtual base::Status run();

 private:
  GstElement *pipeline_ = nullptr;
  GstElement *source_ = nullptr;
  GstElement *convert_ = nullptr;
  GstElement *encode_ = nullptr;
  GstElement *mux_ = nullptr;
  GstElement *sink_ = nullptr;
  GstBuffer *pending_buffer_ = nullptr;
  std::mutex buffer_mutex_;
  std::condition_variable buffer_cv_;
  bool eos_ = false;
};

class NNDEPLOY_CC_API GStreamerStreamDecode : public Decode {
 public:
  GStreamerStreamDecode(const std::string &name)
      : Decode(name, base::CodecFlag::kCodecFlagStreaming) {
    key_ = "nndeploy::codec::GStreamerStreamDecode";
    desc_ =
        "Decode real-time video stream using GStreamer, from network RTSP/HTTP "
        "stream to cv::Mat frames, with auto-reconnection";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    size_ = INT_MAX;
    loop_count_ = size_;
  }
  GStreamerStreamDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                     std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs, base::CodecFlag::kCodecFlagStreaming) {
    key_ = "nndeploy::codec::GStreamerStreamDecode";
    desc_ =
        "Decode real-time video stream using GStreamer, from network RTSP/HTTP "
        "stream to cv::Mat frames, with auto-reconnection";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    size_ = INT_MAX;
    loop_count_ = size_;
  }
  GStreamerStreamDecode(const std::string &name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::GStreamerStreamDecode";
    desc_ =
        "Decode real-time video stream using GStreamer, from network RTSP/HTTP "
        "stream to cv::Mat frames, with auto-reconnection";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    size_ = INT_MAX;
    loop_count_ = size_;
  }
  GStreamerStreamDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                     std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::GStreamerStreamDecode";
    desc_ =
        "Decode real-time video stream using GStreamer, from network RTSP/HTTP "
        "stream to cv::Mat frames, with auto-reconnection";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    size_ = INT_MAX;
    loop_count_ = size_;
  }
  virtual ~GStreamerStreamDecode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setPath(const std::string &path) override;
  virtual base::Status run();

 private:
  GstElement *pipeline_ = nullptr;
  GstSample *sample_ = nullptr;
  std::mutex sample_mutex_;
  GstBus *bus_ = nullptr;
  gint width_gs_ = 0;
  gint height_gs_ = 0;
  int reconnect_attempts_ = 3;
  int reconnect_delay_ms_ = 1000;
};

class NNDEPLOY_CC_API GStreamerStreamEncode : public Encode {
 public:
  GStreamerStreamEncode(const std::string &name)
      : Encode(name, base::CodecFlag::kCodecFlagStreaming) {
    key_ = "nndeploy::codec::GStreamerStreamEncode";
    desc_ =
        "Encode real-time video stream using GStreamer, from cv::Mat frames to "
        "network stream output, supports RTSP/HTTP streaming output";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "output_stream.mp4";
  }
  GStreamerStreamEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                     std::vector<dag::Edge *> outputs)
      : Encode(name, inputs, outputs, base::CodecFlag::kCodecFlagStreaming) {
    key_ = "nndeploy::codec::GStreamerStreamEncode";
    desc_ =
        "Encode real-time video stream using GStreamer, from cv::Mat frames to "
        "network stream output, supports RTSP/HTTP streaming output";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "output_stream.mp4";
  }
  GStreamerStreamEncode(const std::string &name, base::CodecFlag flag)
      : Encode(name, flag) {
    key_ = "nndeploy::codec::GStreamerStreamEncode";
    desc_ =
        "Encode real-time video stream using GStreamer, from cv::Mat frames to "
        "network stream output, supports RTSP/HTTP streaming output";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "output_stream.mp4";
  }
  GStreamerStreamEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                     std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Encode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::GStreamerStreamEncode";
    desc_ =
        "Encode real-time video stream using GStreamer, from cv::Mat frames to "
        "network stream output, supports RTSP/HTTP streaming output";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "output_stream.mp4";
  }
  virtual ~GStreamerStreamEncode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setRefPath(const std::string &ref_path) override;
  virtual base::Status setPath(const std::string &path) override;
  virtual base::Status run();

 private:
  GstElement *pipeline_ = nullptr;
  GstElement *source_ = nullptr;
  GstElement *convert_ = nullptr;
  GstElement *encode_ = nullptr;
  GstElement *mux_ = nullptr;
  GstElement *sink_ = nullptr;
  GstBuffer *pending_buffer_ = nullptr;
  std::mutex buffer_mutex_;
  std::condition_variable buffer_cv_;
  bool eos_ = false;
};

extern NNDEPLOY_CC_API Decode *createGStreamerDecode(base::CodecFlag flag,
                                                  const std::string &name,
                                                  dag::Edge *output);

extern NNDEPLOY_CC_API std::shared_ptr<Decode> createGStreamerDecodeSharedPtr(
    base::CodecFlag flag, const std::string &name, dag::Edge *output);

extern NNDEPLOY_CC_API Encode *createGStreamerEncode(base::CodecFlag flag,
                                                  const std::string &name,
                                                  dag::Edge *input);

extern NNDEPLOY_CC_API std::shared_ptr<Encode> createGStreamerEncodeSharedPtr(
    base::CodecFlag flag, const std::string &name, dag::Edge *input);

}  // namespace codec
}  // namespace nndeploy

#endif /* _NNDEPLOY_CODEC_GSTREAMER_CODEC_H_ */
