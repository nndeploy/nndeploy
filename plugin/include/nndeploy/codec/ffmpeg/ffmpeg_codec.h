
#ifndef _NNDEPLOY_CODEC_FFMPEG_CODEC_H_
#define _NNDEPLOY_CODEC_FFMPEG_CODEC_H_

#include "nndeploy/codec/codec.h"
#include "nndeploy/codec/ffmpeg/ffmpeg_hw_codec.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace nndeploy {
namespace codec {

class NNDEPLOY_CC_API FFmpegImageDecode : public Decode {
 public:
  FFmpegImageDecode(const std::string& name)
      : Decode(name, base::CodecFlag::kCodecFlagImage) {
    key_ = "nndeploy::codec::FFmpegImageDecode";
    desc_ =
        "Decode image using FFmpeg, from image path to cv::Mat, default color "
        "space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
  }
  FFmpegImageDecode(const std::string& name, std::vector<dag::Edge*> inputs,
                    std::vector<dag::Edge*> outputs)
      : Decode(name, inputs, outputs, base::CodecFlag::kCodecFlagImage) {
    key_ = "nndeploy::codec::FFmpegImageDecode";
    desc_ =
        "Decode image using FFmpeg, from image path to cv::Mat, default color "
        "space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
  }
  FFmpegImageDecode(const std::string& name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::FFmpegImageDecode";
    desc_ =
        "Decode image using FFmpeg, from image path to cv::Mat, default color "
        "space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
  }
  FFmpegImageDecode(const std::string& name, std::vector<dag::Edge*> inputs,
                    std::vector<dag::Edge*> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::FFmpegImageDecode";
    desc_ =
        "Decode image using FFmpeg, from image path to cv::Mat, default color "
        "space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
  }

  virtual ~FFmpegImageDecode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setPath(const std::string& path) override;
  virtual base::Status run();

  void setHwDeviceType(base::FFmpegHWDeviceType hw_type);
  void setSwCodecConfig(const FFmpegSWCodecConfig& config);

 private:
  AVFormatContext* fmt_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVFrame* frame_ = nullptr;
  AVPacket* pkt_ = nullptr;
  SwsContext* sws_ctx_ = nullptr;
  FFmpegHWCodec* hw_codec_ = nullptr;
  FFmpegSWCodecConfig sw_codec_config_ = FFmpegSWCodecConfig::getDefaultForArch();
  int video_stream_idx_ = -1;
};

class NNDEPLOY_CC_API FFmpegImagesDecode : public Decode {
 public:
  FFmpegImagesDecode(const std::string& name)
      : Decode(name, base::CodecFlag::kCodecFlagImages) {
    key_ = "nndeploy::codec::FFmpegImagesDecode";
    desc_ =
        "Decode multiple images using FFmpeg, from image paths to cv::Mat, "
        "default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeDir);
  }
  FFmpegImagesDecode(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs)
      : Decode(name, inputs, outputs, base::CodecFlag::kCodecFlagImages) {
    key_ = "nndeploy::codec::FFmpegImagesDecode";
    desc_ =
        "Decode multiple images using FFmpeg, from image paths to cv::Mat, "
        "default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeDir);
  }
  FFmpegImagesDecode(const std::string& name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::FFmpegImagesDecode";
    desc_ =
        "Decode multiple images using FFmpeg, from image paths to cv::Mat, "
        "default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
  }
  FFmpegImagesDecode(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::FFmpegImagesDecode";
    desc_ =
        "Decode multiple images using FFmpeg, from image paths to cv::Mat, "
        "default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
  }
  virtual ~FFmpegImagesDecode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setPath(const std::string& path) override;
  virtual base::Status run();

 private:
  std::vector<std::string> images_;
  AVFormatContext* fmt_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVFrame* frame_ = nullptr;
  AVPacket* pkt_ = nullptr;
  SwsContext* sws_ctx_ = nullptr;
  int video_stream_idx_ = -1;
};

class NNDEPLOY_CC_API FFmpegVideoDecode : public Decode {
 public:
  FFmpegVideoDecode(const std::string& name)
      : Decode(name, base::CodecFlag::kCodecFlagVideo) {
    key_ = "nndeploy::codec::FFmpegVideoDecode";
    desc_ =
        "Decode video using FFmpeg, from video file to cv::Mat frames, default "
        "color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
  }
  FFmpegVideoDecode(const std::string& name, std::vector<dag::Edge*> inputs,
                    std::vector<dag::Edge*> outputs)
      : Decode(name, inputs, outputs, base::CodecFlag::kCodecFlagVideo) {
    key_ = "nndeploy::codec::FFmpegVideoDecode";
    desc_ =
        "Decode video using FFmpeg, from video file to cv::Mat frames, default "
        "color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
  }

  FFmpegVideoDecode(const std::string& name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::FFmpegVideoDecode";
    desc_ =
        "Decode video using FFmpeg, from video file to cv::Mat frames, default "
        "color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
  }
  FFmpegVideoDecode(const std::string& name, std::vector<dag::Edge*> inputs,
                    std::vector<dag::Edge*> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::FFmpegVideoDecode";
    desc_ =
        "Decode video using FFmpeg, from video file to cv::Mat frames, default "
        "color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
  }
  virtual ~FFmpegVideoDecode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setPath(const std::string& path) override;
  virtual base::Status run();

  void setHwDeviceType(base::FFmpegHWDeviceType hw_type);

  // Configure software decode path with architecture-aware tuning.
  // Call before setPath(). If sw config sets prefer_hw_first=false,
  // the HW decoder attempt is skipped entirely.
  void setSwCodecConfig(const FFmpegSWCodecConfig& config);

 private:
  AVFormatContext* fmt_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVFrame* frame_ = nullptr;
  AVPacket* pkt_ = nullptr;
  SwsContext* sws_ctx_ = nullptr;
  FFmpegHWCodec* hw_codec_ = nullptr;
  FFmpegSWCodecConfig sw_codec_config_ = FFmpegSWCodecConfig::getDefaultForArch();
  int video_stream_idx_ = -1;
};

class NNDEPLOY_CC_API FFmpegCameraDecode : public Decode {
 public:
  FFmpegCameraDecode(const std::string& name)
      : Decode(name, base::CodecFlag::kCodecFlagCamera) {
    key_ = "nndeploy::codec::FFmpegCameraDecode";
    desc_ =
        "Decode camera stream using FFmpeg, from camera device to cv::Mat "
        "frames, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
  }
  FFmpegCameraDecode(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs)
      : Decode(name, inputs, outputs, base::CodecFlag::kCodecFlagCamera) {
    key_ = "nndeploy::codec::FFmpegCameraDecode";
    desc_ =
        "Decode camera stream using FFmpeg, from camera device to cv::Mat "
        "frames, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
  }
  FFmpegCameraDecode(const std::string& name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::FFmpegCameraDecode";
    desc_ =
        "Decode camera stream using FFmpeg, from camera device to cv::Mat "
        "frames, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
  }
  FFmpegCameraDecode(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::FFmpegCameraDecode";
    desc_ =
        "Decode camera stream using FFmpeg, from camera device to cv::Mat "
        "frames, default color space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
  }
  virtual ~FFmpegCameraDecode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setPath(const std::string& path) override;
  virtual base::Status run();

  void setHwDeviceType(base::FFmpegHWDeviceType hw_type);
  void setSwCodecConfig(const FFmpegSWCodecConfig& config);

 private:
  AVFormatContext* fmt_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVFrame* frame_ = nullptr;
  AVPacket* pkt_ = nullptr;
  SwsContext* sws_ctx_ = nullptr;
  FFmpegHWCodec* hw_codec_ = nullptr;
  FFmpegSWCodecConfig sw_codec_config_ = FFmpegSWCodecConfig::getDefaultForArch();
  int video_stream_idx_ = -1;
};

class NNDEPLOY_CC_API FFmpegImageEncode : public Encode {
 public:
  FFmpegImageEncode(const std::string& name)
      : Encode(name, base::CodecFlag::kCodecFlagImage) {
    key_ = "nndeploy::codec::FFmpegImageEncode";
    desc_ =
        "Encode image using FFmpeg, from cv::Mat to image file, supports "
        "common image formats (JPEG, PNG, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
    path_ = "resources/images/output.jpg";
  }
  FFmpegImageEncode(const std::string& name, std::vector<dag::Edge*> inputs,
                    std::vector<dag::Edge*> outputs)
      : Encode(name, inputs, outputs, base::CodecFlag::kCodecFlagImage) {
    key_ = "nndeploy::codec::FFmpegImageEncode";
    desc_ =
        "Encode image using FFmpeg, from cv::Mat to image file, supports "
        "common image formats (JPEG, PNG, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
    path_ = "resources/images/output.jpg";
  }
  FFmpegImageEncode(const std::string& name, base::CodecFlag flag)
      : Encode(name, flag) {
    key_ = "nndeploy::codec::FFmpegImageEncode";
    desc_ =
        "Encode image using FFmpeg, from cv::Mat to image file, supports "
        "common image formats (JPEG, PNG, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
    path_ = "resources/images/output.jpg";
  }
  FFmpegImageEncode(const std::string& name, std::vector<dag::Edge*> inputs,
                    std::vector<dag::Edge*> outputs, base::CodecFlag flag)
      : Encode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::FFmpegImageEncode";
    desc_ =
        "Encode image using FFmpeg, from cv::Mat to image file, supports "
        "common image formats (JPEG, PNG, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeImage);
    path_ = "resources/images/output.jpg";
  }
  virtual ~FFmpegImageEncode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setRefPath(const std::string& ref_path) override;
  virtual base::Status setPath(const std::string& path) override;
  virtual base::Status run();

 private:
  AVFormatContext* fmt_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVFrame* frame_ = nullptr;
  AVPacket* pkt_ = nullptr;
  SwsContext* sws_ctx_ = nullptr;
  bool initialized_ = false;
};

class NNDEPLOY_CC_API FFmpegImagesEncode : public Encode {
 public:
  FFmpegImagesEncode(const std::string& name) : Encode(name) {
    key_ = "nndeploy::codec::FFmpegImagesEncode";
    desc_ =
        "Encode multiple images using FFmpeg, from cv::Mat to image files, "
        "supports common image formats (JPEG, PNG, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeDir);
    path_ = "resources/images";
  }
  FFmpegImagesEncode(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs)
      : Encode(name, inputs, outputs, base::CodecFlag::kCodecFlagImages) {
    key_ = "nndeploy::codec::FFmpegImagesEncode";
    desc_ =
        "Encode multiple images using FFmpeg, from cv::Mat to image files, "
        "supports common image formats (JPEG, PNG, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeDir);
    path_ = "resources/images";
  }

  FFmpegImagesEncode(const std::string& name, base::CodecFlag flag)
      : Encode(name, flag) {
    key_ = "nndeploy::codec::FFmpegImagesEncode";
    desc_ =
        "Encode multiple images using FFmpeg, from cv::Mat to image files, "
        "supports common image formats (JPEG, PNG, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeDir);
    path_ = "resources/images";
  }
  FFmpegImagesEncode(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs, base::CodecFlag flag)
      : Encode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::FFmpegImagesEncode";
    desc_ =
        "Encode multiple images using FFmpeg, from cv::Mat to image files, "
        "supports common image formats (JPEG, PNG, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeDir);
    path_ = "resources/images";
  }
  virtual ~FFmpegImagesEncode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setRefPath(const std::string& ref_path) override;
  virtual base::Status setPath(const std::string& path) override;
  virtual base::Status run();

 private:
  AVFormatContext* fmt_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVFrame* frame_ = nullptr;
  AVPacket* pkt_ = nullptr;
  SwsContext* sws_ctx_ = nullptr;
  bool initialized_ = false;
};

class NNDEPLOY_CC_API FFmpegVideoEncode : public Encode {
 public:
  FFmpegVideoEncode(const std::string& name)
      : Encode(name, base::CodecFlag::kCodecFlagVideo) {
    key_ = "nndeploy::codec::FFmpegVideoEncode";
    desc_ =
        "Encode video using FFmpeg, from cv::Mat frames to video file, "
        "supports common video formats (MP4, MKV, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
    path_ = "resources/videos/output.mp4";
  }
  FFmpegVideoEncode(const std::string& name, std::vector<dag::Edge*> inputs,
                    std::vector<dag::Edge*> outputs)
      : Encode(name, inputs, outputs, base::CodecFlag::kCodecFlagVideo) {
    key_ = "nndeploy::codec::FFmpegVideoEncode";
    desc_ =
        "Encode video using FFmpeg, from cv::Mat frames to video file, "
        "supports common video formats (MP4, MKV, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
    path_ = "resources/videos/output.mp4";
  }
  FFmpegVideoEncode(const std::string& name, base::CodecFlag flag)
      : Encode(name, flag) {
    key_ = "nndeploy::codec::FFmpegVideoEncode";
    desc_ =
        "Encode video using FFmpeg, from cv::Mat frames to video file, "
        "supports common video formats (MP4, MKV, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
    path_ = "resources/videos/output.mp4";
  }
  FFmpegVideoEncode(const std::string& name, std::vector<dag::Edge*> inputs,
                    std::vector<dag::Edge*> outputs, base::CodecFlag flag)
      : Encode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::FFmpegVideoEncode";
    desc_ =
        "Encode video using FFmpeg, from cv::Mat frames to video file, "
        "supports common video formats (MP4, MKV, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeVideo);
    path_ = "resources/videos/output.mp4";
  }
  virtual ~FFmpegVideoEncode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setRefPath(const std::string& ref_path) override;
  virtual base::Status setPath(const std::string& path) override;
  virtual base::Status run();

  void setHwDeviceType(base::FFmpegHWDeviceType hw_type);
  void setSwCodecConfig(const FFmpegSWCodecConfig& config);

 private:
  AVFormatContext* fmt_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVFrame* frame_ = nullptr;
  AVPacket* pkt_ = nullptr;
  SwsContext* sws_ctx_ = nullptr;
  FFmpegHWCodec* hw_codec_ = nullptr;
  FFmpegSWCodecConfig sw_codec_config_ = FFmpegSWCodecConfig::getDefaultForArch();
  bool initialized_ = false;
};

class NNDEPLOY_CC_API FFmpegCameraEncode : public Encode {
 public:
  FFmpegCameraEncode(const std::string& name)
      : Encode(name, base::CodecFlag::kCodecFlagCamera) {
    key_ = "nndeploy::codec::FFmpegCameraEncode";
    desc_ =
        "Encode camera stream using FFmpeg, from cv::Mat frames to video "
        "output, supports common video formats (MP4, MKV, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "resources/videos/camera_out.mp4";
  }
  FFmpegCameraEncode(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs)
      : Encode(name, inputs, outputs, base::CodecFlag::kCodecFlagCamera) {
    key_ = "nndeploy::codec::FFmpegCameraEncode";
    desc_ =
        "Encode camera stream using FFmpeg, from cv::Mat frames to video "
        "output, supports common video formats (MP4, MKV, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "resources/videos/camera_out.mp4";
  }
  FFmpegCameraEncode(const std::string& name, base::CodecFlag flag)
      : Encode(name, flag) {
    key_ = "nndeploy::codec::FFmpegCameraEncode";
    desc_ =
        "Encode camera stream using FFmpeg, from cv::Mat frames to video "
        "output, supports common video formats (MP4, MKV, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "resources/videos/camera_out.mp4";
  }
  FFmpegCameraEncode(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs, base::CodecFlag flag)
      : Encode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::FFmpegCameraEncode";
    desc_ =
        "Encode camera stream using FFmpeg, from cv::Mat frames to video "
        "output, supports common video formats (MP4, MKV, etc.)";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "resources/videos/camera_out.mp4";
  }
  virtual ~FFmpegCameraEncode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setRefPath(const std::string& ref_path) override;
  virtual base::Status setPath(const std::string& path) override;
  virtual base::Status run();

  void setHwDeviceType(base::FFmpegHWDeviceType hw_type);
  void setSwCodecConfig(const FFmpegSWCodecConfig& config);

 private:
  AVFormatContext* fmt_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVFrame* frame_ = nullptr;
  AVPacket* pkt_ = nullptr;
  SwsContext* sws_ctx_ = nullptr;
  FFmpegHWCodec* hw_codec_ = nullptr;
  FFmpegSWCodecConfig sw_codec_config_ = FFmpegSWCodecConfig::getDefaultForArch();
  bool initialized_ = false;
};

class NNDEPLOY_CC_API FFmpegStreamDecode : public Decode {
 public:
  FFmpegStreamDecode(const std::string& name)
      : Decode(name, base::CodecFlag::kCodecFlagStreaming) {
    key_ = "nndeploy::codec::FFmpegStreamDecode";
    desc_ =
        "Decode real-time video stream using FFmpeg, from network RTSP/HTTP "
        "stream to cv::Mat frames, with auto-reconnection, default color "
        "space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    size_ = INT_MAX;
    loop_count_ = size_;
  }
  FFmpegStreamDecode(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs)
      : Decode(name, inputs, outputs, base::CodecFlag::kCodecFlagStreaming) {
    key_ = "nndeploy::codec::FFmpegStreamDecode";
    desc_ =
        "Decode real-time video stream using FFmpeg, from network RTSP/HTTP "
        "stream to cv::Mat frames, with auto-reconnection, default color "
        "space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    size_ = INT_MAX;
    loop_count_ = size_;
  }
  FFmpegStreamDecode(const std::string& name, base::CodecFlag flag)
      : Decode(name, flag) {
    key_ = "nndeploy::codec::FFmpegStreamDecode";
    desc_ =
        "Decode real-time video stream using FFmpeg, from network RTSP/HTTP "
        "stream to cv::Mat frames, with auto-reconnection, default color "
        "space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    size_ = INT_MAX;
    loop_count_ = size_;
  }
  FFmpegStreamDecode(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::FFmpegStreamDecode";
    desc_ =
        "Decode real-time video stream using FFmpeg, from network RTSP/HTTP "
        "stream to cv::Mat frames, with auto-reconnection, default color "
        "space is BGR";
    this->setOutputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    size_ = INT_MAX;
    loop_count_ = size_;
  }
  virtual ~FFmpegStreamDecode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setPath(const std::string& path) override;
  virtual base::Status run();

  void setHwDeviceType(base::FFmpegHWDeviceType hw_type);
  void setSwCodecConfig(const FFmpegSWCodecConfig& config);

 private:
  AVFormatContext* fmt_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVFrame* frame_ = nullptr;
  AVPacket* pkt_ = nullptr;
  SwsContext* sws_ctx_ = nullptr;
  FFmpegHWCodec* hw_codec_ = nullptr;
  FFmpegSWCodecConfig sw_codec_config_ = FFmpegSWCodecConfig::getDefaultForArch();
  int video_stream_idx_ = -1;
  int reconnect_attempts_ = 3;
  int reconnect_delay_ms_ = 1000;
};

class NNDEPLOY_CC_API FFmpegStreamEncode : public Encode {
 public:
  FFmpegStreamEncode(const std::string& name)
      : Encode(name, base::CodecFlag::kCodecFlagStreaming) {
    key_ = "nndeploy::codec::FFmpegStreamEncode";
    desc_ =
        "Encode real-time video stream using FFmpeg, from cv::Mat frames to "
        "network stream output, supports RTSP/HTTP streaming output";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "output_stream.mp4";
  }
  FFmpegStreamEncode(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs)
      : Encode(name, inputs, outputs, base::CodecFlag::kCodecFlagStreaming) {
    key_ = "nndeploy::codec::FFmpegStreamEncode";
    desc_ =
        "Encode real-time video stream using FFmpeg, from cv::Mat frames to "
        "network stream output, supports RTSP/HTTP streaming output";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "output_stream.mp4";
  }
  FFmpegStreamEncode(const std::string& name, base::CodecFlag flag)
      : Encode(name, flag) {
    key_ = "nndeploy::codec::FFmpegStreamEncode";
    desc_ =
        "Encode real-time video stream using FFmpeg, from cv::Mat frames to "
        "network stream output, supports RTSP/HTTP streaming output";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "output_stream.mp4";
  }
  FFmpegStreamEncode(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs, base::CodecFlag flag)
      : Encode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::FFmpegStreamEncode";
    desc_ =
        "Encode real-time video stream using FFmpeg, from cv::Mat frames to "
        "network stream output, supports RTSP/HTTP streaming output";
    this->setInputTypeInfo<cv::Mat>();
    this->setIoType(dag::IOType::kIOTypeCamera);
    path_ = "output_stream.mp4";
  }
  virtual ~FFmpegStreamEncode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status setRefPath(const std::string& ref_path) override;
  virtual base::Status setPath(const std::string& path) override;
  virtual base::Status run();

  void setHwDeviceType(base::FFmpegHWDeviceType hw_type);
  void setSwCodecConfig(const FFmpegSWCodecConfig& config);

 private:
  FFmpegHWCodec* hw_codec_ = nullptr;
  FFmpegSWCodecConfig sw_codec_config_ = FFmpegSWCodecConfig::getDefaultForArch();
};

extern NNDEPLOY_CC_API Decode* createFFmpegDecode(base::CodecFlag flag,
                                                  const std::string& name,
                                                  dag::Edge* output);

extern NNDEPLOY_CC_API std::shared_ptr<Decode> createFFmpegDecodeSharedPtr(
    base::CodecFlag flag, const std::string& name, dag::Edge* output);

extern NNDEPLOY_CC_API Encode* createFFmpegEncode(base::CodecFlag flag,
                                                  const std::string& name,
                                                  dag::Edge* input);

extern NNDEPLOY_CC_API std::shared_ptr<Encode> createFFmpegEncodeSharedPtr(
    base::CodecFlag flag, const std::string& name, dag::Edge* input);

}  // namespace codec
}  // namespace nndeploy

#endif /* _NNDEPLOY_CODEC_FFMPEG_CODEC_H_ */
