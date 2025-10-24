#ifndef _NNDEPLOY_CODEC_FFMPEG_CODEC_H_
#define _NNDEPLOY_CODEC_FFMPEG_CODEC_H_

#include "nndeploy/base/ffmpeg_include.h"
#include "nndeploy/codec/codec.h"

namespace nndeploy {
namespace codec {

class NNDEPLOY_CC_API FFmpegImageDecode : public Decode {
 public:
  FFmpegImageDecode(const std::string &name)
      : Decode(name, base::CodecFlag::kCodecFlagImage) {
    key_ = "nndeploy::codec::FFmpegImageDecode";
    desc_ = "Decode image using FFmpeg, from image path to AVFrame";
    this->setOutputTypeInfo<AVFrame>();
  }
  FFmpegImageDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs) {
    key_ = "nndeploy::codec::FFmpegImageDecode";
    desc_ = "Decode image using FFmpeg, from image path to AVFrame";
    this->setOutputTypeInfo<AVFrame>();
  }
  FFmpegImageDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::FFmpegImageDecode";
    desc_ = "Decode image using FFmpeg, from image path to AVFrame";
    this->setOutputTypeInfo<AVFrame>();
  }

  virtual ~FFmpegImageDecode() override = default;

  base::Status init() override;

  base::Status deinit() override;

  base::Status run() override;

  base::Status setPath(const std::string &path) override;
};

class NNDEPLOY_CC_API FFmpegImageEncode : public Encode {
 public:
  FFmpegImageEncode(const std::string &name)
      : Encode(name, base::kCodecFlagImage) {
    key_ = "nndeploy::codec::FFmpegImageEncode";
    desc_ = "Encode image using FFmpeg, from AVFrame to image file";
    this->setInputTypeInfo<AVFrame>();
  }
  FFmpegImageEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs)
      : Encode(name, inputs, outputs, base::kCodecFlagImage) {
    key_ = "nndeploy::codec::FFmpegImageEncode";
    desc_ = "Encode image using FFmpeg, from AVFrame to image file";
    this->setInputTypeInfo<AVFrame>();
  }
  FFmpegImageEncode(const std::string &name, base::CodecFlag flag)
      : Encode(name, flag) {
    key_ = "nndeploy::codec::FFmpegImageEncode";
    desc_ = "Encode image using FFmpeg, from AVFrame to image file";
    this->setInputTypeInfo<AVFrame>();
  }
  FFmpegImageEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Encode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::FFmpegImageEncode";
    desc_ = "Encode image using FFmpeg, from AVFrame to image file";
    this->setInputTypeInfo<AVFrame>();
  }

  ~FFmpegImageEncode() override = default;
};

}  // namespace codec
}  // namespace nndeploy

#endif