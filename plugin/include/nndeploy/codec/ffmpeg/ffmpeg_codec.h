
#ifndef _NNDEPLOY_CODEC_FFMPEG_CODEC_H_
#define _NNDEPLOY_CODEC_FFMPEG_CODEC_H_

#include "nndeploy/base/ffmpeg_include.h"
#include "nndeploy/codec/codec.h"

namespace nndeploy {
namespace codec {
class NNDEPLOY_CC_API FFmpegImageDecode : public Decode {
 public:
  FFmpegImageDecode(const std::string &name) : Decode(name, base::kCodecFlagImage) {
    key_ = "nndeploy::codec::FFmpegImageDecode";
    desc_ = "Decode image using FFmpeg, from image path to AVFrame";
    this->setOutputTypeInfo<AVFrame>();
  }
  FFmpegImageDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs, base::kCodecFlagImage) {
    key_ = "nndeploy::codec::FFmpegImageDecode";
    desc_ = "Decode image using FFmpeg, from image path to AVFrame";
    this->setOutputTypeInfo<AVFrame>();
  }
  FFmpegImageDecode(const std::string &name, base::CodecFlag flag)
      : Decode(name, flag) {
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

  ~FFmpegImageDecode() override = default;

  base::Status init() override;
  base::Status deinit() override;

  base::Status setPath(const std::string &path) override;
  base::Status run() override;
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

  base::Status init() override;
  base::Status deinit() override;

  base::Status setPath(const std::string &path) override;
  base::Status setRefPath(const std::string &ref_path) override;
  base::Status run() override;

 private:
  base::Status encodeFrameToFile(AVFrame *frame);
};

class NNDEPLOY_CC_API FFmpegAudioDecode : public Decode {};
class NNDEPLOY_CC_API FFmpegVideoDecode : public Decode {};

extern NNDEPLOY_CC_API Decode *createFFmpegDecode(base::CodecFlag flag,
                                   const std::string &name, dag::Edge *output);

extern NNDEPLOY_CC_API std::shared_ptr<Decode> createFFmpegDecodeSharedPtr(
    base::CodecFlag flag, const std::string &name, dag::Edge *output);

extern NNDEPLOY_CC_API Encode *createFFmpegEncode(base::CodecFlag flag,
                                   const std::string &name, dag::Edge *input);

extern NNDEPLOY_CC_API std::shared_ptr<Encode> createFFmpegEncodeSharedPtr(
    base::CodecFlag flag, const std::string &name, dag::Edge *input);


}
}
#endif /* _NNDEPLOY_CODEC_FFMPEG_CODEC_H_ */
