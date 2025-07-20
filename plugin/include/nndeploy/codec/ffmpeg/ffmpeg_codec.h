
#ifndef _NNDEPLOY_CODEC_FFMPEG_CODEC_H_
#define _NNDEPLOY_CODEC_FFMPEG_CODEC_H_

#include "nndeploy/codec/codec.h"
#include "nndeploy/base/ffmpeg_include.h"

namespace nndeploy {
namespace codec {
class NNDEPLOY_CC_API FFmpegImageDecode : public Decode {
 public:
  FFmpegImageDecode(const std::string &name) : Decode(name) {
    key_ = "nndeploy::codec::FFmpegImageDecode";
    desc_ = "Decode image using FFmpeg, from image path to AVFrame";
    this->setOutputTypeInfo<AVFrame>();
  }
  FFmpegImageDecode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs)
      : Decode(name, inputs, outputs) {
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
  FFmpegImageDecode(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs, base::CodecFlag flag)
      : Decode(name, inputs, outputs, flag) {
    key_ = "nndeploy::codec::FFmpegImageDecode";
    desc_ = "Decode image using FFmpeg, from image path to AVFrame";
    this->setOutputTypeInfo<AVFrame>();
  }

  virtual ~FFmpegImageDecode() {};

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status setPath(const std::string &path) override;
  virtual base::Status run();
 private:
  AVFormatContext* format_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVFrame *frame_ = nullptr;
  AVPacket packet_;
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