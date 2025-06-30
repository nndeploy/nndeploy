
#ifndef _NNDEPLOY_CODEC_FFMPEG_CODEC_H_
#define _NNDEPLOY_CODEC_FFMPEG_CODEC_H_

#include "nndeploy/codec/codec.h"

namespace nndeploy {
namespace codec {
class NNDEPLOY_CC_API FFmpegImageDecode : public Decode {};

class NNDEPLOY_CC_API FFmpegAudioDecode : public Decode {};
class NNDEPLOY_CC_API FFmpegVideoDecode : public Decode {};

}
}
#endif /* _NNDEPLOY_CODEC_FFMPEG_CODEC_H_ */