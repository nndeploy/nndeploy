
#include "nndeploy/codec/ffmpeg/batch_ffmpeg_codec.h"

namespace nndeploy {
namespace codec {

REGISTER_NODE("nndeploy::codec::BatchFFmpegDecode", BatchFFmpegDecode);
REGISTER_NODE("nndeploy::codec::BatchFFmpegEncode", BatchFFmpegEncode);

}  // namespace codec
}  // namespace nndeploy
