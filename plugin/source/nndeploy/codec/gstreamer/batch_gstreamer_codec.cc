
#include "nndeploy/codec/gstreamer/batch_gstreamer_codec.h"

namespace nndeploy {
namespace codec {

REGISTER_NODE("nndeploy::codec::BatchGStreamerDecode", BatchGStreamerDecode);
REGISTER_NODE("nndeploy::codec::BatchGStreamerEncode", BatchGStreamerEncode);

}  // namespace codec
}  // namespace nndeploy
