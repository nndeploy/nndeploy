#include "nndeploy/codec/opencv/batch_opencv_codec.h"

#include "nndeploy/base/file.h"

namespace nndeploy {
namespace codec {

REGISTER_NODE("nndeploy::codec::BatchOpenCvDecode", BatchOpenCvDecode);
REGISTER_NODE("nndeploy::codec::BatchOpenCvEncode", BatchOpenCvEncode);

}  // namespace codec
}  // namespace nndeploy