
#include "nndeploy/qwen/qwen.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace qwen {

NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(PrefillEmbeddingNode);
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(DecodeEmbeddingNode);
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(PrefillSampleNode);

}  // namespace qwen
}  // namespace nndeploy
