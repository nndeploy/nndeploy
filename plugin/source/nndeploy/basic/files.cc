#include "nndeploy/basic/files.h"

namespace nndeploy {
namespace basic {
  
REGISTER_NODE("nndeploy::basic::InputCppTextFile", InputCppTextFile);
REGISTER_NODE("nndeploy::basic::InputCppBinaryFile", InputCppBinaryFile);

REGISTER_NODE("nndeploy::basic::OutputCppTextFile", OutputCppTextFile);
REGISTER_NODE("nndeploy::basic::OutputCppBinaryFile", OutputCppBinaryFile);

}  // namespace basic
}  // namespace nndeploy
