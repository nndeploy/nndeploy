#include "nndeploy/basic/end.h"

namespace nndeploy {
namespace basic {

base::Status End::run() { return base::Status::Ok(); }
  
REGISTER_NODE("nndeploy::basic::End", End);

}  // namespace basic
}  // namespace nndeploy
