// DrawDepth 节点编译单元
// 所有方法已在 drawdepth.h 中内联实现，此处仅注册节点
#include "nndeploy/depth/drawdepth.h"

namespace nndeploy {
namespace depth {

// 注册 DrawDepth 节点，JSON 工作流通过 "nndeploy::depth::DrawDepth" 实例化
REGISTER_NODE("nndeploy::depth::DrawDepth", DrawDepth);

}  // namespace depth
}  // namespace nndeploy
