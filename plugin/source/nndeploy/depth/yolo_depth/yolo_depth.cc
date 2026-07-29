// YoloDepthGraph 编译单元
// 所有方法已在 yolo_depth.h 中内联实现（与 YoloGraph、DepthGraph 等代码库惯例一致）
// 本文件仅包含 REGISTER_NODE 宏，需要一个编译单元完成符号注册

#include "nndeploy/depth/yolo_depth/yolo_depth.h"

namespace nndeploy {
namespace depth {

// 注册 YoloDepthGraph 节点，JSON 工作流通过 "nndeploy::depth::YoloDepthGraph" 实例化
REGISTER_NODE("nndeploy::depth::YoloDepthGraph", YoloDepthGraph);

}  // namespace depth
}  // namespace nndeploy
