#include "nndeploy/dag/base.h"

namespace nndeploy {
namespace dag {

std::string nodeTypeToString(NodeType node_type) {
  switch (node_type) {
    case NodeType::kNodeTypeInput:
      return "Input";
    case NodeType::kNodeTypeOutput:
      return "Output";
    case NodeType::kNodeTypeIntermediate:
      return "Intermediate";
    default:
      return "Intermediate";
  }
}

NodeType stringToNodeType(const std::string& node_type_str) {
  if (node_type_str == "Input") {
    return NodeType::kNodeTypeInput;
  } else if (node_type_str == "Output") {
    return NodeType::kNodeTypeOutput;
  } else {
    return NodeType::kNodeTypeIntermediate;
  }
}

std::string edgeTypeToString(EdgeTypeFlag edge_type) {
  switch (edge_type) {
    case EdgeTypeFlag::kBuffer:
      return "Buffer";
    case EdgeTypeFlag::kCvMat:
      return "CvMat";
    case EdgeTypeFlag::kTensor:
      return "Tensor";
    case EdgeTypeFlag::kParam:
      return "Param";
    case EdgeTypeFlag::kAny:
      return "Any";
    case EdgeTypeFlag::kNone:
      return "None";
    default:
      return "Unknown";
  }
}

EdgeTypeFlag stringToEdgeType(const std::string& edge_type_str) {
  if (edge_type_str == "Buffer") {
    return EdgeTypeFlag::kBuffer;
  } else if (edge_type_str == "CvMat") {
    return EdgeTypeFlag::kCvMat;
  } else if (edge_type_str == "Tensor") {
    return EdgeTypeFlag::kTensor;
  } else if (edge_type_str == "Param") {
    return EdgeTypeFlag::kParam;
  } else if (edge_type_str == "Any") {
    return EdgeTypeFlag::kAny;
  } else if (edge_type_str == "None") {
    return EdgeTypeFlag::kNone;
  } else {
    return EdgeTypeFlag::kNone;
  }
}

}  // namespace dag
}  // namespace nndeploy
