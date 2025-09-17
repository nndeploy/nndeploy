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

std::string ioTypeToString(IOType io_type) {
  switch (io_type) {
    case IOType::kIOTypeNone:
      return "None";
    case IOType::kIOTypeBool:
      return "Bool";
    case IOType::kIOTypeNum:
      return "Num";
    case IOType::kIOTypeString:
      return "String";
    case IOType::kIOTypeText:
      return "Text";
    case IOType::kIOTypeJson:
      return "Json";
    case IOType::kIOTypeXml:
      return "Xml";
    case IOType::kIOTypeCsv:
      return "Csv";
    case IOType::kIOTypeYaml:
      return "Yaml";
    case IOType::kIOTypeBinary:
      return "Binary";
    case IOType::kIOTypeImage:
      return "Image";
    case IOType::kIOTypeVideo:
      return "Video";
    case IOType::kIOTypeAudio:
      return "Audio";
    case IOType::kIOTypeCamera:
      return "Camera";
    case IOType::kIOTypeMicrophone:
      return "Microphone";
    case IOType::kIOTypeModel:
      return "Model";
    case IOType::kIOTypeDir:
      return "Dir";
    case IOType::kIOTypeAny:
      return "Any";
    default:
      return "None";
  }
}

IOType stringToIoType(const std::string& io_type_str) {
  if (io_type_str == "None") {
    return IOType::kIOTypeNone;
  } else if (io_type_str == "Bool") {
    return IOType::kIOTypeBool;
  } else if (io_type_str == "Num") {
    return IOType::kIOTypeNum;
  } else if (io_type_str == "String") {
    return IOType::kIOTypeString;
  } else if (io_type_str == "Text") {
    return IOType::kIOTypeText;
  } else if (io_type_str == "Json") {
    return IOType::kIOTypeJson;
  } else if (io_type_str == "Xml") {
    return IOType::kIOTypeXml;
  } else if (io_type_str == "Csv") {
    return IOType::kIOTypeCsv;
  } else if (io_type_str == "Yaml") {
    return IOType::kIOTypeYaml;
  } else if (io_type_str == "Binary") {
    return IOType::kIOTypeBinary;
  } else if (io_type_str == "Image") {
    return IOType::kIOTypeImage;
  } else if (io_type_str == "Video") {
    return IOType::kIOTypeVideo;
  } else if (io_type_str == "Audio") {
    return IOType::kIOTypeAudio;
  } else if (io_type_str == "Camera") {
    return IOType::kIOTypeCamera;
  } else if (io_type_str == "Microphone") {
    return IOType::kIOTypeMicrophone;
  } else if (io_type_str == "Model") {
    return IOType::kIOTypeModel;
  } else if (io_type_str == "Dir") {
    return IOType::kIOTypeDir;
  } else if (io_type_str == "Any") {
    return IOType::kIOTypeAny;
  } else {
    return IOType::kIOTypeNone;
  }
}

// std::string edgeTypeToString(EdgeTypeFlag edge_type) {
//   switch (edge_type) {
//     case EdgeTypeFlag::kBuffer:
//       return "Buffer";
//     case EdgeTypeFlag::kCvMat:
//       return "numpy.ndarray";
//     case EdgeTypeFlag::kTensor:
//       return "Tensor";
//     case EdgeTypeFlag::kParam:
//       return "Param";
//     case EdgeTypeFlag::kAny:
//       return "Any";
//     case EdgeTypeFlag::kNone:
//       return "None";
//     default:
//       return "Unknown";
//   }
// }

// EdgeTypeFlag stringToEdgeType(const std::string& edge_type_str) {
//   if (edge_type_str == "Buffer") {
//     return EdgeTypeFlag::kBuffer;
//   } else if (edge_type_str == "numpy.ndarray") {
//     return EdgeTypeFlag::kCvMat;
//   } else if (edge_type_str == "Tensor") {
//     return EdgeTypeFlag::kTensor;
//   } else if (edge_type_str == "Param") {
//     return EdgeTypeFlag::kParam;
//   } else if (edge_type_str == "Any") {
//     return EdgeTypeFlag::kAny;
//   } else if (edge_type_str == "None") {
//     return EdgeTypeFlag::kNone;
//   } else {
//     return EdgeTypeFlag::kNone;
//   }
// }

std::string removeNamespace(const std::string& type_name_with_namespace) {
  std::string name = type_name_with_namespace;

  // 安全的字符串替换函数，检查find()的返回值
  auto safe_replace = [&](const std::string& from, const std::string& to) {
    size_t pos = name.find(from);
    if (pos != std::string::npos) {
      name.replace(pos, from.length(), to);
    }
  };

  // 替换C++类型为Python对应类型
  safe_replace("cv::Mat", "numpy.ndarray");
  safe_replace("std::vector", "list");
  safe_replace("std::map", "dict");
  safe_replace("std::unordered_map", "dict");
  safe_replace("std::set", "set");
  safe_replace("std::unordered_set", "set");
  safe_replace("std::list", "list");
  safe_replace("std::unordered_list", "list");
  safe_replace("std::deque", "list");
  safe_replace("std::string", "str");

  // 预分配结果字符串空间以避免频繁重新分配
  std::string result;
  result.reserve(name.length());

  size_t i = 0;

  if (name.back() == '>' || name.back() == ']') {
    while (i < name.length()) {
      if (name[i] == '<' || name[i] == '[') {
        // 找到容器类型的开始
        char open_bracket = name[i];
        char close_bracket = (open_bracket == '<') ? '>' : ']';

        // 处理容器名称 - 直接在原字符串上操作，避免substr
        size_t container_start = 0;
        std::string_view container_part(name.data() + container_start,
                                        i - container_start);

        // 移除C++命名空间（::） - 使用rfind优化查找
        size_t pos_colon = container_part.rfind("::");
        if (pos_colon != std::string_view::npos) {
          container_part = container_part.substr(pos_colon + 2);
        }

        // 移除Python模块路径（.） - 使用rfind优化查找
        size_t pos_dot = container_part.rfind('.');
        if (pos_dot != std::string_view::npos) {
          container_part = container_part.substr(pos_dot + 1);
        }

        result.append(container_part);
        result.push_back(open_bracket);

        // 找到匹配的闭合括号 - 优化括号匹配逻辑
        int bracket_count = 1;
        size_t start = i + 1;
        i++;

        while (i < name.length() && bracket_count > 0) {
          char current_char = name[i];
          if (current_char == open_bracket) {
            bracket_count++;
          } else if (current_char == close_bracket) {
            bracket_count--;
          }
          i++;
        }

        // 递归处理括号内的内容 - 使用string_view避免不必要的拷贝
        if (start < i - 1) {
          std::string inner_content(name, start, i - start - 1);
          result.append(removeNamespace(inner_content));
        }
        result.push_back(close_bracket);

        // 处理剩余部分
        if (i < name.length()) {
          std::string remaining(name, i);
          result.append(removeNamespace(remaining));
        }

        return result;
      }
      i++;
    }
  }

  // 没有找到模板括号，按原来的逻辑处理
  std::string_view name_view(name);

  // 移除C++命名空间（::） - 使用rfind优化查找
  size_t pos_colon = name_view.rfind("::");
  if (pos_colon != std::string_view::npos) {
    name_view = name_view.substr(pos_colon + 2);
  }

  // 移除Python模块路径（.） - 使用rfind优化查找
  size_t pos_dot = name_view.rfind('.');
  if (pos_dot != std::string_view::npos) {
    name_view = name_view.substr(pos_dot + 1);
  }

  return std::string(name_view);
}

}  // namespace dag
}  // namespace nndeploy
