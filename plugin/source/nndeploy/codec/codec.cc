
#include "nndeploy/codec/codec.h"

namespace nndeploy {
namespace codec {

std::map<base::CodecType, createDecodeNodeFunc>
    &getGlobaCreatelDecodeNodeFuncMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<base::CodecType, createDecodeNodeFunc>>
      creators;
  std::call_once(once, []() {
    creators.reset(new std::map<base::CodecType, createDecodeNodeFunc>);
  });
  return *creators;
}

DecodeNode *createDecodeNode(base::CodecType type, base::CodecFlag flag,
                             const std::string &name, dag::Edge *output) {
  DecodeNode *temp = nullptr;
  auto &map = getGlobaCreatelDecodeNodeFuncMap();
  if (map.count(type) > 0) {
    temp = map[type](flag, name, output);
  }
  return temp;
}

std::map<base::CodecType, createEncodeNodeFunc>
    &getGlobaCreatelEncodeNodeFuncMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<base::CodecType, createEncodeNodeFunc>>
      creators;
  std::call_once(once, []() {
    creators.reset(new std::map<base::CodecType, createEncodeNodeFunc>);
  });
  return *creators;
}

EncodeNode *createEncodeNode(base::CodecType type, base::CodecFlag flag,
                             const std::string &name, dag::Edge *input) {
  EncodeNode *temp = nullptr;
  auto &map = getGlobaCreatelEncodeNodeFuncMap();
  if (map.count(type) > 0) {
    temp = map[type](flag, name, input);
  }
  return temp;
}

}  // namespace codec
}  // namespace nndeploy