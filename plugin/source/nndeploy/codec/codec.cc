
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

std::map<base::CodecType, createDecodeNodeSharedPtrFunc>
    &getGlobaCreatelDecodeNodeSharedPtrFuncMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<base::CodecType, createDecodeNodeSharedPtrFunc>>
      creators;
  std::call_once(once, []() {
    creators.reset(new std::map<base::CodecType, createDecodeNodeSharedPtrFunc>);
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

std::shared_ptr<DecodeNode> createDecodeNodeSharedPtr(base::CodecType type,
                                                    base::CodecFlag flag,
                                                    const std::string &name,
                                                    dag::Edge *output) {
  auto &map = getGlobaCreatelDecodeNodeSharedPtrFuncMap();
  if (map.count(type) > 0) {
    return map[type](flag, name, output);
  }
  return nullptr;
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

std::map<base::CodecType, createEncodeNodeSharedPtrFunc>
    &getGlobaCreatelEncodeNodeSharedPtrFuncMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<base::CodecType, createEncodeNodeSharedPtrFunc>>
      creators;
  std::call_once(once, []() {
    creators.reset(new std::map<base::CodecType, createEncodeNodeSharedPtrFunc>);
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

std::shared_ptr<EncodeNode> createEncodeNodeSharedPtr(base::CodecType type,
                                                    base::CodecFlag flag,
                                                    const std::string &name,
                                                    dag::Edge *input) {
  auto &map = getGlobaCreatelEncodeNodeSharedPtrFuncMap();
  if (map.count(type) > 0) {
    return map[type](flag, name, input);
  }
  return nullptr;
}

}  // namespace codec
}  // namespace nndeploy