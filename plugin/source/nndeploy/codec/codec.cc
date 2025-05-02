
#include "nndeploy/codec/codec.h"

namespace nndeploy {
namespace codec {

std::map<base::CodecType, createDecodeNodeFunc> &
getGlobalCreateDecodeNodeFuncMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<base::CodecType, createDecodeNodeFunc>>
      creators;
  std::call_once(once, []() {
    creators.reset(new std::map<base::CodecType, createDecodeNodeFunc>);
  });
  return *creators;
}

std::map<base::CodecType, createDecodeNodeSharedPtrFunc> &
getGlobalCreateDecodeNodeSharedPtrFuncMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<base::CodecType, createDecodeNodeSharedPtrFunc>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<base::CodecType, createDecodeNodeSharedPtrFunc>);
  });
  return *creators;
}

DecodeNode *createDecodeNode(base::CodecType type, base::CodecFlag flag,
                             const std::string &name, dag::Edge *output) {
  DecodeNode *temp = nullptr;
  auto &map = getGlobalCreateDecodeNodeFuncMap();
  if (map.count(type) > 0) {
    temp = map[type](flag, name, output);
  }
  return temp;
}

std::shared_ptr<DecodeNode> createDecodeNodeSharedPtr(base::CodecType type,
                                                      base::CodecFlag flag,
                                                      const std::string &name,
                                                      dag::Edge *output) {
  auto &map = getGlobalCreateDecodeNodeSharedPtrFuncMap();
  if (map.count(type) > 0) {
    return map[type](flag, name, output);
  }
  return nullptr;
}

std::map<base::CodecType, createEncodeNodeFunc> &
getGlobalCreateEncodeNodeFuncMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<base::CodecType, createEncodeNodeFunc>>
      creators;
  std::call_once(once, []() {
    creators.reset(new std::map<base::CodecType, createEncodeNodeFunc>);
  });
  return *creators;
}

std::map<base::CodecType, createEncodeNodeSharedPtrFunc> &
getGlobalCreateEncodeNodeSharedPtrFuncMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<base::CodecType, createEncodeNodeSharedPtrFunc>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<base::CodecType, createEncodeNodeSharedPtrFunc>);
  });
  return *creators;
}

EncodeNode *createEncodeNode(base::CodecType type, base::CodecFlag flag,
                             const std::string &name, dag::Edge *input) {
  EncodeNode *temp = nullptr;
  auto &map = getGlobalCreateEncodeNodeFuncMap();
  if (map.count(type) > 0) {
    temp = map[type](flag, name, input);
  }
  return temp;
}

std::shared_ptr<EncodeNode> createEncodeNodeSharedPtr(base::CodecType type,
                                                      base::CodecFlag flag,
                                                      const std::string &name,
                                                      dag::Edge *input) {
  auto &map = getGlobalCreateEncodeNodeSharedPtrFuncMap();
  if (map.count(type) > 0) {
    return map[type](flag, name, input);
  }
  return nullptr;
}

}  // namespace codec
}  // namespace nndeploy