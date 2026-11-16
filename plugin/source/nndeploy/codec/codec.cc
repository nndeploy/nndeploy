
#include "nndeploy/codec/codec.h"

namespace nndeploy {
namespace codec {

base::Status Decode::reset(const std::string &path) {
  auto status = deinit();
  if (status != base::kStatusCodeOk) return status;
  path_ = path;
  path_changed_ = false;
  path_ready_ = false;
  index_ = 0;
  size_ = 0;
  width_ = height_ = 0;
  fps_ = 0;
  loop_count_ = 1;
  return base::kStatusCodeOk;
}

base::Status Encode::reset(const std::string &path, const std::string &ref_path) {
  auto status = deinit();
  if (status != base::kStatusCodeOk) return status;
  path_ = path;
  ref_path_ = ref_path;
  path_changed_ = false;
  index_ = 0;
  size_ = 0;
  width_ = height_ = 0;
  fps_ = 0;
  loop_count_ = 1;
  return base::kStatusCodeOk;
}

std::map<base::CodecType, createDecodeFunc> &
getGlobalCreateDecodeFuncMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<base::CodecType, createDecodeFunc>>
      creators;
  std::call_once(once, []() {
    creators.reset(new std::map<base::CodecType, createDecodeFunc>);
  });
  return *creators;
}

std::map<base::CodecType, createDecodeSharedPtrFunc> &
getGlobalCreateDecodeSharedPtrFuncMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<base::CodecType, createDecodeSharedPtrFunc>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<base::CodecType, createDecodeSharedPtrFunc>);
  });
  return *creators;
}

Decode *createDecode(base::CodecType type, base::CodecFlag flag,
                             const std::string &name, dag::Edge *output) {
  Decode *temp = nullptr;
  auto &map = getGlobalCreateDecodeFuncMap();
  if (map.count(type) > 0) {
    temp = map[type](flag, name, output);
  }
  return temp;
}

std::shared_ptr<Decode> createDecodeSharedPtr(base::CodecType type,
                                                      base::CodecFlag flag,
                                                      const std::string &name,
                                                      dag::Edge *output) {
  auto &map = getGlobalCreateDecodeSharedPtrFuncMap();
  if (map.count(type) > 0) {
    return map[type](flag, name, output);
  }
  return nullptr;
}

std::map<base::CodecType, createEncodeFunc> &
getGlobalCreateEncodeFuncMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<base::CodecType, createEncodeFunc>>
      creators;
  std::call_once(once, []() {
    creators.reset(new std::map<base::CodecType, createEncodeFunc>);
  });
  return *creators;
}

std::map<base::CodecType, createEncodeSharedPtrFunc> &
getGlobalCreateEncodeSharedPtrFuncMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<base::CodecType, createEncodeSharedPtrFunc>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<base::CodecType, createEncodeSharedPtrFunc>);
  });
  return *creators;
}

Encode *createEncode(base::CodecType type, base::CodecFlag flag,
                             const std::string &name, dag::Edge *input) {
  Encode *temp = nullptr;
  auto &map = getGlobalCreateEncodeFuncMap();
  if (map.count(type) > 0) {
    temp = map[type](flag, name, input);
  }
  return temp;
}

std::shared_ptr<Encode> createEncodeSharedPtr(base::CodecType type,
                                                      base::CodecFlag flag,
                                                      const std::string &name,
                                                      dag::Edge *input) {
  auto &map = getGlobalCreateEncodeSharedPtrFuncMap();
  if (map.count(type) > 0) {
    return map[type](flag, name, input);
  }
  return nullptr;
}

}  // namespace codec
}  // namespace nndeploy
