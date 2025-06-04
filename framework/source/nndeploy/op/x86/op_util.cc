#include "nndeploy/op/x86/op_util.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/op/x86/op_include.h"

namespace nndeploy {

namespace op {

dnnl::engine &getDnnlEngine() {
  static std::once_flag once;
  static std::shared_ptr<dnnl::engine> engine;
  std::call_once(once, []() {
    engine = std::make_shared<dnnl::engine>(dnnl::engine::kind::cpu, 0);
  });
  return *engine;
}

dnnl::stream &getDnnlStream() {
  static std::once_flag once;
  static std::shared_ptr<dnnl::stream> stream;
  std::call_once(
      once, []() { stream = std::make_shared<dnnl::stream>(getDnnlEngine()); });
  return *stream;
}

void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();

  if (!handle) throw std::runtime_error("handle is nullptr.");

  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
    if (!src) throw std::runtime_error("get_data_handle returned nullptr.");
    for (size_t i = 0; i < size; ++i) ((uint8_t *)handle)[i] = src[i];
    return;
  }

  assert(!"not expected");
}

}  // namespace op
}  // namespace nndeploy