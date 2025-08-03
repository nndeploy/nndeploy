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
    try {
      // 强制使用 CPU 引擎，避免 SYCL
      engine = std::make_shared<dnnl::engine>(dnnl::engine::kind::cpu, 0);
    } catch (const std::exception& e) {
      // 如果失败，尝试其他方式
      NNDEPLOY_LOGE("Failed to create DNNL CPU engine: %s", e.what());
      // 可以尝试其他引擎类型或回退方案
    }
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

std::string get_format_tag_str(const dnnl::memory::desc &md) {
    // 检查未定义格式
    if (md.is_zero()) return "undef";

    // 遍历常见的 format tag 并进行比较
#define CHECK_TAG(tag) \
    do { \
        auto temp_md = dnnl::memory::desc(md.get_dims(), md.get_data_type(), dnnl::memory::format_tag::tag, true); \
        if (temp_md && temp_md == md) return #tag; \
    } while (0)

    // 添加常用的或期望检查的格式标签
    CHECK_TAG(a);
    CHECK_TAG(ab);
    CHECK_TAG(abc);
    CHECK_TAG(abcd);
    CHECK_TAG(abdc);
    CHECK_TAG(acbd);
    CHECK_TAG(acdb);
    CHECK_TAG(adbc);
    CHECK_TAG(bacd);
    CHECK_TAG(bcda);
    CHECK_TAG(cdba);
    CHECK_TAG(dcab);
    CHECK_TAG(abcde);
    CHECK_TAG(abcdef);

    CHECK_TAG(nchw);
    CHECK_TAG(nhwc);
    CHECK_TAG(chwn);
    
    CHECK_TAG(ncdhw);
    CHECK_TAG(ndhwc);

    CHECK_TAG(oihw);
    CHECK_TAG(hwio);
    CHECK_TAG(goihw);

    CHECK_TAG(x);
    CHECK_TAG(nc);
    CHECK_TAG(cn);
    CHECK_TAG(nwc);
    
    // 如果没有匹配的已知格式，则返回 "unknown"
    return "unknown";
}

void print_memory_desc(const dnnl::memory::desc &md) {    
    auto dims = md.get_dims();    
    auto data_type = md.get_data_type();    
    auto strides = md.get_strides();  
    auto format_tag = md.get_format_kind();  
    std::cout << "ndims: " << dims.size() << std::endl;    
    std::cout << "dims: ";    
    for (size_t i = 0; i < dims.size(); i++) {    
        std::cout << dims[i] << " ";    
    }    
    std::cout << std::endl;    
        
    std::cout << "data_type: " << static_cast<int>(data_type) << std::endl;  
      
    std::cout << "format_kind: " << static_cast<int>(format_tag) << std::endl;  
      
    std::cout << "strides: ";    
    for (size_t i = 0; i < strides.size(); i++) {    
        std::cout << strides[i] << " ";    
    }    
    std::cout << std::endl;    
    std::cout << "format_tag: " << get_format_tag_str(md) << std::endl;
    std::cout << std::endl;    

}

// Helper to handle negative indices and clamp values, similar to torch/numpy
long long normalize_index(long long index, long long dim_size) {
    if (index < 0) {
        index += dim_size;
    }
    // Clamp to [0, dim_size] for starts and [0, dim_size] for ends
    return std::max(0LL, std::min(index, dim_size));
}

}  // namespace op
}  // namespace nndeploy