#ifndef _NNDEPLOY_OP_X86_OP_UTIL_H_
#define _NNDEPLOY_OP_X86_OP_UTIL_H_

#include "nndeploy/op/x86/op_include.h"

namespace nndeploy {

namespace op {

// 获取dnnl engine和stream
dnnl::engine& getDnnlEngine();
dnnl::stream& getDnnlStream();

// Read from dnnl memory, write to handle
void read_from_dnnl_memory(void* handle, dnnl::memory& mem);

}  // namespace op
}  // namespace nndeploy

#endif