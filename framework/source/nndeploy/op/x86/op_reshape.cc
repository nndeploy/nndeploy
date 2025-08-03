#include "nndeploy/op/op_reshape.h"
#include "nndeploy/op/x86/op_include.h"
#include "nndeploy/op/x86/op_util.h"
#include "nndeploy/op/x86/op_convert.h"



namespace nndeploy {
namespace op {

// Reshape 算子的 X86 (oneDNN) 实现
class X86OpReshape : public OpReshape {
 public:
  X86OpReshape() : OpReshape() {}
  virtual ~X86OpReshape() {}

  virtual base::Status init() {
    base::Status status = OpReshape::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    dnnl_engine_ = getDnnlEngine();
    dnnl_stream_ = getDnnlStream();

    return kStatusCodeOk;
  }

  virtual base::Status preRun() {
    return base::kStatusCodeOk;
  }

  virtual base::Status run() {
	if (outputs_[0]->getData() != inputs_[0]->getData()) {
		size_t size_output = outputs_[0]->getSize();
		size_t size_input = inputs_[0]->getSize();

		// 为防止意外的内存越界，取输入和输出字节大小中的较小值进行拷贝。
		// 在一个有效的 Reshape 操作中，两者的元素总数应当相同，因此它们的总字节大小也应相等。
		size_t size_to_copy = std::min(size_output, size_input);
		memcpy(outputs_[0]->getData(), inputs_[0]->getData(), size_to_copy);
	}

	return base::kStatusCodeOk;
  }

  virtual base::Status postRun() {
    is_changed_ = false;
    return base::kStatusCodeOk;
  }

 private:
  dnnl::engine dnnl_engine_;
  dnnl::stream dnnl_stream_;
};

// 注册 OpReshape 算子
REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86, ir::kOpTypeReshape, X86OpReshape)

}  // namespace op
}  // namespace nndeploy