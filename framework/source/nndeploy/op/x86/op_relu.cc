#include "nndeploy/op/op_relu.h"
#include "nndeploy/op/x86/op_include.h"
#include "nndeploy/op/x86/op_util.h"
#include "nndeploy/op/x86/op_convert.h"



namespace nndeploy {
namespace op {

// Relu 算子的 X86 (oneDNN) 实现
class X86OpRelu : public OpRelu {
 public:
  X86OpRelu() : OpRelu() {}
  virtual ~X86OpRelu() {}

  virtual base::Status init() {
    base::Status status = OpRelu::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    dnnl_engine_ = getDnnlEngine();
    dnnl_stream_ = getDnnlStream();

    return kStatusCodeOk;
  }

  virtual base::Status preRun() {
    if (!is_changed_) {
      return kStatusCodeOk;  //若算子状态未改变，则直接沿用上一次创建memory、算子等
    }

    dnnl::memory::dims src_dims =
        X86OpConvert::convertFromShape(inputs_[0]->getShape());
    dnnl::memory::data_type src_data_type =
        X86OpConvert::convertFromDataType(inputs_[0]->getDataType());
    dnnl::memory::format_tag src_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[0]->getDataFormat());

    auto src_md = dnnl::memory::desc(src_dims, src_data_type, src_dataformat);
    auto dst_md = dnnl::memory::desc(src_dims, src_data_type, src_dataformat);
    relu_src_mem_ = dnnl::memory(src_md, dnnl_engine_, inputs_[0]->getData());
    relu_dst_mem_ = dnnl::memory(src_md, dnnl_engine_, outputs_[0]->getData());

    dnnl_relu_pd_ = dnnl::eltwise_forward::primitive_desc(dnnl_engine_,
            dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_relu, src_md, dst_md, 0.f, 0.f);

	  return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    auto relu_e = dnnl::eltwise_forward(dnnl_relu_pd_);
    relu_e.execute(dnnl_stream_,
                    {{DNNL_ARG_SRC, relu_src_mem_}, {DNNL_ARG_DST, relu_dst_mem_}});

    dnnl_stream_.wait();

    return base::kStatusCodeOk;
  }

  virtual base::Status postRun() {
    is_changed_ = false;
    return base::kStatusCodeOk;
  }

 private:
  dnnl::engine dnnl_engine_;
  dnnl::stream dnnl_stream_;
  dnnl::eltwise_forward::primitive_desc dnnl_relu_pd_;
  dnnl::memory relu_src_mem_;
  dnnl::memory relu_dst_mem_;
};

// 注册 OpRelu 算子
REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86, ir::kOpTypeRelu, X86OpRelu)

}  // namespace op
}  // namespace nndeploy