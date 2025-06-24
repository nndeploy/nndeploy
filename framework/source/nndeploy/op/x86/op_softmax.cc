#include "nndeploy/op/op_softmax.h"
#include "nndeploy/op/x86/op_include.h"
#include "nndeploy/op/x86/op_util.h"
#include "nndeploy/op/x86/op_convert.h"



namespace nndeploy {
namespace op {

// Softmax 算子的 X86 (oneDNN) 实现
class X86OpSoftmax : public OpSoftmax {
 public:
  X86OpSoftmax() : OpSoftmax() {}
  virtual ~X86OpSoftmax() {}

  virtual base::Status init() {
    base::Status status = OpSoftmax::init();
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

	  NNDEPLOY_ASSERT(inputs_[0]->getShape().size() == 2);

    auto param = dynamic_cast<ir::SoftmaxParam*>(op_desc_.op_param_.get());
    int axis = param->axis_;

    dnnl::memory::data_type mat_data_type =
        X86OpConvert::convertFromDataType(inputs_[0]->getDataType());
    dnnl::memory::format_tag mat_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[0]->getDataFormat());

    auto mat_md = dnnl::memory::desc({inputs_[0]->getShapeIndex(0), inputs_[0]->getShapeIndex(1)}, mat_data_type, mat_dataformat);
    auto dst_md = dnnl::memory::desc({inputs_[0]->getShapeIndex(0), inputs_[0]->getShapeIndex(1)}, mat_data_type, mat_dataformat);
    softmax_src_mem_ = dnnl::memory(mat_md, dnnl_engine_, inputs_[0]->getData());
    softmax_dst_mem_ = dnnl::memory(mat_md, dnnl_engine_, outputs_[0]->getData());

    dnnl_softmax_pd_ = dnnl::softmax_forward::primitive_desc(dnnl_engine_, 
        dnnl::prop_kind::forward_inference, dnnl::algorithm::softmax_accurate, mat_md, dst_md, axis);

	  return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    auto softmax_e = dnnl::softmax_forward(dnnl_softmax_pd_);
    softmax_e.execute(dnnl_stream_,
                    {{DNNL_ARG_SRC, softmax_src_mem_}, {DNNL_ARG_DST, softmax_dst_mem_}});

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
  dnnl::softmax_forward::primitive_desc dnnl_softmax_pd_;
  dnnl::memory softmax_src_mem_;
  dnnl::memory softmax_dst_mem_;
};

// 注册 OpSoftmax 算子
REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86, ir::kOpTypeSoftmax, X86OpSoftmax)

}  // namespace op
}  // namespace nndeploy