#include "nndeploy/op/op_rmsnorm.h"
#include "nndeploy/op/x86/op_convert.h"
#include "nndeploy/op/x86/op_include.h"
#include "nndeploy/op/x86/op_util.h"

namespace nndeploy {
namespace op {
class X86OpRMSNorm : public OpRMSNorm {
 public:
  X86OpRMSNorm() : OpRMSNorm() {}
  virtual ~X86OpRMSNorm() {}

  virtual base::Status init() {
    base::Status status = OpRMSNorm::init();
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

    auto param = dynamic_cast<ir::RMSNormParam*>(op_desc_.op_param_.get());
    auto epsilon = param->eps_;

    dnnl::memory::dims src_dims =
        X86OpConvert::convertFromShape(inputs_[0]->getShape());
    dnnl::memory::dims weights_dims =
        X86OpConvert::convertFromShape(inputs_[1]->getShape());
    dnnl::memory::dims dst_dims =
        X86OpConvert::convertFromShape(outputs_[0]->getShape());

    dnnl::memory::data_type src_data_type =
        X86OpConvert::convertFromDataType(inputs_[0]->getDataType());
    dnnl::memory::data_type weights_data_type =
        X86OpConvert::convertFromDataType(inputs_[1]->getDataType());
    dnnl::memory::data_type dst_data_type =
        X86OpConvert::convertFromDataType(outputs_[0]->getDataType());

    dnnl::memory::format_tag src_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[0]->getDataFormat());
    dnnl::memory::format_tag weights_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[1]->getDataFormat());
    dnnl::memory::format_tag dst_dataformat =
        X86OpConvert::convertFromDataFormat(outputs_[0]->getDataFormat());

    auto src_md = dnnl::memory::desc(src_dims, src_data_type, src_dataformat);
    auto weights_md =
        dnnl::memory::desc(weights_dims, weights_data_type, weights_dataformat);
    auto dst_md = dnnl::memory::desc(dst_dims, dst_data_type, dst_dataformat);

    rmsnorm_src_mem_ = dnnl::memory(src_md, dnnl_engine_, inputs_[0]->getData());
    rmsnorm_weights_mem_ =
        dnnl::memory(weights_md, dnnl_engine_, inputs_[1]->getData());
    // TODO:in-place
    rmsnorm_dst_mem_ = dnnl::memory(dst_md, dnnl_engine_, outputs_[0]->getData());

    dnnl_rmsnorm_pd_ = dnnl::layer_normalization_forward::primitive_desc(dnnl_engine_,
            dnnl::prop_kind::forward_inference, src_md, dst_md, dnnl::memory::data_type::f32,
            epsilon,dnnl::normalization_flags::rms_norm);
            // dnnl::normalization_flags::use_scale | 
            
    return base::kStatusCodeOk;
    // rmsnorm_mean_mem_ = dnnl::memory(dnnl_rmsnorm_pd_.mean_desc(), dnnl_engine_);
    // rmsnorm_variance_mem_ = dnnl::memory(dnnl_rmsnorm_pd_.variance_desc(), dnnl_engine_);
  }
  virtual base::Status run() {
    std::unordered_map<int, dnnl::memory> rmsnorm_args;
    rmsnorm_args.insert({DNNL_ARG_SRC, rmsnorm_src_mem_});
    // rmsnorm_args.insert({DNNL_ARG_MEAN, rmsnorm_mean_mem_});
    // rmsnorm_args.insert({DNNL_ARG_VARIANCE, rmsnorm_variance_mem_});
    rmsnorm_args.insert({DNNL_ARG_SCALE, rmsnorm_weights_mem_});
    rmsnorm_args.insert({DNNL_ARG_DST, rmsnorm_dst_mem_});
    auto dnnl_rmsnorm = dnnl::layer_normalization_forward(dnnl_rmsnorm_pd_);
    dnnl_rmsnorm.execute(dnnl_stream_, rmsnorm_args);

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
  dnnl::layer_normalization_forward::primitive_desc dnnl_rmsnorm_pd_;
  dnnl::memory rmsnorm_src_mem_;
  dnnl::memory rmsnorm_weights_mem_;
  // dnnl::memory rmsnorm_variance_mem_;
  dnnl::memory rmsnorm_dst_mem_;
};

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86, ir::kOpTypeRMSNorm, X86OpRMSNorm)

}  // namespace op
}  // namespace nndeploy