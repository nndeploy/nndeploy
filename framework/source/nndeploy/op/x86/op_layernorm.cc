#include "nndeploy/op/op_layernorm.h"
#include "nndeploy/op/x86/op_convert.h"
#include "nndeploy/op/x86/op_include.h"
#include "nndeploy/op/x86/op_util.h"

namespace nndeploy {
namespace op {

class X86OpLayerNorm : public OpLayerNorm {
 public:
  X86OpLayerNorm() : OpLayerNorm() {}
  virtual ~X86OpLayerNorm() {}

  virtual base::Status init() {
    base::Status status = OpLayerNorm::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    dnnl_engine_ = getDnnlEngine();
    dnnl_stream_ = getDnnlStream();
    return base::kStatusCodeOk;
  }

  virtual base::Status preRun() {
    if (!is_changed_) {
      return base::kStatusCodeOk;
    }

    auto param =
        dynamic_cast<ir::LayerNormalizationParam *>(op_desc_.op_param_.get());
    NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param,
                                         "op_desc_.op_param_ is nullptr");

    const auto &input_shape = inputs_[0]->getShape();
    const auto &weight_shape = inputs_[1]->getShape();
    if (input_shape.empty() || weight_shape.size() != 1 ||
        input_shape.back() != weight_shape[0]) {
      NNDEPLOY_LOGE("x86 LayerNorm only supports last-dimension normalization "
                    "with 1D weight.\n");
      return base::kStatusCodeErrorInvalidParam;
    }

    const bool has_bias = inputs_.size() == 3;
    if (has_bias && inputs_[2]->getShape() != weight_shape) {
      NNDEPLOY_LOGE("LayerNorm bias shape must equal weight shape.\n");
      return base::kStatusCodeErrorInvalidParam;
    }

    dnnl::memory::dims src_dims = X86OpConvert::convertFromShape(input_shape);
    dnnl::memory::dims weight_dims =
        X86OpConvert::convertFromShape(weight_shape);
    dnnl::memory::dims dst_dims =
        X86OpConvert::convertFromShape(outputs_[0]->getShape());

    dnnl::memory::data_type src_data_type =
        X86OpConvert::convertFromDataType(inputs_[0]->getDataType());
    dnnl::memory::data_type weight_data_type =
        X86OpConvert::convertFromDataType(inputs_[1]->getDataType());
    dnnl::memory::data_type dst_data_type =
        X86OpConvert::convertFromDataType(outputs_[0]->getDataType());

    dnnl::memory::format_tag src_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[0]->getDataFormat());
    dnnl::memory::format_tag weight_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[1]->getDataFormat());
    dnnl::memory::format_tag dst_dataformat =
        X86OpConvert::convertFromDataFormat(outputs_[0]->getDataFormat());

    auto src_md = dnnl::memory::desc(src_dims, src_data_type, src_dataformat);
    auto weight_md =
        dnnl::memory::desc(weight_dims, weight_data_type, weight_dataformat);
    auto dst_md = dnnl::memory::desc(dst_dims, dst_data_type, dst_dataformat);

    layernorm_src_mem_ = dnnl::memory(src_md, dnnl_engine_, inputs_[0]->getData());
    layernorm_scale_mem_ =
        dnnl::memory(weight_md, dnnl_engine_, inputs_[1]->getData());
    if (has_bias) {
      dnnl::memory::data_type bias_data_type =
          X86OpConvert::convertFromDataType(inputs_[2]->getDataType());
      dnnl::memory::format_tag bias_dataformat =
          X86OpConvert::convertFromDataFormat(inputs_[2]->getDataFormat());
      auto bias_md =
          dnnl::memory::desc(weight_dims, bias_data_type, bias_dataformat);
      layernorm_shift_mem_ =
          dnnl::memory(bias_md, dnnl_engine_, inputs_[2]->getData());
    }
    layernorm_dst_mem_ =
        dnnl::memory(dst_md, dnnl_engine_, outputs_[0]->getData());

    auto flags = dnnl::normalization_flags::use_scale;
    if (has_bias) {
      flags = flags | dnnl::normalization_flags::use_shift;
    }
    dnnl_layernorm_pd_ = dnnl::layer_normalization_forward::primitive_desc(
        dnnl_engine_, dnnl::prop_kind::forward_inference, src_md, dst_md,
        param->epsilon_, flags);

    return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    std::unordered_map<int, dnnl::memory> layernorm_args;
    layernorm_args.insert({DNNL_ARG_SRC, layernorm_src_mem_});
    layernorm_args.insert({DNNL_ARG_SCALE, layernorm_scale_mem_});
    if (inputs_.size() == 3) {
      layernorm_args.insert({DNNL_ARG_SHIFT, layernorm_shift_mem_});
    }
    layernorm_args.insert({DNNL_ARG_DST, layernorm_dst_mem_});
    auto layernorm = dnnl::layer_normalization_forward(dnnl_layernorm_pd_);
    layernorm.execute(dnnl_stream_, layernorm_args);

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
  dnnl::layer_normalization_forward::primitive_desc dnnl_layernorm_pd_;
  dnnl::memory layernorm_src_mem_;
  dnnl::memory layernorm_scale_mem_;
  dnnl::memory layernorm_shift_mem_;
  dnnl::memory layernorm_dst_mem_;
};

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86, ir::kOpTypeLayerNormalization,
                         X86OpLayerNorm)

}  // namespace op
}  // namespace nndeploy
