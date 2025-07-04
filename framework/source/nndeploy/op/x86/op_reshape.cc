#include "nndeploy/op/op_reshape.h"
#include "nndeploy/op/x86/op_include.h"
#include "nndeploy/op/x86/op_util.h"
#include "nndeploy/op/x86/op_convert.h"
#include "nndeploy/op/op_concat.h"



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
    if (!is_changed_) {
      return kStatusCodeOk;  //若算子状态未改变，则直接沿用上一次创建memory、算子等
    }

	  NNDEPLOY_ASSERT(inputs_[0]->getShape().size() == 4);

    auto param = dynamic_cast<ir::ReshapeParam*>(op_desc_.op_param_.get());

    int N = inputs_[0]->getShapeIndex(0);
    int C = inputs_[0]->getShapeIndex(1);
    int H = inputs_[0]->getShapeIndex(2);
    int W = inputs_[0]->getShapeIndex(3);

    dnnl::memory::data_type mat_data_type =
        X86OpConvert::convertFromDataType(inputs_[0]->getDataType());
    dnnl::memory::format_tag mat_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[0]->getDataFormat());

    dnnl::memory::data_type ptr_data_type =
        X86OpConvert::convertFromDataType(inputs_[1]->getDataType());
    dnnl::memory::format_tag ptr_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[1]->getDataFormat());


    // TODO:支持融合relu
    auto data_md = dnnl::memory::desc({N, C, H, W}, mat_data_type,
                                mat_dataformat);
    auto ptr_md =
        dnnl::memory::desc({C}, ptr_data_type, ptr_dataformat);

    auto dst_md = dnnl::memory::desc({N, C, H, W}, mat_data_type, mat_dataformat);
    batchnorm_src_mem_ = dnnl::memory(data_md, dnnl_engine_, inputs_[0]->getData());
    batchnorm_scale_mem_ = dnnl::memory(ptr_md, dnnl_engine_, inputs_[1]->getData());
    batchnorm_bias_mem_ = dnnl::memory(ptr_md, dnnl_engine_, inputs_[2]->getData());
    batchnorm_mean_mem_ = dnnl::memory(ptr_md, dnnl_engine_, inputs_[3]->getData());
    batchnorm_var_mem_ = dnnl::memory(ptr_md, dnnl_engine_, inputs_[4]->getData());
    batchnorm_dst_mem_ = dnnl::memory(data_md, dnnl_engine_, outputs_[0]->getData());

    dnnl_batchnorm_pd_ = dnnl::batch_normalization_forward::primitive_desc(dnnl_engine_, 
        dnnl::prop_kind::forward_inference, data_md, dst_md, epsilon, 
        dnnl::normalization_flags::use_global_stats |
        dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift);

	  return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    auto batchnorm_e = dnnl::batch_normalization_forward(dnnl_batchnorm_pd_);
    std::unordered_map<int, dnnl::memory> bnorm_args;
    bnorm_args.insert({DNNL_ARG_SRC, batchnorm_src_mem_});
    bnorm_args.insert({DNNL_ARG_MEAN, batchnorm_mean_mem_});
    bnorm_args.insert({DNNL_ARG_VARIANCE, batchnorm_var_mem_});
    bnorm_args.insert({DNNL_ARG_SCALE, batchnorm_scale_mem_});
    bnorm_args.insert({DNNL_ARG_SHIFT, batchnorm_bias_mem_});
    bnorm_args.insert({DNNL_ARG_DST, batchnorm_dst_mem_});
    batchnorm_e.execute(dnnl_stream_,
                        bnorm_args);

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
  dnnl::batch_normalization_forward::primitive_desc dnnl_batchnorm_pd_;
  dnnl::memory batchnorm_src_mem_;
  dnnl::memory batchnorm_scale_mem_;
  dnnl::memory batchnorm_bias_mem_;
  dnnl::memory batchnorm_mean_mem_;
  dnnl::memory batchnorm_var_mem_;
  dnnl::memory batchnorm_dst_mem_;
};

// 注册 OpReshape 算子
REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86, ir::kOpTypeReshapealization, X86OpReshape)

}  // namespace op
}  // namespace nndeploy