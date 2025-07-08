#include "nndeploy/op/op_transpose.h"
#include "nndeploy/op/x86/op_include.h"
#include "nndeploy/op/x86/op_util.h"
#include "nndeploy/op/x86/op_convert.h"



namespace nndeploy {
namespace op {

// Transpose 算子的 X86 (oneDNN) 实现
class X86OpTranspose : public OpTranspose {
 public:
  X86OpTranspose() : OpTranspose() {}
  virtual ~X86OpTranspose() {}
  
  virtual base::Status init() {
    base::Status status = OpTranspose::init();
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
    NNDEPLOY_LOGE("3");
    auto param = dynamic_cast<ir::TransposeParam*>(op_desc_.op_param_.get());
    auto axis = param->perm_;
    
    dnnl::memory::dims src_dims =
        X86OpConvert::convertFromShape(inputs_[0]->getShape());
    dnnl::memory::data_type src_data_type =
        X86OpConvert::convertFromDataType(inputs_[0]->getDataType());
    dnnl::memory::format_tag src_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[0]->getDataFormat());

    auto src_md = dnnl::memory::desc(src_dims, src_data_type, dnnl::memory::format_tag::abcd);
    dnnl::memory::desc dst_md;
    if (axis == std::vector<int>{0, 2, 1, 3}) dst_md = dnnl::memory::desc(src_dims, src_data_type, dnnl::memory::format_tag::acbd);
    else if (axis == std::vector<int>{0, 1, 3, 2}) dst_md = dnnl::memory::desc(src_dims, src_data_type, dnnl::memory::format_tag::abdc);
    else NNDEPLOY_LOGE("x86 transpose do not support this transformation of dataformats!\n");
    

    print_memory_desc(src_md);
    print_memory_desc(dst_md);

    transpose_src_mem_ = dnnl::memory(src_md, dnnl_engine_, inputs_[0]->getData());
    transpose_dst_mem_ = dnnl::memory(dst_md_, dnnl_engine_, outputs_[0]->getData());

    dnnl_order_pd_ = dnnl::reorder::primitive_desc(
            dnnl_engine_, src_md, dnnl_engine_, dst_md);

	  return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    auto reorder = dnnl::reorder(dnnl_order_pd_);
    std::unordered_map<int, dnnl::memory> reorder_args;
    reorder_args.insert({DNNL_ARG_SRC,transpose_src_mem_});
    reorder_args.insert({DNNL_ARG_DST, transpose_dst_mem_});

    reorder.execute(dnnl_stream_, reorder_args);

    return base::kStatusCodeOk;
  }

  virtual base::Status postRun() {
    is_changed_ = false;
    return base::kStatusCodeOk;
  }

 private:
  dnnl::engine dnnl_engine_;
  dnnl::stream dnnl_stream_;
  dnnl::reorder::primitive_desc dnnl_order_pd_;
  dnnl::memory::desc dst_md_;
  dnnl::memory transpose_src_mem_;
  dnnl::memory transpose_dst_mem_;

};

// 注册 OpTranspose 算子
REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86, ir::kOpTypeTranspose, X86OpTranspose)

}  // namespace op
}  // namespace nndeploy