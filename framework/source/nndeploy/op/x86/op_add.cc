#include "nndeploy/op/op_add.h"
#include "nndeploy/op/x86/op_include.h"
#include "nndeploy/op/x86/op_util.h"
#include "nndeploy/op/x86/op_convert.h"



namespace nndeploy {
namespace op {

// Add 算子的 X86 (oneDNN) 实现
class X86OpAdd : public OpAdd {
 public:
  X86OpAdd() : OpAdd() {}
  virtual ~X86OpAdd() {}

  virtual base::Status init() {
    base::Status status = OpAdd::init();
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

    dnnl::memory::dims src1_dims =
        X86OpConvert::convertFromShape(inputs_[0]->getShape());
    dnnl::memory::data_type src1_data_type =
        X86OpConvert::convertFromDataType(inputs_[0]->getDataType());
    dnnl::memory::format_tag src1_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[0]->getDataFormat());

    dnnl::memory::dims src2_dims =
        X86OpConvert::convertFromShape(inputs_[1]->getShape());
    dnnl::memory::data_type src2_data_type =
        X86OpConvert::convertFromDataType(inputs_[1]->getDataType());
    dnnl::memory::format_tag src2_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[1]->getDataFormat());

    auto src1_md = dnnl::memory::desc(src1_dims, src1_data_type, src1_dataformat);
    auto src2_md = dnnl::memory::desc(src2_dims, src2_data_type, src2_dataformat);
    auto dst_md = dnnl::memory::desc(src1_dims, src1_data_type, src1_dataformat);
    add_src1_mem_ = dnnl::memory(src1_md, dnnl_engine_, inputs_[0]->getData());
    add_src2_mem_ = dnnl::memory(src2_md, dnnl_engine_, inputs_[1]->getData());
    add_dst_mem_ = dnnl::memory(src1_md, dnnl_engine_, outputs_[0]->getData());

    dnnl_add_pd_ = dnnl::binary::primitive_desc(dnnl_engine_, dnnl::algorithm::binary_add,
              src1_md, src2_md, dst_md);

	  return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    auto add_e = dnnl::binary(dnnl_add_pd_);
    add_e.execute(dnnl_stream_,
                    {{DNNL_ARG_SRC_0, add_src1_mem_}, {DNNL_ARG_SRC_1, add_src2_mem_}, {DNNL_ARG_DST, add_dst_mem_}});

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
  dnnl::binary::primitive_desc dnnl_add_pd_;
  dnnl::memory add_src1_mem_;
  dnnl::memory add_src2_mem_;
  dnnl::memory add_dst_mem_;
};

// 注册 OpAdd 算子
REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86, ir::kOpTypeAdd, X86OpAdd)

}  // namespace op
}  // namespace nndeploy