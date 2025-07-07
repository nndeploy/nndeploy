#include "nndeploy/op/op_where.h"
#include "nndeploy/op/x86/op_include.h"
#include "nndeploy/op/x86/op_util.h"
#include "nndeploy/op/x86/op_convert.h"



namespace nndeploy {
namespace op {

// Where 算子的 X86 (oneDNN) 实现
class X86OpWhere : public OpWhere {
 public:
  X86OpWhere() : OpWhere() {}
  virtual ~X86OpWhere() {}

  virtual base::Status init() {
    base::Status status = OpWhere::init();
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

    dnnl::memory::dims src0_dims =
        X86OpConvert::convertFromShape(inputs_[0]->getShape());
    dnnl::memory::data_type src0_data_type =
        X86OpConvert::convertFromDataType(inputs_[0]->getDataType());
    dnnl::memory::format_tag src0_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[0]->getDataFormat());

    dnnl::memory::dims src1_dims =
        X86OpConvert::convertFromShape(inputs_[1]->getShape());
    dnnl::memory::data_type src1_data_type =
        X86OpConvert::convertFromDataType(inputs_[1]->getDataType());
    dnnl::memory::format_tag src1_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[1]->getDataFormat());

    dnnl::memory::dims condition_dims =
        X86OpConvert::convertFromShape(inputs_[2]->getShape());
    dnnl::memory::data_type condition_data_type =
        X86OpConvert::convertFromDataType(inputs_[2]->getDataType());
    dnnl::memory::format_tag condition_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[2]->getDataFormat());      

    auto src0_md = dnnl::memory::desc(src0_dims, src0_data_type, src0_dataformat);
    auto src1_md = dnnl::memory::desc(src1_dims, src1_data_type, src1_dataformat);
    auto condition_md = dnnl::memory::desc(condition_dims, condition_data_type, condition_dataformat);
    auto dst_md = dnnl::memory::desc(src0_dims, src0_data_type, src0_dataformat);
    select_src0_mem_ = dnnl::memory(src0_md, dnnl_engine_, inputs_[0]->getData());
    select_src1_mem_ = dnnl::memory(src1_md, dnnl_engine_, inputs_[1]->getData());
    select_condition_mem_ = dnnl::memory(condition_md, dnnl_engine_, inputs_[2]->getData());
    select_dst_mem_ = dnnl::memory(src0_md, dnnl_engine_, outputs_[0]->getData());

    dnnl_select_pd_ = dnnl::binary::primitive_desc(dnnl_engine_, dnnl::algorithm::binary_select,
              src0_md, src1_md, condition_md, dst_md);

	  return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    auto select_e = dnnl::binary(dnnl_select_pd_);
    std::unordered_map<int, dnnl::memory> binary_args;
    binary_args.insert({DNNL_ARG_SRC_0,select_src0_mem_});
    binary_args.insert({DNNL_ARG_SRC_1, select_src1_mem_});
    binary_args.insert({DNNL_ARG_SRC_2, select_condition_mem_}); 
    binary_args.insert({DNNL_ARG_DST, select_dst_mem_});

    select_e.execute(dnnl_stream_, binary_args);
    
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
  dnnl::binary::primitive_desc dnnl_select_pd_;
  dnnl::memory select_src0_mem_;
  dnnl::memory select_src1_mem_;
  dnnl::memory select_condition_mem_;
  dnnl::memory select_dst_mem_;
};

// 注册 OpWhere 算子
REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86, ir::kOpTypeWhere, X86OpWhere)

}  // namespace op
}  // namespace nndeploy