#include "nndeploy/op/op_concat.h"
#include "nndeploy/op/x86/op_include.h"
#include "nndeploy/op/x86/op_util.h"
#include "nndeploy/op/x86/op_convert.h"



namespace nndeploy {
namespace op {

// Concat 算子的 X86 (oneDNN) 实现
class X86OpConcat : public OpConcat {
 public:
  X86OpConcat() : OpConcat() {}
  virtual ~X86OpConcat() {}

  virtual base::Status init() {
    base::Status status = OpConcat::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    dnnl_engine_ = getDnnlEngine();
    dnnl_stream_ = getDnnlStream();

    return kStatusCodeOk;
  }

  virtual base::Status preRun() {
    NNDEPLOY_LOGE("concat\n");
    if (!is_changed_) {
      return kStatusCodeOk;  //若算子状态未改变，则直接沿用上一次创建memory、算子等
    }

    if (inputs_.size() < 1) {
      NNDEPLOY_LOGE("Concat op needs at least 1 input.");
      return base::kStatusCodeErrorInvalidParam;
    }
    
    auto param = dynamic_cast<ir::ConcatParam*>(op_desc_.op_param_.get());
    int axis = param->axis_;

	  NNDEPLOY_ASSERT(inputs_[0]->getShape().size() == inputs_[1]->getShape().size());
	  NNDEPLOY_ASSERT(inputs_[1]->getShape().size() == outputs_[0]->getShape().size());

	  int now_ndim = inputs_[0]->getShape().size();

    for (int i = 0; i < now_ndim; i++) {
        if (i != axis) {
            assert(inputs_[0]->getShapeIndex(i) == inputs_[1]->getShapeIndex(i));
            assert(inputs_[1]->getShapeIndex(i) == outputs_[0]->getShapeIndex(i));
        } else {
            assert(inputs_[0]->getShapeIndex(i) + inputs_[1]->getShapeIndex(i) == outputs_[0]->getShapeIndex(i));
        }
    }

    std::vector<long int> shape1, format1, shape2, format2, shape3, format3;
    for (int i = 0; i < now_ndim; i++) {
        shape1.push_back(inputs_[0]->getShapeIndex(i));
        shape2.push_back(inputs_[1]->getShapeIndex(i));
        shape3.push_back(outputs_[0]->getShapeIndex(i));
    }
    format1.resize(now_ndim);
    format2.resize(now_ndim);
    format3.resize(now_ndim);

    format1[(now_ndim) - 1] = format2[(now_ndim) - 1] =
        format3[(now_ndim) - 1] = 1;
    for (int i = format1.size() - 2; i >= 0; i--) {
        format1[i] = format1[i + 1] * shape1[i + 1];
        format2[i] = format2[i + 1] * shape2[i + 1];
        format3[i] = format3[i + 1] * shape3[i + 1];
    }

    dnnl::memory::data_type matA_data_type =
        X86OpConvert::convertFromDataType(inputs_[0]->getDataType());
    dnnl::memory::data_type matB_data_type =
        X86OpConvert::convertFromDataType(inputs_[1]->getDataType());
    dnnl::memory::data_type dst_data_type =
        X86OpConvert::convertFromDataType(outputs_[0]->getDataType());

    auto matA_md = dnnl::memory::desc(shape1, matA_data_type, format1);
    auto matB_md = dnnl::memory::desc(shape2, matB_data_type, format2);
    auto concat_src_memA = dnnl::memory(matA_md, dnnl_engine_, inputs_[0]->getData());
    auto concat_src_memB = dnnl::memory(matB_md, dnnl_engine_, inputs_[1]->getData());
    auto concat_dst_md = dnnl::memory::desc(shape3, dst_data_type, format3);
    // TODO
    // auto concat_dst_mem_ = dnnl::memory(concat_dst_md, dnnl_engine_, outputs_[0]->getData());
    concat_dst_mem_ = dnnl::memory(concat_dst_md, dnnl_engine_, outputs_[0]->getData());
    outputs_[0]->getDesc().print();

    std::vector<dnnl::memory::desc> concat_srcs_md;
    
    concat_srcs_md.push_back(matA_md);
    concat_srcs_md.push_back(matB_md);
    concat_srcs_mem_.push_back(concat_src_memA);
    concat_srcs_mem_.push_back(concat_src_memB);
    NNDEPLOY_LOGE("concat axis = %d\n", axis);
    try {
      // TODO
      dnnl_concat_pd_ = dnnl::concat::primitive_desc(dnnl_engine_, concat_dst_md, axis, concat_srcs_md);
      // auto dnnl_concat_pd_ = dnnl::concat::primitive_desc(dnnl_engine_, concat_dst_md, axis, concat_srcs_md);
    } catch (const dnnl::error& e) {
      std::string error_msg = std::string(e.what());
      NNDEPLOY_LOGE("DNNL concat primitive_desc创建失败: %s\n", error_msg.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
    NNDEPLOY_LOGE("concat\n");

	  return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    NNDEPLOY_LOGE("concat\n");
    auto concat_e = dnnl::concat(dnnl_concat_pd_);
    NNDEPLOY_LOGE("concat\n");
    try {
      concat_e.execute(dnnl_stream_, {{DNNL_ARG_DST, concat_dst_mem_},
                                     {DNNL_ARG_MULTIPLE_SRC + 0, concat_srcs_mem_[0]},
                                     {DNNL_ARG_MULTIPLE_SRC + 1, concat_srcs_mem_[1]}});
    } catch (const dnnl::error& e) {
      std::string error_msg = std::string(e.what());
      NNDEPLOY_LOGE("DNNL concat execute失败: %s\n", error_msg.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
    NNDEPLOY_LOGE("concat\n");
    dnnl_stream_.wait();
    NNDEPLOY_LOGE("concat\n");

    return base::kStatusCodeOk;
  }

  virtual base::Status postRun() {
    NNDEPLOY_LOGE("concat\n");
    is_changed_ = false;
    NNDEPLOY_LOGE("concat\n");
    return base::kStatusCodeOk;
  }

 private:
  dnnl::engine dnnl_engine_;
  dnnl::stream dnnl_stream_;
  dnnl::concat::primitive_desc dnnl_concat_pd_;
  std::vector<dnnl::memory> concat_srcs_mem_;
  dnnl::memory concat_dst_mem_;
};

// 注册 OpConcat 算子
REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86, ir::kOpTypeConcat, X86OpConcat)

}  // namespace op
}  // namespace nndeploy