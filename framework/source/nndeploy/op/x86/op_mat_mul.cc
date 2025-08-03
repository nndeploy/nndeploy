#include "nndeploy/op/op_mat_mul.h"
#include "nndeploy/op/x86/op_include.h"
#include "nndeploy/op/x86/op_util.h"
#include "nndeploy/op/x86/op_convert.h"



namespace nndeploy {
namespace op {

// MatMul 算子的 X86 (oneDNN) 实现
class X86OpMatMul : public OpMatMul {
 public:
  X86OpMatMul() : OpMatMul() {}
  virtual ~X86OpMatMul() {}

  virtual base::Status init() {
    base::Status status = OpMatMul::init();
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

	  NNDEPLOY_ASSERT(inputs_[0]->getShape().size() && inputs_[1]->getShape().size() && outputs_[0]->getShape().size());
    
    // TODO:支持多维矩阵转置
    // auto param = dynamic_cast<ir::MatMulParam*>(op_desc_.op_param_.get());
    // bool transposeA_ = param->transposeA_;
    // bool transposeB_ = param->transposeB_;

    have_bias_ = false;
    if (inputs_.size() > 2) {
      have_bias_ = true;
    }

    dnnl::memory::dims matA_dims =
        X86OpConvert::convertFromShape(inputs_[0]->getShape());
    dnnl::memory::dims matB_dims =
        X86OpConvert::convertFromShape(inputs_[1]->getShape());
    dnnl::memory::dims dst_dims =
        X86OpConvert::convertFromShape(outputs_[0]->getShape());


    dnnl::memory::data_type matA_data_type =
        X86OpConvert::convertFromDataType(inputs_[0]->getDataType());
    dnnl::memory::data_type matB_data_type =
        X86OpConvert::convertFromDataType(inputs_[1]->getDataType());
    dnnl::memory::data_type dst_data_type =
        X86OpConvert::convertFromDataType(outputs_[0]->getDataType());        


    dnnl::memory::format_tag matA_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[0]->getDataFormat());
    dnnl::memory::format_tag matB_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[1]->getDataFormat());
    dnnl::memory::format_tag dst_dataformat =
        X86OpConvert::convertFromDataFormat(outputs_[0]->getDataFormat());

    auto matA_md = dnnl::memory::desc(matA_dims, matA_data_type, matA_dataformat);
    auto matB_md = dnnl::memory::desc(matB_dims, matB_data_type, matB_dataformat);
    auto dst_md = dnnl::memory::desc(dst_dims, dst_data_type, dst_dataformat);

    matmul_matA_mem_ = dnnl::memory(matA_md, dnnl_engine_, inputs_[0]->getData());
    matmul_matB_mem_ = dnnl::memory(matB_md, dnnl_engine_, inputs_[1]->getData());
    matmul_dst_mem_ = dnnl::memory(dst_md, dnnl_engine_, outputs_[0]->getData());

    if (have_bias_){
      dnnl::memory::dims bias_dims =
          X86OpConvert::convertFromShape(inputs_[2]->getShape());
      dnnl::memory::data_type bias_data_type =
          X86OpConvert::convertFromDataType(inputs_[2]->getDataType());
      dnnl::memory::format_tag bias_dataformat =
          X86OpConvert::convertFromDataFormat(inputs_[2]->getDataFormat());
      auto bias_md = dnnl::memory::desc(bias_dims, bias_data_type, bias_dataformat);
      matmul_bias_mem_ = dnnl::memory(bias_md, dnnl_engine_, inputs_[2]->getData());

      auto dst_dims = outputs_[0]->getShape();
      if (dst_dims.size() != 2) {  
        NNDEPLOY_LOGE("Output (destination) must be 2D for this bias check.\n");  
        return base::kStatusCodeErrorInvalidParam;  
      }  

      long long M = dst_dims[0]; 
      long long N = dst_dims[1]; 

      if (bias_dims.size() != 2) {  
        NNDEPLOY_LOGE("Bias tensor must be 2D for matmul with bias.\n");  
        return base::kStatusCodeErrorInvalidParam;  
      }  

      bool is_valid_bias_m_dim = (bias_dims[0] == M || bias_dims[0] == 1);
      bool is_valid_bias_n_dim = (bias_dims[1] == N || bias_dims[1] == 1);

      if (!is_valid_bias_m_dim) {  
        NNDEPLOY_LOGE("Bias first dimension (rows) must be M or 1.\n");  
        return base::kStatusCodeErrorInvalidParam;  
      }  
        
      if (!is_valid_bias_n_dim) {  
        NNDEPLOY_LOGE("Bias second dimension (cols) must be N or 1.\n");  
        return base::kStatusCodeErrorInvalidParam;  
      }

      dnnl_matmul_pd_ = dnnl::matmul::primitive_desc(dnnl_engine_, 
        matA_md, matB_md, bias_md, dst_md);
      } else {
        dnnl_matmul_pd_ = dnnl::matmul::primitive_desc(dnnl_engine_, 
          matA_md, matB_md, dst_md);
      }

    // TODO:支持算子后融合处理：https://www.intel.com/content/www/us/en/docs/onednn/developer-guide-reference/2024-2/matmul-primitive-example-001.html#DOXID-MATMUL-EXAMPLE-CPP 97:93

	  return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    std::unordered_map<int, dnnl::memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, matmul_matA_mem_});
    matmul_args.insert({DNNL_ARG_WEIGHTS, matmul_matB_mem_});
    if (have_bias_) {
      matmul_args.insert({DNNL_ARG_BIAS, matmul_bias_mem_});
    }
    matmul_args.insert({DNNL_ARG_DST, matmul_dst_mem_});
    auto matmul_e = dnnl::matmul(dnnl_matmul_pd_);
    matmul_e.execute(dnnl_stream_, matmul_args);

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
  dnnl::matmul::primitive_desc dnnl_matmul_pd_;
  dnnl::memory matmul_matA_mem_;
  dnnl::memory matmul_matB_mem_;
  dnnl::memory matmul_dst_mem_;
  dnnl::memory matmul_bias_mem_;
  bool have_bias_;
};

// 注册 OpMatMul 算子
REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86, ir::kOpTypeMatMul, X86OpMatMul)

}  // namespace op
}  // namespace nndeploy