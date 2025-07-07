#include "nndeploy/op/op_transpose.h"
#include "nndeploy/op/x86/op_include.h"
#include "nndeploy/op/x86/op_util.h"
#include "nndeploy/op/x86/op_convert.h"



namespace nndeploy {
namespace op {

std::string get_format_tag_str(const dnnl::memory::desc &md) {
    // 检查未定义格式
    if (md.is_zero()) return "undef";

    // 遍历常见的 format tag 并进行比较
    // Iterate through common format tags and compare
#define CHECK_TAG(tag) \
    do { \
        auto temp_md = dnnl::memory::desc(md.get_dims(), md.get_data_type(), dnnl::memory::format_tag::tag, true); \
        if (temp_md && temp_md == md) return #tag; \
    } while (0)

    // 添加您常用的或期望检查的格式标签
    // Add the format tags you commonly use or expect to check
    CHECK_TAG(a);
    CHECK_TAG(ab);
    CHECK_TAG(abc);
    CHECK_TAG(abcd);
    CHECK_TAG(abdc);
    CHECK_TAG(acbd);
    CHECK_TAG(acdb);
    CHECK_TAG(adbc);
    CHECK_TAG(bacd);
    CHECK_TAG(bcda);
    CHECK_TAG(cdba);
    CHECK_TAG(dcab);
    CHECK_TAG(abcde);
    CHECK_TAG(abcdef);

    CHECK_TAG(nchw);
    CHECK_TAG(nhwc);
    CHECK_TAG(chwn);
    
    CHECK_TAG(ncdhw);
    CHECK_TAG(ndhwc);

    CHECK_TAG(oihw);
    CHECK_TAG(hwio);
    CHECK_TAG(goihw);

    CHECK_TAG(x);
    CHECK_TAG(nc);
    CHECK_TAG(cn);
    CHECK_TAG(nwc);
    
    // 如果没有匹配的已知格式，则返回 "unknown"
    // If no known format matches, return "unknown"
    return "unknown";
}

void print_memory_desc(const dnnl::memory::desc &md) {    
    // 现有代码...  
    auto dims = md.get_dims();    
    auto data_type = md.get_data_type();    
    auto strides = md.get_strides();  
    auto format_tag = md.get_format_kind();  // 添加这行  
    std::cout << "ndims: " << dims.size() << std::endl;    
    std::cout << "dims: ";    
    for (size_t i = 0; i < dims.size(); i++) {    
        std::cout << dims[i] << " ";    
    }    
    std::cout << std::endl;    
        
    std::cout << "data_type: " << static_cast<int>(data_type) << std::endl;  
      
    // 添加 format tag 打印  
    std::cout << "format_kind: " << static_cast<int>(format_tag) << std::endl;  
      
    std::cout << "strides: ";    
    for (size_t i = 0; i < strides.size(); i++) {    
        std::cout << strides[i] << " ";    
    }    
    std::cout << std::endl;    
    std::cout << "format_tag: " << get_format_tag_str(md) << std::endl;
    std::cout << std::endl;    

}


// Transpose 算子的 X86 (oneDNN) 实现
class X86OpTranspose : public OpTranspose {
 public:
  X86OpTranspose() : OpTranspose() {}
  virtual ~X86OpTranspose() {}
  
  // virtual base::Status X86OpTranspose::inferDataFormat() {
  //   auto param = dynamic_cast<ir::TransposeParam*>(op_desc_.op_param_.get());
  //   auto axis = param->perm_;
  //   auto src_dataformat = inputs_[0]->getDataFormat();
  //   if (axis == std::vector<int>{1, 2}) outputs_[0]->setDataFormat(base::kDataFormatAuto);
    
  // }

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

    auto src_md = dnnl::memory::desc(src_dims, src_data_type, src_dataformat);
    NNDEPLOY_LOGE("3.5");
    auto dst_md = src_md.permute_axes(axis);
    NNDEPLOY_LOGE("3.8");

    print_memory_desc(src_md);
    print_memory_desc(dst_md);

    transpose_dst_mem_ = dnnl::memory(src_md, dnnl_engine_, inputs_[0]->getData());
    transpose_dst_mem_ = dnnl::memory(dst_md_, dnnl_engine_, outputs_[0]->getData());

    dnnl_order_pd_ = dnnl::reorder::primitive_desc(
            dnnl_engine_, src_md, dnnl_engine_, dst_md);
    NNDEPLOY_LOGE("4");

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