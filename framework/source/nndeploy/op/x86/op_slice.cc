#include "nndeploy/op/op_slice.h"
#include "nndeploy/op/x86/op_include.h"
#include "nndeploy/op/x86/op_util.h"
#include "nndeploy/op/x86/op_convert.h"



namespace nndeploy {
namespace op {

// Slice 算子的 X86 (oneDNN) 实现
class X86OpSlice : public OpSlice {
 public:
  X86OpSlice() : OpSlice() {}
  virtual ~X86OpSlice() {}

  virtual base::Status init() {
    base::Status status = OpSlice::init();
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

    dnnl::memory::dims input_dims =
        X86OpConvert::convertFromShape(inputs_[0]->getShape());
    dnnl::memory::data_type input_data_type =
        X86OpConvert::convertFromDataType(inputs_[0]->getDataType());
    dnnl::memory::format_tag input_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[0]->getDataFormat());

    auto input_md = dnnl::memory::desc(input_dims, input_data_type, input_dataformat);

    int ndims = input_md.get_ndims();
    auto input_strides = input_md.get_strides();
    dnnl::memory::dims slice_offsets(ndims, 0);
    dnnl::memory::dims slice_dims = input_dims;
    dnnl::memory::dims slice_strides = input_strides;

    int32_t* starts = static_cast<int32_t*>(inputs_[1]->getData());
    int32_t* ends = static_cast<int32_t*>(inputs_[2]->getData());
    int32_t* axes = static_cast<int32_t*>(inputs_[3]->getData());
    int32_t* steps = static_cast<int32_t*>(inputs_[4]->getData());

    int startslen = input_dims.size();

    for (int i = 0; i < startslen; ++i) {
        int axis = axes[i];
        if (axis < 0) { // 处理负数轴
            axis += input_dims.size();
        }

        long start = starts[i];
        long end = ends[i];
        long max_dim = input_dims[axis];

        // 对 start 和 end 的值进行规范化，处理负数和越界值
        if (start < 0) start += max_dim;
        if (end < 0) end += max_dim;

        // PyTorch/ONNX 行为: 将 start 和 end 裁剪到 [0, max_dim] 范围内
        start = std::max(0L, std::min(start, max_dim));
        end = std::max(0L, std::min(end, max_dim));

        if (start >= end) {
             // 如果起始大于等于结束，该维度大小为0，结果是空张量
             // 这里需要根据框架的具体定义来处理，通常是返回一个该维度为0的张量
             slice_dims[axis] = 0;
        } else {
             slice_offsets[axis] = start;
             slice_dims[axis] = end - start;
        }
    }

    slice_input_mem_ = dnnl::memory(input_md, dnnl_engine_, inputs_[0]->getData());
    auto src_sub_md = input_md.submemory_desc(slice_dims, slice_offsets);
    auto dst_md = dnnl::memory::desc(slice_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw); 
    slice_dst_mem_ = dnnl::memory(dst_md, dnnl_engine_, outputs_[0]->getData());

    dnnl_slice_pd_ = dnnl::reorder::primitive_desc(dnnl_engine_, src_sub_md, 
        dnnl_engine_, dst_md);

	  return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    auto slice = dnnl::reorder(dnnl_slice_pd_);
    std::unordered_map<int, dnnl::memory> slice_args;
    slice_args.insert({DNNL_ARG_SRC,slice_input_mem_});
    slice_args.insert({DNNL_ARG_DST, slice_dst_mem_});

    slice.execute(dnnl_stream_, slice_args);

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
  dnnl::memory slice_input_mem_;
  dnnl::memory slice_dst_mem_;
  dnnl::reorder::primitive_desc dnnl_slice_pd_;
};

// 注册 OpSlice 算子
REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86, ir::kOpTypeSlice, X86OpSlice)

}  // namespace op
}  // namespace nndeploy