#include "nndeploy/op/op_gather.h"
#include "nndeploy/op/x86/op_include.h"
#include "nndeploy/op/x86/op_util.h"
#include "nndeploy/op/x86/op_convert.h"



namespace nndeploy {
namespace op {

// Gather 算子的 X86 (oneDNN) 实现
class X86OpGather : public OpGather {
 public:
  X86OpGather() : OpGather() {}
  virtual ~X86OpGather() {}

  virtual base::Status init() {
    base::Status status = OpGather::init();
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
    auto param = dynamic_cast<ir::GatherParam *>(op_desc_.op_param_.get());
    gather_dim_ = param->axis_;

    input_dims_ =
        X86OpConvert::convertFromShape(inputs_[0]->getShape());
    dnnl::memory::data_type input_data_type =
        X86OpConvert::convertFromDataType(inputs_[0]->getDataType());
    dnnl::memory::format_tag input_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[0]->getDataFormat());

    index_dims_ =
        X86OpConvert::convertFromShape(inputs_[1]->getShape());
    dnnl::memory::data_type index_data_type =
        X86OpConvert::convertFromDataType(inputs_[1]->getDataType());
    dnnl::memory::format_tag index_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[1]->getDataFormat());

    auto input_md = dnnl::memory::desc(input_dims_, input_data_type, input_dataformat);
    auto index_md = dnnl::memory::desc(index_dims_, index_data_type, index_dataformat);
    dst_md_ = dnnl::memory::desc(index_dims_, input_data_type, index_dataformat);
    gather_input_mem_ = dnnl::memory(input_md, dnnl_engine_, inputs_[0]->getData());
    gather_index_mem_ = dnnl::memory(index_md, dnnl_engine_, inputs_[1]->getData());
    gather_dst_mem_ = dnnl::memory(input_md, dnnl_engine_, outputs_[0]->getData());

	  return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    {
        // 从 memory 对象获取原始指针
        float* src_ptr = static_cast<float*>(gather_input_mem_.get_data_handle());
        int32_t* index_ptr = static_cast<int32_t*>(gather_index_mem_.get_data_handle());
        float* dst_ptr = static_cast<float*>(gather_dst_mem_.get_data_handle());

        // 获取维度的信息
        const auto& dims = index_dims_;
        const int ndims = (int)dims.size();
        const long total_elements = dst_md_.get_size() / sizeof(float); //目前暂时只支持fp32
        // 预先计算源张量的步长 (strides)
        std::vector<long> src_strides(ndims);
        src_strides[ndims - 1] = 1;
        for (int i = ndims - 2; i >= 0; --i) {
            src_strides[i] = src_strides[i + 1] * input_dims_[i + 1];
        }

        for (long i = 0; i < total_elements; ++i) {
            // `i` 是输出张量中的线性索引 (dst_offset)
            long dst_offset = i;

            // 获取要 gather 的索引值
            int32_t gather_index_val = index_ptr[dst_offset];

            // 构造源张量的多维索引 (src_coords)
            std::vector<long> src_coords(ndims);
            long temp_idx = dst_offset;

            // 从线性索引 dst_offset 计算输出张量的多维索引
            // 这个多维索引大部分将直接用于源张量
            for (int d = ndims - 1; d >= 0; --d) {
                // `out_dims` 和 `index_dims` 相同
                int current_dim_size = index_dims_[d];
                src_coords[d] = temp_idx % current_dim_size;
                temp_idx /= current_dim_size;
            }

            // 将 GATHER_DIM 维度的索引替换为从 index tensor 中获取的值
            src_coords[gather_dim_] = gather_index_val;

            // 使用步长将源的多维索引 (src_coords) 转换为线性偏移 (src_offset)
            long src_offset = 0;
            for (int d = 0; d < ndims; ++d) {
                src_offset += src_coords[d] * src_strides[d];
            }

            // 执行 gather
            dst_ptr[dst_offset] = src_ptr[src_offset];
        }
    }

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
  dnnl::binary::primitive_desc dnnl_gather_pd_;
  dnnl::memory gather_input_mem_;
  dnnl::memory gather_index_mem_;
  dnnl::memory gather_dst_mem_;
  dnnl::memory::dims index_dims_;
  dnnl::memory::dims input_dims_;
  dnnl::memory::desc dst_md_;
  uint gather_dim_;
};

// 注册 OpGather 算子
REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86, ir::kOpTypeGather, X86OpGather)

}  // namespace op
}  // namespace nndeploy