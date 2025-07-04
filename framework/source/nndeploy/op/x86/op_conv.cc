#include "nndeploy/op/op_conv.h"
#include "nndeploy/op/x86/op_convert.h"
#include "nndeploy/op/x86/op_include.h"
#include "nndeploy/op/x86/op_util.h"

namespace nndeploy {
namespace op {
class X86OpConv : public OpConv {
 public:
  X86OpConv() : OpConv() {}
  virtual ~X86OpConv() {}

  virtual base::Status init() {
    base::Status status = OpConv::init();
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
    dnnl::memory::dims src_dims =
        X86OpConvert::convertFromShape(inputs_[0]->getShape());
    dnnl::memory::dims weights_dims =
        X86OpConvert::convertFromShape(inputs_[1]->getShape());
    dnnl::memory::dims dst_dims =
        X86OpConvert::convertFromShape(outputs_[0]->getShape());

    dnnl::memory::data_type src_data_type =
        X86OpConvert::convertFromDataType(inputs_[0]->getDataType());
    dnnl::memory::data_type weights_data_type =
        X86OpConvert::convertFromDataType(inputs_[1]->getDataType());
    dnnl::memory::data_type dst_data_type =
        X86OpConvert::convertFromDataType(outputs_[0]->getDataType());

    dnnl::memory::format_tag src_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[0]->getDataFormat());
    dnnl::memory::format_tag weights_dataformat =
        X86OpConvert::convertFromDataFormat(inputs_[1]->getDataFormat());
    dnnl::memory::format_tag dst_dataformat =
        X86OpConvert::convertFromDataFormat(outputs_[0]->getDataFormat());

    auto src_md = dnnl::memory::desc(src_dims, src_data_type, src_dataformat);
    auto weights_md =
        dnnl::memory::desc(weights_dims, weights_data_type, weights_dataformat);
    auto dst_md = dnnl::memory::desc(dst_dims, dst_data_type, dst_dataformat);

    auto src_mem = dnnl::memory(src_md, dnnl_engine_, inputs_[0]->getData());
    auto weights_mem =
        dnnl::memory(weights_md, dnnl_engine_, inputs_[1]->getData());
    auto dst_mem = dnnl::memory(dst_md, dnnl_engine_, outputs_[0]->getData());

    auto conv_src_md = dnnl::memory::desc(src_dims, src_data_type,
                                          dnnl::memory::format_tag::any);
    auto conv_weights_md = dnnl::memory::desc(weights_dims, weights_data_type,
                                              dnnl::memory::format_tag::any);
    auto conv_dst_md = dnnl::memory::desc(dst_dims, dst_data_type,
                                          dnnl::memory::format_tag::any);

    auto user_bias_md =
        inputs_.size() <= 2
            ? dnnl::memory::desc()
            : dnnl::memory::desc(
                  X86OpConvert::convertFromShape(inputs_[2]->getShape()),
                  X86OpConvert::convertFromDataType(inputs_[2]->getDataType()),
                  X86OpConvert::convertFromDataFormat(
                      inputs_[2]->getDataFormat()));
    conv_bias_mem_ =
        inputs_.size() <= 2
            ? dnnl::memory(user_bias_md, dnnl_engine_)
            : dnnl::memory(user_bias_md, dnnl_engine_, inputs_[2]->getData());

    dnnl::primitive_attr conv_attr;

    // 融合激活函数
    auto param = dynamic_cast<ir::ConvParam*>(op_desc_.op_param_.get());
    switch (param->activate_op_) {
      // NOT FUSE
      case ir::kOpTypeNone:
        break;
      case ir::kOpTypeRelu: {
        const float alpha = 0.f;
        const float beta = 0.f;
        dnnl::post_ops conv_ops;
        conv_ops.append_eltwise(dnnl::algorithm::eltwise_relu, alpha, beta);
        conv_attr.set_post_ops(conv_ops);
        break;
      }
      default:
        NNDEPLOY_LOGI("not implemented.\n");
        return base::kStatusCodeOk;
    }

    dnnl_conv_pd_ = dnnl::convolution_forward::primitive_desc(
        dnnl_engine_, dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_auto, conv_src_md, conv_weights_md,
        user_bias_md, conv_dst_md,
        X86OpConvert::convertFromShape(param->strides_),
        {param->pads_[0], param->pads_[1]}, {param->pads_[2], param->pads_[3]},
        conv_attr);

    // 如果由dnnl原语生成的源数据（src）和权重（weights）的内存布局与用户提供的不同，
    // 那么需要重新排列数据。
    conv_src_mem_ = src_mem;
    conv_weights_mem_ = weights_mem;
    user_dst_mem_ = dst_mem;

    if (dnnl_conv_pd_.src_desc() != src_mem.get_desc()) {
      conv_src_mem_ = dnnl::memory(dnnl_conv_pd_.src_desc(), dnnl_engine_);
      dnnl::reorder(src_mem, conv_src_mem_)
          .execute(dnnl_stream_, src_mem, conv_src_mem_);
    }
    if (dnnl_conv_pd_.weights_desc() != weights_mem.get_desc()) {
      conv_weights_mem_ =
          dnnl::memory(dnnl_conv_pd_.weights_desc(), dnnl_engine_);
      dnnl::reorder(weights_mem, conv_weights_mem_)
          .execute(dnnl_stream_, weights_mem, conv_weights_mem_);
    }
    // 按照原语提供的dst_desc 描述 dst_mem
    // 在运行完毕后，需要再转换为nndeploy规定的内存布局。即user_dst_mem_
    if (dnnl_conv_pd_.dst_desc() != dst_mem.get_desc()) {
      conv_dst_mem_ = dnnl::memory(dnnl_conv_pd_.dst_desc(), dnnl_engine_);
      dnnl::reorder(dst_mem, conv_dst_mem_)
          .execute(dnnl_stream_, dst_mem, conv_dst_mem_);
    }

    return base::kStatusCodeOk;
  }
  
  virtual base::Status run() {
    auto dnnl_conv = dnnl::convolution_forward(dnnl_conv_pd_);
    dnnl_conv.execute(dnnl_stream_, {{DNNL_ARG_SRC, conv_src_mem_},
                                     {DNNL_ARG_WEIGHTS, conv_weights_mem_},
                                     {DNNL_ARG_BIAS, conv_bias_mem_},
                                     {DNNL_ARG_DST, conv_dst_mem_}});

    // dnnl生成的dst内存排布可能与nndeploy要求的不同，需要重排
    if (dnnl_conv_pd_.dst_desc() != user_dst_mem_.get_desc()) {
      dnnl::reorder(conv_dst_mem_, user_dst_mem_)
          .execute(dnnl_stream_, conv_dst_mem_, user_dst_mem_);
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
  dnnl::convolution_forward::primitive_desc dnnl_conv_pd_;
  dnnl::memory conv_src_mem_;
  dnnl::memory conv_weights_mem_;
  dnnl::memory conv_bias_mem_;
  dnnl::memory conv_dst_mem_;  // dnnl生成的高效dst内存排布
  dnnl::memory user_dst_mem_;  // nndeploy规定的dst内存排布，运行完后需转换回来
};

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86, ir::kOpTypeConv, X86OpConv)

}  // namespace op
}  // namespace nndeploy