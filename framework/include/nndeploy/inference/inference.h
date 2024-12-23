
#ifndef _NNDEPLOY_INFERENCE_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_INFERENCE_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference_param.h"

namespace nndeploy {
namespace inference {

/**
 * @brief 推理的基类
 * @details
 * # 根据InferencParam *param初始化
 * # 写入输入tensor数据
 * # 推理
 * # 得到输出tensor数据
 * # 其他
 * ## 获取输入输出tensor的信息
 * ### 动态输入
 * ### 动态输出
 * ### 是否可以操作推理框架分配的输入
 * ### 是否可以操作推理框架分配的输出
 * ## 获取推理初始化后的各种信息
 * ### 例如内存大小
 * ### gflops等
 *
 */
class NNDEPLOY_CC_API Inference {
 public:
  Inference(base::InferenceType type);
  virtual ~Inference();

  base::InferenceType getInferenceType();

  /**
   * @brief Set the Inference Param（这里使用基类指针）
   *
   * @param param
   * @return base::Status
   */
  base::Status setParam(base::Param *param);
  /**
   * @brief Get the Inference Param（这里使用基类指针）
   *
   * @return base::Param*
   */
  base::Param *getParam();

  /**
   * @brief 初始化推理
   *
   * @return base::Status
   */
  virtual base::Status init() = 0;
  /**
   * @brief 反初始化推理
   *
   * @return base::Status
   */
  virtual base::Status deinit() = 0;

  /**
   * @brief 针对动态输入的推理，获取输入tensor的min_shape
   *
   * @return base::ShapeMap
   */
  base::ShapeMap getMinShape();
  /**
   * @brief 针对动态输入的推理，获取输入tensor的opt_shape
   *
   * @return base::ShapeMap
   */
  base::ShapeMap getOptShape();
  /**
   * @brief 针对动态输入的推理，获取输入tensor的max_shape
   *
   * @return base::ShapeMap
   */
  base::ShapeMap getMaxShape();
  /**
   * @brief 针对动态输入的推理，设置输入tensor的shape
   *
   * @param shape_map
   * @return base::Status
   */
  virtual base::Status reshape(base::ShapeMap &shape_map) = 0;

  /**
   * @brief 获取推理所需的内存大小
   *
   * @return int64_t
   */
  virtual int64_t getMemorySize();
  /**
   * @brief 设置推理所需的内存（推理内存由外部分配）
   *
   * @param buffer
   * @return base::Status
   */
  virtual base::Status setMemory(device::Buffer *buffer);
  /**
   * @brief 获得推理计算量
   *
   * @return float
   */
  virtual float getGFLOPs();

  /**
   * @brief 是否为多batch推理
   *
   * @return bool
   */
  virtual bool isBatch();
  /**
   * @brief 该推理实例是否与nndeploy共享一个command queue
   *
   * @return bool
   */
  virtual bool isShareCommanQueue();
  /**
   * @brief 是否为动态输入
   *
   * @return bool
   */
  virtual bool isInputDynamic();
  /**
   * @brief 是否为动态输出
   *
   * @return bool
   */
  virtual bool isOutputDynamic();
  /**
   * @brief 是否可以操作推理框架内部分配的输入tensor
   *
   * @return true
   * @details
   * 部分推理框架会为输入tensor分配内存，以TNN为例，其会为输入tensor分配内存，当为CPU推理时，可以操作这些内存
   * 但是当为OpenCL推理时，这些内存是在OpenCL上分配的，TNN无法与外部共享GPU上下文，故无法操作这些内存
   */
  virtual bool canOpInput();
  /**
   * @brief 是否可以操作推理框架内部分配的输出tensor
   *
   * @return true
   * @details
   * 部分推理框架会为输出tensor分配内存，以TNN为例，其会为输出tensor分配内存，当为CPU推理时，可以操作这些内存
   * 但是当为OpenCL推理时，这些内存是在OpenCL上分配的，TNN无法与外部共享GPU上下文，故无法操作这些内存
   */
  virtual bool canOpOutput();

  /**
   * @brief Get the Num Of Input Tensor object
   *
   * @return int
   */
  virtual int getNumOfInputTensor();
  /**
   * @brief Get the Num Of Output Tensor object
   *
   * @return int
   */
  virtual int getNumOfOutputTensor();

  /**
   * @brief Get the Input Name object
   *
   * @param i
   * @return std::string
   */
  virtual std::string getInputName(int i);
  /**
   * @brief Get the Output Name object
   *
   * @param i
   * @return std::string
   */
  virtual std::string getOutputName(int i);
  /**
   * @brief Get the All Input Tensor Name object
   *
   * @return std::vector<std::string>
   */
  virtual std::vector<std::string> getAllInputTensorName();
  /**
   * @brief Get the All Output Tensor Name object
   *
   * @return std::vector<std::string>
   */
  virtual std::vector<std::string> getAllOutputTensorName();

  /**
   * @brief Get the Input Shape object
   *
   * @param name
   * @return base::IntVector
   */
  virtual base::IntVector getInputShape(const std::string &name);
  /**
   * @brief Get the Output Shape object
   *
   * @param name
   * @return base::IntVector
   */
  virtual base::ShapeMap getAllInputShape();
  /**
   * @brief Get the Input Tensor Desc object
   *
   * @param name
   * @return device::TensorDesc
   */
  virtual device::TensorDesc getInputTensorDesc(const std::string &name);
  /**
   * @brief Get the Output Tensor Desc object
   *
   * @param name
   * @return device::TensorDesc
   */
  virtual device::TensorDesc getOutputTensorDesc(const std::string &name);

  /**
   * @brief Get the Input Tensor Align Desc object
   *
   * @param name
   * @return device::TensorDesc
   */
  virtual device::TensorDesc getInputTensorAlignDesc(const std::string &name);
  /**
   * @brief Get the Output Tensor Align Desc object
   *
   * @param name
   * @return device::TensorDesc
   */
  virtual device::TensorDesc getOutputTensorAlignDesc(const std::string &name);

  /**
   * @brief Get the All Input Tensor Map object
   *
   * @return std::map<std::string, device::Tensor *>
   */
  virtual std::map<std::string, device::Tensor *> getAllInputTensorMap();
  /**
   * @brief Get the All Output Tensor Map object
   *
   * @return std::map<std::string, device::Tensor *>
   */
  virtual std::map<std::string, device::Tensor *> getAllOutputTensorMap();

  /**
   * @brief Get the All Input Tensor Vector object
   *
   * @return std::vector<device::Tensor *>
   */
  virtual std::vector<device::Tensor *> getAllInputTensorVector();
  /**
   * @brief Get the All Output Tensor Vector object
   *
   * @return std::vector<device::Tensor *>
   */
  virtual std::vector<device::Tensor *> getAllOutputTensorVector();

  /**
   * @brief Get the Input Tensor object
   *
   * @param name
   * @return device::Tensor*
   */
  virtual device::Tensor *getInputTensor(const std::string &name);
  /**
   * @brief Get the Output Tensor object
   *
   * @param name
   * @return device::Tensor*
   */
  virtual device::Tensor *getOutputTensor(const std::string &name);

  /**
   * @brief Set the Input Tensor object
   *
   * @param name
   * @param input_tensor
   * @return base::Status
   * @details
   * 传入外部的tensor
   */
  virtual base::Status setInputTensor(const std::string &name,
                                      device::Tensor *input_tensor);

  /**
   * @brief 推理
   *
   * @return base::Status
   */
  virtual base::Status run() = 0;

  /**
   * @brief Get the Output Tensor object
   *
   * @param name
   * @return device::Tensor*
   */
  virtual device::Tensor *getOutputTensorAfterRun(
      const std::string &name, base::DeviceType device_type, bool is_copy,
      base::DataFormat data_format = base::kDataFormatAuto) = 0;

 protected:
  /**
   * @brief 推理框架的类型
   */
  base::InferenceType type_;
  /**
   * @brief 推理框架的配置
   * @details
   * 构造推理类时，也会构造一个默认的配置，可尽量减少用户需要配置的参数
   */
  InferenceParam *inference_param_;

  /**
   * @brief 第三方推理框架是否与nndeploy共用一个command queue
   * @details
   * 在初始化时，如果第三方推理框架与nndeploy共用一个command
   * queue，需要设置为true，否则设置为false
   */
  bool is_share_command_queue_ = false;

  /**
   * @brief 输入tensor的map
   * @details
   * 部分第三方推理框架（TNN、MNN、Openvino等）内部分配输入tensor，这是第三方推理框架内部分配的输入tensor的一个浅拷贝
   * 部分推理框架（TensorRt、onnxruntime等）内部不分配输入tensor（也或许是外部无法获得到内部分配的输入tensor），nndeploy的推理将尝试为第三方推理框架分配输入tensor
   */
  std::map<std::string, device::Tensor *> input_tensors_;
  /**
   * @brief 输出tensor的map
   * @details
   * 部分第三方推理框架（TNN、MNN、Openvino等）内部分配输出tensor，这是第三方推理框架内部分配的输出tensor的一个浅拷贝
   * 部分推理框架（TensorRt、onnxruntime等）内部不分配输出tensor（也或许是外部无法获得到内部分配的输出tensor），nndeploy的推理将尝试为第三方推理框架分配输出tensor
   */
  std::map<std::string, device::Tensor *> output_tensors_;

  /**
   * @brief 外部输入tensor的map
   * @details
   * 外部输入tensor的map，也可以是input_tensors_
   */
  std::map<std::string, device::Tensor *> external_input_tensors_;
};

/**
 * @brief 推理框架的创建类
 *
 */
class InferenceCreator {
 public:
  virtual ~InferenceCreator(){};
  virtual Inference *createInference(base::InferenceType type) = 0;
};

/**
 * @brief 推理框架的创建类模板
 *
 * @tparam T
 */
template <typename T>
class TypeInferenceCreator : public InferenceCreator {
  virtual Inference *createInference(base::InferenceType type) {
    return new T(type);
  }
};

/**
 * @brief Get the Global Inference Creator Map object
 *
 * @return std::map<base::InferenceType, std::shared_ptr<InferenceCreator>>&
 */
std::map<base::InferenceType, std::shared_ptr<InferenceCreator>>
    &getGlobalInferenceCreatorMap();

/**
 * @brief 推理框架的创建类的注册类模板
 *
 * @tparam T
 */
template <typename T>
class TypeInferenceRegister {
 public:
  explicit TypeInferenceRegister(base::InferenceType type) {
    getGlobalInferenceCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

/**
 * @brief Create a Inference object
 *
 * @param type
 * @return Inference*
 */
extern NNDEPLOY_CC_API Inference *createInference(base::InferenceType type);

}  // namespace inference
}  // namespace nndeploy

#endif