
#include "nndeploy/inference/coreml/coreml_inference.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<CoremlInference>> g_coreml_inference_register(
    base::kInferenceTypeCoreML);

CoremlInference::CoremlInference(base::InferenceType type) : Inference(type) {}
CoremlInference::~CoremlInference() {}

base::Status CoremlInference::init() {
  base::Status status = base::kStatusCodeOk;

  if (device::isHostDeviceType(inference_param_->device_type_)) {
    is_share_command_queue_ = true;
  } else {
    is_share_command_queue_ = false;
  }

  CoremlInferenceParam *coreml_inference_param =
      dynamic_cast<CoremlInferenceParam *>(inference_param_);
  config_ = [MLModelConfiguration alloc];

  CoremlConvert::convertFromInferenceParam(coreml_inference_param, config_);

  if (inference_param_->is_path_) {
    NSURL *model_path =
        [NSURL fileURLWithPath:[NSString stringWithCString:inference_param_->model_value_[0].c_str()
                                                  encoding:NSASCIIStringEncoding]];
    mlmodel_ = [MLModel modelWithContentsOfURL:model_path configuration:config_ error:&err_];
    CHECK_ERR(err_);
  } else {
    NNDEPLOY_LOGI("You will load model from memory\n");
    // TODO: APPLE only support load model when running so not here

    //      [MLModel loadModelAsset:[MLModelAsset
    //      modelAssetWithSpecificationData:[NSData dataWithBytesNoCopy:(void
    //      *)inference_param_->model_value_[0].c_str()
    //      length:inference_param_->model_value_[0].length()] error:&err]
    //      configuration:config completionHandler:(mlmodel, err){
    //
    //      }];
  }
  status = allocateInputOutputTensor();
  return status;
}

base::Status CoremlInference::deinit() {
  base::Status status = deallocateInputOutputTensor();
  if (mlmodel_) {
    [mlmodel_ dealloc];
  }
  if (config_) {
    [config_ dealloc];
  }
  return status;
}

base::Status CoremlInference::reshape(base::ShapeMap &shape_map) { return base::kStatusCodeOk; }

int64_t CoremlInference::getMemorySize() { return 0; }

float CoremlInference::getGFLOPs() { return 1000.f; }

device::TensorDesc CoremlInference::getInputTensorAlignDesc(const std::string &name) {
  if (input_tensors_.count(name) > 0) {
    device::TensorDesc desc = input_tensors_[name]->getDesc();
    if (desc.shape_.size() == 5) {
      if (desc.data_format_ != base::kDataFormatNCDHW &&
          desc.data_format_ != base::kDataFormatNDHWC) {
        desc.data_format_ = base::kDataFormatNCDHW;
      }
    } else if (desc.shape_.size() == 4) {
      if (desc.data_format_ != base::kDataFormatNHWC &&
          desc.data_format_ != base::kDataFormatNCHW) {
        desc.data_format_ = base::kDataFormatNCHW;
      }
    } else if (desc.shape_.size() == 3) {
      if (desc.data_format_ != base::kDataFormatNHW && desc.data_format_ != base::kDataFormatNWC &&
          desc.data_format_ != base::kDataFormatNCW) {
        desc.data_format_ = base::kDataFormatNHW;
      }
    } else if (desc.shape_.size() == 2) {
      if (desc.data_format_ != base::kDataFormatNC) {
        desc.data_format_ = base::kDataFormatNC;
      }
    } else if (desc.shape_.size() == 1) {
      if (desc.data_format_ != base::kDataFormatN) {
        desc.data_format_ = base::kDataFormatN;
      }
    } else {
      desc.data_format_ = base::kDataFormatNotSupport;
    }
    return desc;
  } else {
    return device::TensorDesc();
  }
}

device::TensorDesc CoremlInference::getOutputTensorAlignDesc(const std::string &name) {
  if (output_tensors_.count(name) > 0) {
    device::TensorDesc desc = output_tensors_[name]->getDesc();
    if (desc.shape_.size() == 5) {
      if (desc.data_format_ != base::kDataFormatNCDHW &&
          desc.data_format_ != base::kDataFormatNDHWC) {
        desc.data_format_ = base::kDataFormatNCDHW;
      }
    } else if (desc.shape_.size() == 4) {
      if (desc.data_format_ != base::kDataFormatNHWC &&
          desc.data_format_ != base::kDataFormatNCHW) {
        desc.data_format_ = base::kDataFormatNCHW;
      }
    } else if (desc.shape_.size() == 3) {
      if (desc.data_format_ != base::kDataFormatNHW && desc.data_format_ != base::kDataFormatNWC &&
          desc.data_format_ != base::kDataFormatNCW) {
        desc.data_format_ = base::kDataFormatNHW;
      }
    } else if (desc.shape_.size() == 2) {
      if (desc.data_format_ != base::kDataFormatNC) {
        desc.data_format_ = base::kDataFormatNC;
      }
    } else if (desc.shape_.size() == 1) {
      if (desc.data_format_ != base::kDataFormatN) {
        desc.data_format_ = base::kDataFormatN;
      }
    } else {
      desc.data_format_ = base::kDataFormatNotSupport;
    }
    return desc;
  } else {
    return device::TensorDesc();
  }
}

base::Status CoremlInference::run() {
  if (dict_ == nil) {
    dict_ = [[NSMutableDictionary alloc] init];
  }
  for (auto iter : external_input_tensors_) {
    CVPixelBufferRef photodata = NULL;
    int width = iter.second->getWidth();
    int height = iter.second->getHeight();
    int stride = width;
    OSType pixelFormat = kCVPixelFormatType_OneComponent8;
    CVReturn status =
        CVPixelBufferCreateWithBytes(kCFAllocatorDefault, width, height, pixelFormat,
                                     iter.second->getPtr(), stride, NULL, NULL, NULL, &photodata);
    if (status != 0) {
      NNDEPLOY_LOGE("Tensor create failed");
    }
    MLFeatureValue *input_data = [MLFeatureValue featureValueWithPixelBuffer:photodata];
    [dict_ setObject:[NSString stringWithCString:iter.first.c_str() encoding:NSASCIIStringEncoding]
              forKey:input_data];
  }
  MLDictionaryFeatureProvider *provider =
      [[MLDictionaryFeatureProvider alloc] initWithDictionary:dict_ error:&err_];
  NSDictionary<NSString *, MLFeatureValue *> *res =
      [[mlmodel_ predictionFromFeatures:provider error:&err_] dictionary];
  // for (auto iter : external_output_tensors_) {
  //   MLFeatureValue *value =
  //       res[[NSString stringWithCString:iter.first.c_str() encoding:NSASCIIStringEncoding]];
  //   [&](void *&&data) -> void *& { return data; }(iter.second->getPtr()) =
  //                            CVPixelBufferGetBaseAddress([value imageBufferValue]);
  // }
  return base::kStatusCodeOk;
}

device::Tensor *CoremlInference::getOutputTensorAfterRun(const std::string &name,  base::DeviceType device_type, bool is_copy,
      base::DataFormat data_format) {



}

base::Status CoremlInference::allocateInputOutputTensor() {
  device::Device *device = nullptr;
  if (device::isHostDeviceType(inference_param_->device_type_)) {
    device = device::getDevice(inference_param_->device_type_);
  }
  MLModelDescription *model_description = [mlmodel_ modelDescription];
  NSDictionary<NSString *, MLFeatureDescription *> *model_input_feature =
      [model_description inputDescriptionsByName];
  for (NSString *iter in model_input_feature) {
    device::Tensor *input_tensor =
        CoremlConvert::convertToTensor(model_input_feature[iter], iter, device);
    input_tensors_.insert(
        {std::string([iter cStringUsingEncoding:NSASCIIStringEncoding]), input_tensor});
  }
  NSDictionary<NSString *, MLFeatureDescription *> *model_output_feature =
      [model_description outputDescriptionsByName];
  for (NSString *iter in model_output_feature) {
    device::Tensor *dst = CoremlConvert::convertToTensor(model_input_feature[iter], iter, device);
    output_tensors_.insert({std::string([iter cStringUsingEncoding:NSASCIIStringEncoding]), dst});
  }
  return base::kStatusCodeOk;
}

base::Status CoremlInference::deallocateInputOutputTensor() {
  for (auto iter : input_tensors_) {
    delete iter.second;
  }
  input_tensors_.clear();
  for (auto iter : output_tensors_) {
    delete iter.second;
  }
  output_tensors_.clear();
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
