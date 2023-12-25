
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
    if ([[NSFileManager defaultManager] fileExistsAtPath:model_path.path]) {
        if (@available(iOS 12.0, macOS 10.14, *)) {
          NSError* error = nil;
          NSURL* mlmodelc_url = [MLModel compileModelAtURL:model_path error:&error];
        
          mlmodel_ = [MLModel modelWithContentsOfURL:mlmodelc_url configuration:config_ error:&err_];
          CHECK_ERR(err_);
        } else {
          NNDEPLOY_LOGE("Error: CoreML only support iOS 12+.\n");
        }
    }
    
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

    auto input_name = [NSString stringWithCString:iter.first.c_str()
                                         encoding:[NSString defaultCStringEncoding]];
    
    auto desc = iter.second->getDesc();
    std::vector<int> input_strides;
    for (int i = 0; i < desc.shape_.size(); i++) {
      int strides = 1;
      for (int j = i + 1; j < desc.shape_.size(); j++) {
        strides *= desc.shape_[j]; 
      }
      input_strides.push_back(strides);

    }

    NSMutableArray * ml_shape = [[NSMutableArray alloc] init];
    NSMutableArray * ml_strides = [[NSMutableArray alloc] init];
    for(int i=0; i<desc.shape_.size(); i++){
        [ml_shape addObject:@(desc.shape_[i])];
        [ml_strides addObject:@(input_strides[i])];
    }

    MLMultiArrayDataType ml_data_type;
    switch (desc.data_type_.code_) {
      case base::kDataTypeCodeFp:
        ml_data_type = MLMultiArrayDataTypeFloat32;
        break;
      case base::kDataTypeCodeInt:
        ml_data_type = MLMultiArrayDataTypeInt32;
        break;
      default:
        NNDEPLOY_LOGE("[coreml]:input data type error!");
        break;
    }

    auto input_data = iter.second->getPtr();
    auto input_array = [[MLMultiArray alloc] initWithDataPointer:input_data
                                            shape:ml_shape
                                            dataType:ml_data_type
                                            strides:ml_strides
                                            dealloc:^(void *_Nonnull bytes){}
                                            error:&err_];
    NNDEPLOY_LOGE("coreml alloc input array error: %s\n", err_.debugDescription.UTF8String);
    
    auto input_feature_value =  [MLFeatureValue featureValueWithMultiArray:input_array];
   
    [dict_ setObject:input_feature_value
              forKey:input_name];
  }

  //forward
  MLDictionaryFeatureProvider *provider =
      [[MLDictionaryFeatureProvider alloc] initWithDictionary:dict_ error:&err_];
  auto res = (MLDictionaryFeatureProvider *)[mlmodel_ predictionFromFeatures:provider error:&err_];
  // for (auto iter : external_output_tensors_) {
  //   MLFeatureValue *value =
  //       res[[NSString stringWithCString:iter.first.c_str() encoding:NSASCIIStringEncoding]];
  //   [&](void *&&data) -> void *& { return data; }(iter.second->getPtr()) =
  //                            CVPixelBufferGetBaseAddress([value imageBufferValue]);
  // }

  //copy output
  for (auto iter : output_tensors_) {
    auto output_name = [NSString stringWithCString:iter.first.c_str() encoding:[NSString defaultCStringEncoding]];
    auto output_array = [res objectForKeyedSubscript:output_name].multiArrayValue;
    if (!output_array) {
      NNDEPLOY_LOGE("The CoreML Output is invalid");
      return base::kStatusCodeErrorInferenceCoreML;
    }
    memcpy(iter.second->getPtr(), output_array.dataPointer, iter.second->getSize());
    
  }
  return base::kStatusCodeOk;
}

device::Tensor *CoremlInference::getOutputTensorAfterRun(const std::string &name,  base::DeviceType device_type, bool is_copy,
      base::DataFormat data_format) {
  device::Device *device = device::getDevice(device_type);
  device::Tensor *internal_tensor = output_tensors_[name];
  device::TensorDesc desc = internal_tensor->getDesc();
  bool flag = is_copy || (internal_tensor->getDevice() != device);
  if (flag) {
    device::Tensor *output_tensor = new device::Tensor(device, desc, name);
    deepCopyBuffer(internal_tensor->getBuffer(), output_tensor->getBuffer());
    return output_tensor;
  } else {
    device::Tensor *output_tensor =
        new device::Tensor(desc, internal_tensor->getBuffer(), name);
    return output_tensor;
  } 
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
