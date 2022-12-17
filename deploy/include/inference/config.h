

struct InferenceConfig {
  InferenceType model_type_;
  InferenceType inference_type_;
  bool is_path_ = false;
  bool is_encrypt_ = false;
  std::vector<std::string> params_;

  std::vector<DeviceType> device_types_;
  ShareMemoryType share_memory_mode_ = SHARE_MEMORY_TYPE_NO_SHARE;
  PrecisionType precision_ = PRECISION_TYPE_AUTO;
  PowerType power_type = POWER_TYPE_AUTO;

  bool is_dynamic_shape_ = false;
  ShapeMap input_shape_ = ShapeMap();
  bool is_quant = false;

  InferenceOptType opt_type_ = INFERENCE_OPT_TYPE_AUTO;
  std::string cache_path_ = "";
  bool is_tune_kernel_ = false;
};