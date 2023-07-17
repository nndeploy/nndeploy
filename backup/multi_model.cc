/**
 * @brief 单模型推理异构推理
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char* argv[]) {
  Pipeline pipeline_0("model_0");

  // input_0 output_0
  Packet* input_0 = pipeline_0.createPacket("input_0");
  Packet* preprocess_output_0 = pipeline_0.createPacket("preprocess_output_0");
  Packet* preprocess_output_0 = pipeline_0.createPacket("preprocess_output_0");
  Packet* output_0 = pipeline_0.createPacket("output_0");

  // task
  Task* inference_0 = pipeline_0.createTask<TemplateInfernce>(
      "inference_0", preprocess_output_0, preprocess_output_0);
  Task* preprocess_0 = pipeline_0.createTask<PreProcess>(
      "preprocess_0", input_0, preprocess_output_0);
  Task* postprocess_0 = pipeline_0.createTask<PostProcess>(
      "postprocess_0", preprocess_output_0, output_0);

  // param
  PreProcessParam* pre_param_0 = new PreProcessParam();
  preprocess_0->setParam(pre_param_0);
  InfereceParam* infer_param_0 = new InfereceParam();
  inference_0->setParam(infer_param_0);
  PostProcessParam* post_param_0 = new PostProcessParam();
  postprocess_0->setParam(post_param_0);

  Pipeline pipeline_1("model_1");

  // input_1 output_1
  // Packet* input_1 = pipeline_1.createPacket("input_1");
  Packet* preprocess_output_1 = pipeline_1.createPacket("preprocess_output_1");
  Packet* preprocess_output_1 = pipeline_1.createPacket("preprocess_output_1");
  Packet* output_1 = pipeline_1.createPacket("output_1");

  // task
  Task* preprocess_1 = pipeline_1.createTask<PreProcess>(
      "preprocess_1", pipeline_0->getOutput(), preprocess_output_1);
  Task* inference_1 = pipeline_1.createTask<TemplateInfernce>(
      "inference_1", preprocess_output_1, preprocess_output_1);
  Task* inference_hetro = pipeline_1.createTask<TemplateInfernce>(
      "inference_1", preprocess_output_1, preprocess_output_1);
  Task* postprocess_1 = pipeline_1.createTask<PostProcess>(
      "postprocess_1", preprocess_output_1, output_1);

  // param
  PreProcessParam* pre_param_1 = new PreProcessParam();
  preprocess_1->setParam(pre_param_1);
  InfereceParam* infer_param_1 = new InfereceParam();
  inference_1->setParam(infer_param_1);
  PostProcessParam* post_param_1 = new PostProcessParam();
  postprocess_1->setParam(post_param_1);

  Pipeline pipeline("model_0_1");
  pipeline.addNode(&pipeline_0);
  pipeline.addNode(&pipeline_1);

  pipeline.init();

  while (true) {
    cv::Mat mat = cv::imread("test.jpg");
    input_0->set(mat);
    pipeline_0.run();
    Result* result = output_1->get<Base::Param>();
  }

  pipeline_0.deinit();

  return 0;
}

Pipeline* createModel_0(Packet* input_0, Packet* output_0) {
  Pipeline* pipeline = new Pipeline("model_0");

  // input_0 output_0
  // Packet* input_0 = pipeline->createPacket("input_0");
  Packet* preprocess_output_0 = pipeline->createPacket("preprocess_output_0");
  Packet* preprocess_output_0 = pipeline->createPacket("preprocess_output_0");
  // Packet* output_0 = pipeline->createPacket("output_0");

  // task
  Task* preprocess_0 = pipeline->createTask<PreProcess>("preprocess_0", input_0,
                                                        preprocess_output_0);
  Task* inference_0 = pipeline->createTask<TemplateInfernce>(
      "inference_0", preprocess_output_0, preprocess_output_0);
  Task* inference_hetro = pipeline->createTask<TemplateInfernce>(
      "inference_0", preprocess_output_0, preprocess_output_0);
  Task* postprocess_0 = pipeline->createTask<PostProcess>(
      "postprocess_0", preprocess_output_0, output_0);

  // param
  PreProcessParam* pre_param_0 = new PreProcessParam();
  preprocess_0->setParam(pre_param_0);
  InfereceParam* infer_param_0 = new InfereceParam();
  inference_0->setParam(infer_param_0);
  PostProcessParam* post_param_0 = new PostProcessParam();
  postprocess_0->setParam(post_param_0);

  return pipeline;
}

Pipeline* createModel_1(Packet* input_1, Packet* output_1) {
  Pipeline* pipeline = new Pipeline("model_1");

  // input_1 output_1
  // Packet* input_1 = pipeline->createPacket("input_1");
  Packet* preprocess_output_1 = pipeline->createPacket("preprocess_output_1");
  Packet* preprocess_output_1 = pipeline->createPacket("preprocess_output_1");
  // Packet* output_1 = pipeline->createPacket("output_1");

  // task
  Task* preprocess_1 = pipeline->createTask<PreProcess>("preprocess_1", input_1,
                                                        preprocess_output_1);
  Task* inference_1 = pipeline->createTask<TemplateInfernce>(
      "inference_1", preprocess_output_1, preprocess_output_1);
  Task* inference_hetro = pipeline->createTask<TemplateInfernce>(
      "inference_1", preprocess_output_1, preprocess_output_1);
  Task* postprocess_1 = pipeline->createTask<PostProcess>(
      "postprocess_1", preprocess_output_1, output_1);

  // param
  PreProcessParam* pre_param_1 = new PreProcessParam();
  preprocess_1->setParam(pre_param_1);
  InfereceParam* infer_param_1 = new InfereceParam();
  inference_1->setParam(infer_param_1);
  PostProcessParam* post_param_1 = new PostProcessParam();
  postprocess_1->setParam(post_param_1);

  return pipeline;
}

int main() {
  Pipeline pipeline("model_0_1");
  input_0 = pipeline.createPacket("input_0");
  output_0 = pipeline.createPacket("output_0");
  output_1 = pipeline.createPacket("output_1");

  void* ptr = malloc(100m);

  Pipeline* pipeline_0 = createModel_0(input_0, output_0, ptr);
  Pipeline* pipeline_1 = createModel_1(output_0, output_1, ptr);

  pipeline.addTask(pipeline_0);
  pipeline.addTask(pipeline_1);

  pipeline.init();

  while (true) {
    cv::Mat mat = cv::imread("test.jpg");
    input_0->set(mat);
    pipeline_0.run();
    Result* result = output_1->get<Base::Param>();
  }

  pipeline_0.deinit();

  return 0;
}