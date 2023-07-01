/**
 * @brief 单模型推理异构推理
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char* argv[]) {
  Pipeline pipeline("single_model");

  // input output
  Packet* input = pipeline.createPacket("input");
  Packet* preprocess_output = pipeline.createPacket("preprocess_output");
  Packet* inferece_output = pipeline.createPacket("inferece_output");
  Packet* output = pipeline.createPacket("output");

  // task
  Task* preprocess =
      pipeline.createTask<PreProcess>("preprocess", input, preprocess_output);
  Task* condition = pipeline.createTask<Condition>("loop", nullptr, nullptr);
  Task* inference = pipeline.createTask<TemplateInfernce>(
      "inference", preprocess_output, inferece_output);
  Task* inference_hetro = pipeline.createTask<TemplateInfernce>(
      "inference", preprocess_output, inferece_output);
  Task* postprocess =
      pipeline.createTask<PostProcess>("postprocess", inferece_output, output);

  // param
  PreProcessParam* pre_param = new PreProcessParam();
  preprocess->setParam(pre_param);
  InfereceParam* infer_param = new InfereceParam();
  inference->setParam(infer_param);
  InfereceParam* infer_param = new InfereceParam();
  inference_hetro->setParam(infer_param);
  PostProcessParam* post_param = new PostProcessParam();
  postprocess->setParam(post_param);

  pipeline.init();

  while (true) {
    cv::Mat mat = cv::imread("test.jpg");
    input->set(mat);
    pipeline.run();
    Result* result = output->get<Base::Param>();
  }

  pipeline.deinit();

  return 0;
}