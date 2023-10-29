# 部署一个新算法

以YOLOV5为例。源文件 - source\nndeploy\model\detect\yolo\yolo.cc，头文件 - include\nndeploy\model\detect\yolo\yolo.h
+ 准备模型文件
+ 搭建模型部署的有向无环图
  ```c++
  dag::Graph* createYoloV5Graph(const std::string& name,
                                      base::InferenceType inference_type,
                                      base::DeviceType device_type,
                                      dag::Edge* input, dag::Edge* output,
                                      base::ModelType model_type, bool is_path,
                                      std::vector<std::string>& model_value) {
    dag::Graph* graph = new dag::Graph(name, input, output); // 有向无环图
    dag::Edge* infer_input = graph->createEdge("infer_input"); // infer任务的输入
    dag::Edge* infer_output = graph->createEdge("infer_output"); // infer任务的输出
    // YOLOV5模型前处理任务model::CvtColorResize，输入边为input，输出边为infer_input
    dag:::Node* pre = graph->createNode<model::CvtColorResize>(
        "preprocess", input, infer_input);
    // YOLOV5模型推理任务model::Infer(通用模板)，输入边为infer_input，输出边为infer_output
    dag:::Node* infer = graph->createInfer<model::Infer>(
        "infer", inference_type, infer_input, infer_output);
    // YOLOV5模型后处理任务YoloPostProcess，输入边为infer_output，输出边为output
    dag:::Node* post = graph->createNode<YoloPostProcess>(
        "postprocess", infer_output, output);
    // YOLOV5模型前处理任务pre的参数配置
    model::CvtclorResizeParam* pre_param =
        dynamic_cast<model::CvtclorResizeParam*>(pre->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    // YOLOV5模型推理任务infer的参数配置
    inference::InferenceParam* inference_param =
        (inference::InferenceParam*)(infer->getParam());
    inference_param->is_path_ = is_path;
    inference_param->model_value_ = model_value;
    inference_param->device_type_ = device_type;

    // YOLOV5模型后处理任务post的参数配置
    YoloPostParam* post_param = dynamic_cast<YoloPostParam*>(post->getParam());
    post_param->score_threshold_ = 0.5;
    post_param->nms_threshold_ = 0.45;
    post_param->num_classes_ = 80;
    post_param->model_h_ = 640;
    post_param->model_w_ = 640;
    post_param->version_ = 5;

    return graph;
  }
  ```
  `注：前后处理任务有时候需要自己写`
+ 注册createYoloV5Graph
  ```c++
  #define NNDEPLOY_YOLOV5 "NNDEPLOY_YOLOV5"
  class dag::TypeGraphRegister g_register_yolov5_graph(NNDEPLOY_YOLOV5,
                                                    createYoloV5Graph);
  ```