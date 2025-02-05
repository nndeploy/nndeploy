// #include "flag.h"
// #include "nndeploy/base/glic_stl_include.h"
// #include "nndeploy/base/shape.h"
// #include "nndeploy/base/time_profiler.h"
// #include "nndeploy/dag/condition.h"
// #include "nndeploy/dag/edge.h"
// #include "nndeploy/dag/graph.h"
// #include "nndeploy/dag/loop.h"
// #include "nndeploy/dag/node.h"
// #include "nndeploy/device/device.h"
// #include "nndeploy/framework.h"
// #include "nndeploy/thread_pool/thread_pool.h"

// using namespace nndeploy;

// class ProcessParam : public base::Param {
//  public:
//   base::DataType data_type_ = base::dataTypeOf<float>();
//   base::DataFormat data_format_ = base::DataFormat::kDataFormatNCHW;
//   base::IntVector shape_ = {1, 3, 512, 512};

//   size_t execute_time_ = 10;
// };

// class NNDEPLOY_CC_API ProcessNode : public dag::Node {
//  public:
//   ProcessNode(const std::string &name, dag::Edge *input, dag::Edge *output)
//       : Node(name, input, output) {
//     param_ = std::make_shared<ProcessParam>();
//     ProcessParam *op_param = dynamic_cast<ProcessParam *>(param_.get());
//   }
//   virtual ~ProcessNode() {}

//   virtual base::Status run() {
//     // NNDEPLOY_LOGE("Node name[%s], Thread ID: %d.\n", name_.c_str(),
//     //               std::this_thread::get_id());
//     ProcessParam *tmp_param = dynamic_cast<ProcessParam *>(param_.get());
//     device::Tensor *src = inputs_[0]->getTensor(this);
//     device::Device *device = device::getDefaultHostDevice();
//     device::TensorDesc desc;
//     desc.data_type_ = tmp_param->data_type_;
//     desc.data_format_ = tmp_param->data_format_;
//     desc.shape_ = tmp_param->shape_;
//     device::Tensor *dst =
//         outputs_[0]->create(device, desc, inputs_[0]->getIndex(this));

//     // execute time
//     std::this_thread::sleep_for(
//         std::chrono::milliseconds(tmp_param->execute_time_));
//     // NNDEPLOY_LOGI("running node = [%s]!\n", name_.c_str());

//     outputs_[0]->notifyWritten(dst);
//     return base::kStatusCodeOk;
//   }
// };

// class NNDEPLOY_CC_API MIMOProcessNode : public dag::Node {
//  public:
//   MIMOProcessNode(const std::string &name,
//                   std::initializer_list<dag::Edge *> inputs,
//                   std::initializer_list<dag::Edge *> outputs)
//       : dag::Node(name, inputs, outputs) {
//     param_ = std::make_shared<ProcessParam>();
//     ProcessParam *op_param = dynamic_cast<ProcessParam *>(param_.get());
//   }
//   MIMOProcessNode(const std::string &name, std::vector<dag::Edge *> inputs,
//                   std::vector<dag::Edge *> outputs)
//       : dag::Node(name, inputs, outputs) {
//     param_ = std::make_shared<ProcessParam>();
//     ProcessParam *op_param = dynamic_cast<ProcessParam *>(param_.get());
//   }
//   virtual ~MIMOProcessNode() {}

//   virtual base::Status run() {
//     // NNDEPLOY_LOGE("Node name[%s], Thread ID: %d.\n", name_.c_str(),
//     //               std::this_thread::get_id());
//     ProcessParam *tmp_param = dynamic_cast<ProcessParam *>(param_.get());
//     for (int i = 0; i < inputs_.size(); ++i) {
//       device::Tensor *src = inputs_[i]->getTensor(this);
//     }

//     // execute time
//     std::this_thread::sleep_for(
//         std::chrono::milliseconds(tmp_param->execute_time_));
//     // NNDEPLOY_LOGI("running node = [%s]!\n", name_.c_str());

//     for (int i = 0; i < outputs_.size(); ++i) {
//       device::Device *device = device::getDefaultHostDevice();
//       device::TensorDesc desc;
//       desc.data_type_ = tmp_param->data_type_;
//       desc.data_format_ = tmp_param->data_format_;
//       desc.shape_ = tmp_param->shape_;
//       device::Tensor *dst =
//           outputs_[i]->create(device, desc, inputs_[0]->getIndex(this));
//       outputs_[i]->notifyWritten(dst);
//     }

//     return base::kStatusCodeOk;
//   }
// };

// dag::Graph *createGraph(const std::string &name, dag::Edge *input,
//                         dag::Edge *output, base::ParallelType pt) {
//   dag::Graph *graph = new dag::Graph(name, input, output);
//   dag::Edge *preprocess_out = graph->createEdge(name + "_preprocess_out");
//   dag::Edge *infer_out = graph->createEdge(name + "_infer_out");
//   dag::Node *preprocess = graph->createNode<ProcessNode>(name +
//   "_preprocess",
//                                                          input,
//                                                          preprocess_out);
//   dag::Node *infer = graph->createNode<ProcessNode>(name + "_infer",
//                                                     preprocess_out,
//                                                     infer_out);
//   dag::Node *postprocess =
//       graph->createNode<ProcessNode>(name + "_postprocess", infer_out,
//       output);
//   graph->setParallelType(pt);
//   return graph;
// };

// dag::Graph *createGraphMISO(const std::string &name,
//                             std::initializer_list<dag::Edge *> inputs,
//                             dag::Edge *output, base::ParallelType pt) {
//   dag::Graph *graph = new dag::Graph(name, inputs, {output});

//   std::vector<dag::Edge *> preprocess_out;
//   int i = 0;
//   for (auto input : inputs) {
//     dag::Edge *out =
//         graph->createEdge(name + "_preprocess_out_" + std::to_string(i));
//     i++;
//     preprocess_out.emplace_back(out);
//   }
//   dag::Edge *tmp;
//   dag::Edge *infer_out = graph->createEdge(name + "_infer_out");
//   dag::Node *preprocess = graph->createNode<MIMOProcessNode>(
//       name + "_preprocess", inputs, preprocess_out);
//   dag::Node *infer = graph->createNode<MIMOProcessNode>(
//       name + "_infer", preprocess_out, {infer_out});
//   dag::Node *postprocess =
//       graph->createNode<ProcessNode>(name + "_postprocess", infer_out,
//       output);
//   graph->setParallelType(pt);
//   return graph;
// };

// class NNDEPLOY_CC_API DemoLoop : public dag::Loop {
//  public:
//   DemoLoop(const std::string &name, dag::Edge *input, dag::Edge *output)
//       : dag::Loop(name, input, output){};
//   DemoLoop(const std::string &name, std::initializer_list<dag::Edge *>
//   inputs,
//            std::initializer_list<dag::Edge *> outputs)
//       : dag::Loop(name, inputs, outputs){};
//   virtual ~DemoLoop(){};

//   virtual int loops() { return 2; };
// };

// dag::Graph *createGraphLoop(const std::string &name,
//                             std::initializer_list<dag::Edge *> inputs,
//                             dag::Edge *output, base::ParallelType pt) {
//   dag::Graph *loop_graph = new DemoLoop(name, inputs, {output});
//   return loop_graph;
// };

// class NNDEPLOY_CC_API DemoCondition : public dag::Condition {
//  public:
//   DemoCondition(const std::string &name, dag::Edge *input, dag::Edge *output)
//       : dag::Condition(name, input, output){};
//   DemoCondition(const std::string &name,
//                 std::initializer_list<dag::Edge *> inputs,
//                 std::initializer_list<dag::Edge *> outputs)
//       : dag::Condition(name, inputs, outputs){};
//   virtual ~DemoCondition(){};

//   virtual int choose() { return 1; };
// };

// dag::Graph *createGraphCondition(const std::string &name,
//                                  std::initializer_list<dag::Edge *> inputs,
//                                  dag::Edge *output, base::ParallelType pt) {
//   dag::Graph *condition_graph = new DemoCondition(name, inputs, {output});
//   return condition_graph;
// }

// int serialGraph(base::ParallelType pt_0, base::ParallelType pt_1,
//                 base::ParallelType pt, int count = 16) {
//   //
//   base::Status status = base::kStatusCodeOk;
//   dag::Edge graph_in("graph_in");
//   dag::Edge graph_out("graph_out");
//   dag::Graph *graph = new dag::Graph("serial_graph", &graph_in, &graph_out);
//   dag::Edge *model_0_out = graph->createEdge("model_0_out");
//   dag::Edge *op_link_out = graph->createEdge("op_link_out");

//   dag::Graph *model_0_graph =
//       createGraph("model_0_graph", &graph_in, model_0_out, pt_0);
//   graph->addNode(model_0_graph);
//   dag::Node *op_link =
//       graph->createNode<ProcessNode>("op_link", model_0_out, op_link_out);
//   dag::Graph *model_1_graph =
//       createGraph("model_1_graph", op_link_out, &graph_out, pt_1);
//   graph->addNode(model_1_graph);
//   status = graph->setParallelType(pt);

//   // init
//   status = graph->init();
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("graph init failed.\n");
//     return -1;
//   }

//   // dump
//   graph->dump();
//   model_0_graph->dump();
//   model_1_graph->dump();

//   // run
//   NNDEPLOY_TIME_POINT_START("graph->run");
//   for (int i = 0; i < count; ++i) {
//     // set input
//     device::Device *device = device::getDefaultHostDevice();
//     device::TensorDesc desc;
//     desc.data_type_ = base::dataTypeOf<float>();
//     desc.data_format_ = base::DataFormat::kDataFormatNCHW;
//     desc.shape_ = {1, 3, 512, 512};
//     device::Tensor *input_tensor = new device::Tensor(device, desc,
//     "graph_in"); graph_in.set(input_tensor, i, false);

//     // run
//     status = graph->run();
//     if (status != base::kStatusCodeOk) {
//       NNDEPLOY_LOGE("graph dump failed.\n");
//       return -1;
//     }

//     // get output (not base::kParallelTypePipeline)
//     if (pt != base::kParallelTypePipeline) {
//       device::Tensor *result = graph_out.getGraphOutputTensor();
//       if (result == nullptr) {
//         NNDEPLOY_LOGE("result is nullptr");
//         return -1;
//       }
//     }
//   }
//   // get output (base::kParallelTypePipeline)
//   if (pt == base::kParallelTypePipeline) {
//     for (int i = 0; i < count; ++i) {
//       device::Tensor *result = graph_out.getGraphOutputTensor();
//       if (result == nullptr) {
//         NNDEPLOY_LOGE("result is nullptr");
//         return -1;
//       }
//     }
//   }
//   NNDEPLOY_TIME_POINT_END("graph->run");
//   NNDEPLOY_TIME_PROFILER_PRINT("demo");
//   NNDEPLOY_TIME_PROFILER_RESET();

//   // 有向无环图graph反初始化
//   status = graph->deinit();
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("graph deinit failed");
//     return -1;
//   }

//   // 有向无环图graph销毁
//   delete model_0_graph;
//   delete model_1_graph;
//   delete graph;

//   return 0;
// }

// int parallelGraph(base::ParallelType pt_0, base::ParallelType pt_1,
//                   base::ParallelType pt, int count = 16) {
//   base::Status status = base::kStatusCodeOk;
//   dag::Edge graph_in("graph_in");
//   dag::Edge graph_out_0("graph_out_0");
//   dag::Edge graph_out_1("graph_out_1");
//   dag::Edge graph_out_2("graph_out_2");
//   dag::Edge graph_out_3("graph_out_3");
//   dag::Graph *graph =
//       new dag::Graph("parallel_graph", {&graph_in},
//                      {&graph_out_0, &graph_out_1, &graph_out_2,
//                      &graph_out_3});

//   dag::Graph *model_0_graph =
//       createGraph("model_0_graph", &graph_in, &graph_out_0, pt_0);
//   graph->addNode(model_0_graph);
//   dag::Node *mode_post_0 =
//       graph->createNode<ProcessNode>("mode_post_0", &graph_out_0,
//       &graph_out_1);
//   dag::Graph *model_1_graph =
//       createGraph("model_1_graph", &graph_in, &graph_out_2, pt_1);
//   graph->addNode(model_1_graph);
//   dag::Node *mode_post_1 =
//       graph->createNode<ProcessNode>("mode_post_1", &graph_out_2,
//       &graph_out_3);
//   status = graph->setParallelType(pt);

//   // init
//   status = graph->init();
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("graph init failed.\n");
//     return -1;
//   }

//   // dump
//   status = graph->dump();
//   model_0_graph->dump();
//   model_1_graph->dump();

//   // run
//   NNDEPLOY_TIME_POINT_START("graph->run");
//   for (int i = 0; i < count; ++i) {
//     device::Device *device = device::getDefaultHostDevice();
//     device::TensorDesc desc;
//     desc.data_type_ = base::dataTypeOf<float>();
//     desc.data_format_ = base::DataFormat::kDataFormatNCHW;
//     desc.shape_ = {1, 3, 512, 512};
//     device::Tensor *input_tensor =
//         new device::Tensor(device, desc, "graph_in ");
//     graph_in.set(input_tensor, i, false);

//     graph->run();
//     if (status != base::kStatusCodeOk) {
//       NNDEPLOY_LOGE("graph dump failed.\n");
//       return -1;
//     }

//     // get output (not base::kParallelTypePipeline)
//     if (pt != base::kParallelTypePipeline) {
//       device::Tensor *result_0 = graph_out_0.getGraphOutputTensor();
//       if (result_0 == nullptr) {
//         NNDEPLOY_LOGE("result_1 is nullptr");
//         return -1;
//       }
//       device::Tensor *result_1 = graph_out_1.getGraphOutputTensor();
//       if (result_1 == nullptr) {
//         NNDEPLOY_LOGE("result_1 is nullptr");
//         return -1;
//       }
//       device::Tensor *result_2 = graph_out_2.getGraphOutputTensor();
//       if (result_2 == nullptr) {
//         NNDEPLOY_LOGE("result_1 is nullptr");
//         return -1;
//       }
//       device::Tensor *result_3 = graph_out_3.getGraphOutputTensor();
//       if (result_3 == nullptr) {
//         NNDEPLOY_LOGE("result_1 is nullptr");
//         return -1;
//       }
//     }
//   }
//   if (pt == base::kParallelTypePipeline) {
//     for (int i = 0; i < count; ++i) {
//       if (pt != base::kParallelTypePipeline) {
//         device::Tensor *result_0 = graph_out_0.getGraphOutputTensor();
//         if (result_0 == nullptr) {
//           NNDEPLOY_LOGE("result_1 is nullptr");
//           return -1;
//         }
//         device::Tensor *result_1 = graph_out_1.getGraphOutputTensor();
//         if (result_1 == nullptr) {
//           NNDEPLOY_LOGE("result_1 is nullptr");
//           return -1;
//         }
//         device::Tensor *result_2 = graph_out_2.getGraphOutputTensor();
//         if (result_2 == nullptr) {
//           NNDEPLOY_LOGE("result_1 is nullptr");
//           return -1;
//         }
//         device::Tensor *result_3 = graph_out_3.getGraphOutputTensor();
//         if (result_3 == nullptr) {
//           NNDEPLOY_LOGE("result_1 is nullptr");
//           return -1;
//         }
//       }
//     }
//   }
//   NNDEPLOY_TIME_POINT_END("graph->run");
//   NNDEPLOY_TIME_PROFILER_PRINT("demo");
//   NNDEPLOY_TIME_PROFILER_RESET();

//   // 有向无环图graph反初始化
//   status = graph->deinit();
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("graph deinit failed");
//     return -1;
//   }

//   // 有向无环图graph销毁
//   delete model_0_graph;
//   delete model_1_graph;
//   delete graph;

//   return 0;
// }

// int photosRepairGraph(base::ParallelType pt_0, base::ParallelType pt_1,
//                       base::ParallelType pt, int count = 16) {
//   //
//   base::Status status = base::kStatusCodeOk;
//   dag::Edge graph_in("graph_in");
//   dag::Edge graph_out("graph_out");
//   dag::Graph *graph =
//       new dag::Graph("photosRepairGraph", &graph_in, &graph_out);

//   dag::Edge *scratch_detection_out =
//   graph->createEdge("scratch_detection_out"); dag::Graph *scratch_detection =
//       createGraph("scratch_detection", &graph_in, scratch_detection_out,
//       pt_0);
//   graph->addNode(scratch_detection);

//   dag::Edge *scratch_repair_out = graph->createEdge("scratch_repair_out");
//   dag::Graph *scratch_repair =
//       createGraphMISO("scratch_repair", {&graph_in, scratch_detection_out},
//                       scratch_repair_out, pt_0);
//   graph->addNode(scratch_repair);

//   dag::Edge *super_resolution_out =
//   graph->createEdge("super_resolution_out"); dag::Graph *super_resolution =
//   createGraph(
//       "super_resolution", scratch_repair_out, super_resolution_out, pt_0);
//   graph->addNode(super_resolution);
//   // super_resolution_out->markGraphOutput();

//   dag::Edge *face_detection_out = graph->createEdge("face_detection_out");
//   dag::Graph *face_detection = createGraph(
//       "face_detection", super_resolution_out, face_detection_out, pt_0);
//   graph->addNode(face_detection);

//   dag::Graph *condition_face_graph = createGraphCondition(
//       "condition_face_graph", {super_resolution_out, face_detection_out},
//       &graph_out, pt_0);
//   graph->addNode(condition_face_graph);

//   dag::Node *condition_no_face =
//   condition_face_graph->createNode<ProcessNode>(
//       "condition_no_face", super_resolution_out, &graph_out);

//   dag::Graph *loop_multi_face_graph = createGraphLoop(
//       "loop_multi_face_graph", {super_resolution_out, face_detection_out},
//       &graph_out, pt_0);
//   condition_face_graph->addNode(loop_multi_face_graph);

//   dag::Edge *face_correction_out =
//       loop_multi_face_graph->createEdge("face_correction_out");
//   dag::Node *face_correction =
//   loop_multi_face_graph->createNode<ProcessNode>(
//       "face_correction", face_detection_out, face_correction_out);

//   dag::Edge *face_repair_out =
//       loop_multi_face_graph->createEdge("face_repair_out");
//   dag::Graph *face_repair =
//       createGraph("face_repair", face_correction_out, face_repair_out, pt_0);
//   loop_multi_face_graph->addNode(face_repair);

//   dag::Graph *face_back = createGraphMISO(
//       "face_back", {super_resolution_out, face_repair_out}, &graph_out,
//       pt_0);
//   loop_multi_face_graph->addNode(face_back);

//   // init
//   status = graph->init();
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("graph init failed.\n");
//     return -1;
//   }

//   // dump
//   graph->dump();
//   scratch_detection->dump();
//   scratch_repair->dump();
//   super_resolution->dump();
//   face_detection->dump();
//   condition_face_graph->dump();
//   loop_multi_face_graph->dump();
//   face_repair->dump();
//   face_back->dump();
//   // model_0_graph->dump();
//   // model_1_graph->dump();

//   // run
//   NNDEPLOY_TIME_POINT_START("graph->run");
//   for (int i = 0; i < count; ++i) {
//     NNDEPLOY_LOGE("RUN START i = %d\n", i);
//     // set input
//     device::Device *device = device::getDefaultHostDevice();
//     device::TensorDesc desc;
//     desc.data_type_ = base::dataTypeOf<float>();
//     desc.data_format_ = base::DataFormat::kDataFormatNCHW;
//     desc.shape_ = {1, 3, 512, 512};
//     device::Tensor *input_tensor = new device::Tensor(device, desc,
//     "graph_in"); graph_in.set(input_tensor, i, false);

//     // run
//     status = graph->run();
//     if (status != base::kStatusCodeOk) {
//       NNDEPLOY_LOGE("graph dump failed.\n");
//       return -1;
//     }

//     // get output (not base::kParallelTypePipeline)
//     if (pt != base::kParallelTypePipeline) {
//       device::Tensor *result = graph_out.getGraphOutputTensor();
//       if (result == nullptr) {
//         NNDEPLOY_LOGE("result is nullptr");
//         return -1;
//       }
//     }
//     NNDEPLOY_LOGE("RUN END i = %d\n", i);
//   }
//   // get output (base::kParallelTypePipeline)
//   if (pt == base::kParallelTypePipeline) {
//     for (int i = 0; i < count; ++i) {
//       device::Tensor *result = graph_out.getGraphOutputTensor();
//       if (result == nullptr) {
//         NNDEPLOY_LOGE("result is nullptr");
//         return -1;
//       }
//     }
//   }
//   NNDEPLOY_TIME_POINT_END("graph->run");
//   NNDEPLOY_TIME_PROFILER_PRINT("demo");
//   NNDEPLOY_TIME_PROFILER_RESET();

//   NNDEPLOY_LOGE("graph->deinit() start.\n");
//   // 有向无环图graph反初始化
//   status = graph->deinit();
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("graph deinit failed");
//     return -1;
//   }
//   NNDEPLOY_LOGE("graph->deinit() end.\n");

//   // 有向无环图graph销毁
//   // delete model_0_graph;
//   // delete model_1_graph;
//   delete graph;

//   NNDEPLOY_LOGE("delete graph end.\n");

//   return 0;
// }

// int main(int argc, char *argv[]) {
//   NNDEPLOY_LOGE("start!\n");
//   int ret = nndeployFrameworkInit();
//   if (ret != 0) {
//     NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
//     return ret;
//   }

//   int count = 1;
//   for (int i = 0; i < count; i++) {
//     ret = photosRepairGraph(base::kParallelTypeSequential,
//                             base::kParallelTypeSequential,
//                             base::kParallelTypeSequential, 1);
//     if (ret != 0) {
//       return ret;
//     }
//     // ret = serialGraph(base::kParallelTypeSequential,
//     //                   base::kParallelTypeSequential,
//     //                   base::kParallelTypeSequential, 1);
//     // if (ret != 0) {
//     //   return ret;
//     // }
//     // ret = serialGraph(base::kParallelTypeSequential,
//     //                   base::kParallelTypeSequential,
//     //                   base::kParallelTypeSequential, 1);
//     // if (ret != 0) {
//     //   return ret;
//     // }
//     // ret = parallelGraph(base::kParallelTypeSequential,
//     //                     base::kParallelTypeSequential,
//     //                     base::kParallelTypeSequential);
//     // if (ret != 0) {
//     //   return ret;
//     // }
//     //  // parallel task grah
//     //  ret = serialGraph(base::kParallelTypeTask, base::kParallelTypeTask,
//     //                    base::kParallelTypeTask);
//     //  if (ret != 0) {
//     //    return ret;
//     //  }
//     //  ret = parallelGraph(base::kParallelTypeTask, base::kParallelTypeTask,
//     //                      base::kParallelTypeTask);
//     //  if (ret != 0) {
//     //    return ret;
//     //  }
//     //  // parallel pipepline graph
//     // ret = serialGraph(base::kParallelTypeNone, base::kParallelTypeNone,
//     //                  base::kParallelTypePipeline, 100);
//     // if (ret != 0) {
//     //  return ret;
//     //}
//     // ret = parallelGraph(base::kParallelTypeNone, base::kParallelTypeNone,
//     //                    base::kParallelTypePipeline, 100);
//     // if (ret != 0) {
//     //  return ret;
//     //}
//     //  // parallel pipepline graph / sugraph sequential
//     //  ret =
//     //      serialGraph(base::kParallelTypeSequential,
//     //                  base::kParallelTypeSequential,
//     //                  base::kParallelTypePipeline);
//     //  if (ret != 0) {
//     //    return ret;
//     //  }
//     //  ret = parallelGraph(base::kParallelTypeSequential,
//     //                      base::kParallelTypeSequential,
//     //                      base::kParallelTypePipeline);
//     //  if (ret != 0) {
//     //    return ret;
//     //  }
//     //  // parallel pipepline graph / sugraph task
//     //  ret = serialGraph(base::kParallelTypeTask, base::kParallelTypeTask,
//     //                    base::kParallelTypePipeline);
//     //  if (ret != 0) {
//     //    return ret;
//     //  }
//     //  ret = parallelGraph(base::kParallelTypeTask, base::kParallelTypeTask,
//     //                      base::kParallelTypePipeline);
//     //  if (ret != 0) {
//     //    return ret;
//     //  }

//     // TODO
//     // loop graph - 暂不支持流水线并行模式
//     // condition graph
//     // condition running graph
//   }

//   NNDEPLOY_LOGE("end!\n");

//   ret = nndeployFrameworkDeinit();
//   if (ret != 0) {
//     NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
//     return ret;
//   }
//   return 0;
// }

int main(int argc, char *argv[]) { return 0; }
