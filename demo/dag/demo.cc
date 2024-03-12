#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/model/detect/yolo/yolo.h"
#include "nndeploy/thread_pool/thread_pool.h"

using namespace nndeploy;

class ProcessParam : public base::Param {
 public:
  base::DataType data_type_ = base::dataTypeOf<float>();
  base::DataFormat data_format_ = base::DataFormat::kDataFormatNCHW;
  base::IntVector shape_ = {1, 3, 512, 512};

  size_t execute_time_ = 10;
};

class NNDEPLOY_CC_API ProcessNode : public dag::Node {
 public:
  ProcessNode(const std::string &name, dag::Edge *input, dag::Edge *output)
      : Node(name, input, output) {
    param_ = std::make_shared<ProcessParam>();
    ProcessParam *op_param = dynamic_cast<ProcessParam *>(param_.get());
  }
  virtual ~ProcessNode() {}

  virtual base::Status run() {
    // NNDEPLOY_LOGE("Node name[%s], Thread ID: %d.\n", name_.c_str(),
    //               std::this_thread::get_id());
    ProcessParam *tmp_param = dynamic_cast<ProcessParam *>(param_.get());
    device::Tensor *src = inputs_[0]->getTensor(this);
    device::Device *device = device::getDefaultHostDevice();
    device::TensorDesc desc;
    desc.data_type_ = tmp_param->data_type_;
    desc.data_format_ = tmp_param->data_format_;
    desc.shape_ = tmp_param->shape_;
    device::Tensor *dst =
        outputs_[0]->create(device, desc, inputs_[0]->getIndex(this));

    // execute time
    std::this_thread::sleep_for(
        std::chrono::milliseconds(tmp_param->execute_time_));
    // NNDEPLOY_LOGI("running node = [%s]!\n", name_.c_str());

    outputs_[0]->notifyWritten(dst);
    return base::kStatusCodeOk;
  }
};

dag::Graph *createGraph(const std::string &name, dag::Edge *input,
                        dag::Edge *output, base::ParallelType pt) {
  dag::Graph *graph = new dag::Graph(name, input, output);
  dag::Edge *preprocess_out = graph->createEdge(name + "_preprocess_out");
  dag::Edge *infer_out = graph->createEdge(name + "_infer_out");
  dag::Node *preprocess = graph->createNode<ProcessNode>(name + "_preprocess",
                                                         input, preprocess_out);
  dag::Node *infer = graph->createNode<ProcessNode>(name + "_infer",
                                                    preprocess_out, infer_out);
  dag::Node *postprocess =
      graph->createNode<ProcessNode>(name + "_postprocess", infer_out, output);
  graph->setParallelType(pt);
  return graph;
}

int serialGraph(base::ParallelType pt_0, base::ParallelType pt_1,
                base::ParallelType pt, int count = 16) {
  //
  base::Status status = base::kStatusCodeOk;
  dag::Edge graph_in("graph_in");
  dag::Edge graph_out("graph_out");
  dag::Graph *graph = new dag::Graph("serial_graph", &graph_in, &graph_out);
  dag::Edge *model_0_out = graph->createEdge("model_0_out");
  dag::Edge *op_link_out = graph->createEdge("op_link_out");

  dag::Graph *model_0_graph =
      createGraph("model_0_graph", &graph_in, model_0_out, pt_0);
  graph->addNode(model_0_graph);
  dag::Node *op_link =
      graph->createNode<ProcessNode>("op_link", model_0_out, op_link_out);
  dag::Graph *model_1_graph =
      createGraph("model_1_graph", op_link_out, &graph_out, pt_1);
  graph->addNode(model_1_graph);
  status = graph->setParallelType(pt);

  // init
  status = graph->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph init failed.\n");
    return -1;
  }

  // dump
  graph->dump();
  model_0_graph->dump();
  model_1_graph->dump();

  // run
  NNDEPLOY_TIME_POINT_START("graph->run");
  for (int i = 0; i < count; ++i) {
    // set input
    device::Device *device = device::getDefaultHostDevice();
    device::TensorDesc desc;
    desc.data_type_ = base::dataTypeOf<float>();
    desc.data_format_ = base::DataFormat::kDataFormatNCHW;
    desc.shape_ = {1, 3, 512, 512};
    device::Tensor *input_tensor = new device::Tensor(device, desc, "graph_in");
    graph_in.set(input_tensor, i, false);

    // run
    status = graph->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("graph dump failed.\n");
      return -1;
    }

    // get output (not base::kParallelTypePipeline)
    if (pt != base::kParallelTypePipeline) {
      device::Tensor *result = graph_out.getGraphOutputTensor();
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }
    }
  }
  // get output (base::kParallelTypePipeline)
  if (pt == base::kParallelTypePipeline) {
    for (int i = 0; i < count; ++i) {
      device::Tensor *result = graph_out.getGraphOutputTensor();
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }
    }
  }
  NNDEPLOY_TIME_POINT_END("graph->run");
  NNDEPLOY_TIME_PROFILER_PRINT("demo");
  NNDEPLOY_TIME_PROFILER_RESET();

  // 有向无环图graph反初始化
  status = graph->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph deinit failed");
    return -1;
  }

  // 有向无环图graph销毁
  delete model_0_graph;
  delete model_1_graph;
  delete graph;

  return 0;
}

int parallelGraph(base::ParallelType pt_0, base::ParallelType pt_1,
                  base::ParallelType pt, int count = 16) {
  base::Status status = base::kStatusCodeOk;
  dag::Edge graph_in("graph_in");
  dag::Edge graph_out_0("graph_out_0");
  dag::Edge graph_out_1("graph_out_1");
  dag::Edge graph_out_2("graph_out_2");
  dag::Edge graph_out_3("graph_out_3");
  dag::Graph *graph =
      new dag::Graph("parallel_graph", {&graph_in},
                     {&graph_out_0, &graph_out_1, &graph_out_2, &graph_out_3});

  dag::Graph *model_0_graph =
      createGraph("model_0_graph", &graph_in, &graph_out_0, pt_0);
  graph->addNode(model_0_graph);
  dag::Node *mode_post_0 =
      graph->createNode<ProcessNode>("mode_post_0", &graph_out_0, &graph_out_1);
  dag::Graph *model_1_graph =
      createGraph("model_1_graph", &graph_in, &graph_out_2, pt_1);
  graph->addNode(model_1_graph);
  dag::Node *mode_post_1 =
      graph->createNode<ProcessNode>("mode_post_1", &graph_out_2, &graph_out_3);
  status = graph->setParallelType(pt);

  // init
  status = graph->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph init failed.\n");
    return -1;
  }

  // dump
  status = graph->dump();
  model_0_graph->dump();
  model_1_graph->dump();

  // run
  NNDEPLOY_TIME_POINT_START("graph->run");
  for (int i = 0; i < count; ++i) {
    device::Device *device = device::getDefaultHostDevice();
    device::TensorDesc desc;
    desc.data_type_ = base::dataTypeOf<float>();
    desc.data_format_ = base::DataFormat::kDataFormatNCHW;
    desc.shape_ = {1, 3, 512, 512};
    device::Tensor *input_tensor =
        new device::Tensor(device, desc, "graph_in ");
    graph_in.set(input_tensor, i, false);

    graph->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("graph dump failed.\n");
      return -1;
    }

    // get output (not base::kParallelTypePipeline)
    if (pt != base::kParallelTypePipeline) {
      device::Tensor *result_0 = graph_out_0.getGraphOutputTensor();
      if (result_0 == nullptr) {
        NNDEPLOY_LOGE("result_1 is nullptr");
        return -1;
      }
      device::Tensor *result_1 = graph_out_1.getGraphOutputTensor();
      if (result_1 == nullptr) {
        NNDEPLOY_LOGE("result_1 is nullptr");
        return -1;
      }
      device::Tensor *result_2 = graph_out_2.getGraphOutputTensor();
      if (result_2 == nullptr) {
        NNDEPLOY_LOGE("result_1 is nullptr");
        return -1;
      }
      device::Tensor *result_3 = graph_out_3.getGraphOutputTensor();
      if (result_3 == nullptr) {
        NNDEPLOY_LOGE("result_1 is nullptr");
        return -1;
      }
    }
  }
  if (pt == base::kParallelTypePipeline) {
    for (int i = 0; i < count; ++i) {
      if (pt != base::kParallelTypePipeline) {
        device::Tensor *result_0 = graph_out_0.getGraphOutputTensor();
        if (result_0 == nullptr) {
          NNDEPLOY_LOGE("result_1 is nullptr");
          return -1;
        }
        device::Tensor *result_1 = graph_out_1.getGraphOutputTensor();
        if (result_1 == nullptr) {
          NNDEPLOY_LOGE("result_1 is nullptr");
          return -1;
        }
        device::Tensor *result_2 = graph_out_2.getGraphOutputTensor();
        if (result_2 == nullptr) {
          NNDEPLOY_LOGE("result_1 is nullptr");
          return -1;
        }
        device::Tensor *result_3 = graph_out_3.getGraphOutputTensor();
        if (result_3 == nullptr) {
          NNDEPLOY_LOGE("result_1 is nullptr");
          return -1;
        }
      }
    }
  }
  NNDEPLOY_TIME_POINT_END("graph->run");
  NNDEPLOY_TIME_PROFILER_PRINT("demo");
  NNDEPLOY_TIME_PROFILER_RESET();

  // 有向无环图graph反初始化
  status = graph->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph deinit failed");
    return -1;
  }

  // 有向无环图graph销毁
  delete model_0_graph;
  delete model_1_graph;
  delete graph;

  return 0;
}

int main(int argc, char *argv[]) {
  NNDEPLOY_LOGE("start!\n");
  int ret = 0;

  int count = 1;
  for (int i = 0; i < count; i++) {
    ret = serialGraph(base::kParallelTypeSequential,
                      base::kParallelTypeSequential,
                      base::kParallelTypeSequential, 1);
    if (ret != 0) {
      return ret;
    }
    ret = serialGraph(base::kParallelTypeSequential,
                      base::kParallelTypeSequential,
                      base::kParallelTypeSequential, 1000);
    if (ret != 0) {
      return ret;
    }
    // ret = parallelGraph(base::kParallelTypeSequential,
    //                     base::kParallelTypeSequential,
    //                     base::kParallelTypeSequential);
    // if (ret != 0) {
    //   return ret;
    // }
    //  // parallel task grah
    //  ret = serialGraph(base::kParallelTypeTask, base::kParallelTypeTask,
    //                    base::kParallelTypeTask);
    //  if (ret != 0) {
    //    return ret;
    //  }
    //  ret = parallelGraph(base::kParallelTypeTask, base::kParallelTypeTask,
    //                      base::kParallelTypeTask);
    //  if (ret != 0) {
    //    return ret;
    //  }
    //  // parallel pipepline graph
    ret = serialGraph(base::kParallelTypeNone, base::kParallelTypeNone,
                      base::kParallelTypePipeline, 1000);
    if (ret != 0) {
      return ret;
    }
    //  ret = parallelGraph(base::kParallelTypeNone, base::kParallelTypeNone,
    //                      base::kParallelTypePipeline);
    //  if (ret != 0) {
    //    return ret;
    //  }
    //  // parallel pipepline graph / sugraph sequential
    //  ret =
    //      serialGraph(base::kParallelTypeSequential,
    //                  base::kParallelTypeSequential,
    //                  base::kParallelTypePipeline);
    //  if (ret != 0) {
    //    return ret;
    //  }
    //  ret = parallelGraph(base::kParallelTypeSequential,
    //                      base::kParallelTypeSequential,
    //                      base::kParallelTypePipeline);
    //  if (ret != 0) {
    //    return ret;
    //  }
    //  // parallel pipepline graph / sugraph task
    //  ret = serialGraph(base::kParallelTypeTask, base::kParallelTypeTask,
    //                    base::kParallelTypePipeline);
    //  if (ret != 0) {
    //    return ret;
    //  }
    //  ret = parallelGraph(base::kParallelTypeTask, base::kParallelTypeTask,
    //                      base::kParallelTypePipeline);
    //  if (ret != 0) {
    //    return ret;
    //  }

    // TODO
    // loop graph - 暂不支持流水线并行模式
    // condition graph
    // condition running graph
  }

  NNDEPLOY_LOGE("end!\n");

  return ret;
}