#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/model/detect/yolo/yolo.h"
#include "nndeploy/thread_pool/thread_pool.h"

using namespace nndeploy;

enum OpType {
  kOpTypeConv,
  kOpTypeRelu,
};

class OpParam : public base::Param {
 public:
  OpType type_ = kOpTypeConv;
  base::DataType data_type_ = base::dataTypeOf<float>();
  base::DataFormat data_format_ = base::DataFormat::kDataFormatNCHW;
  base::IntVector shape_ = {1, 3, 512, 512};

  size_t execute_time_ = 10;
};

class NNDEPLOY_CC_API OpNode : public dag::Node {
 public:
  OpNode(const std::string &name, dag::Edge *input, dag::Edge *output)
      : Node(name, input, output) {
    param_ = std::make_shared<OpParam>();
    OpParam *op_param = dynamic_cast<OpParam *>(param_.get());
    op_param->type_ = kOpTypeConv;
  }
  virtual ~OpNode() {}

  virtual base::Status run() {
    NNDEPLOY_LOGE("Node name[%s], Thread ID: %d.\n", name_.c_str(),
                  std::this_thread::get_id());
    OpParam *tmp_param = dynamic_cast<OpParam *>(param_.get());
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

    outputs_[0]->notifyWritten(dst);
    return base::kStatusCodeOk;
  }
};

// {
//   sub_graph_0->init();
//   sub_graph_0->run();

//   sub_graph_0->dump();

//   int count = 2;
//   for (int i = 0; i < count; ++i) {
//     device::Device *device = device::getDefaultHostDevice();
//     device::TensorDesc desc;
//     desc.data_type_ = base::dataTypeOf<float>();
//     desc.data_format_ = base::DataFormat::kDataFormatNCHW;
//     desc.shape_ = {1, 3, 512, 512};
//     device::Tensor *input_tensor =
//         new device::Tensor(device, desc, "sub_in_0 ");

//     sub_in_0.set(input_tensor, i, false);
//   }

//   for (int i = 0; i < count; ++i) {
//     device::Tensor *result = sub_out_0.getGraphOutputTensor();
//     if (result == nullptr) {
//       NNDEPLOY_LOGE("result is nullptr");
//       return -1;
//     }
//   }

//   // 有向无环图graph反初始化
//   status = sub_graph_0->deinit();
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("graph deinit failed");
//     return -1;
//   }

//   // delete sep_graph;
//   return 0;
// }

int main(int argc, char *argv[]) {
  dag::Edge sub_in_0("sub_in_0");
  dag::Edge sub_out_0("sub_out_0");
  dag::Graph *sub_graph_0 =
      new dag::Graph("sub_graph_0", &sub_in_0, &sub_out_0);
  dag::Edge *op_0_1_out = sub_graph_0->createEdge("op_1_1_out");
  dag::Node *op_0_1 =
      sub_graph_0->createNode<OpNode>("op_0_1", &sub_in_0, op_0_1_out);
  dag::Node *op_0_2 =
      sub_graph_0->createNode<OpNode>("op_0_2", op_0_1_out, &sub_out_0);
  base::Status status =
      sub_graph_0->setParallelType(dag::kParallelTypePipeline);

  dag::Edge sub_in_1("sub_in_1");
  dag::Edge sub_out_1("sub_out_1");
  dag::Graph *sub_graph_1 =
      new dag::Graph("sub_graph_1", &sub_in_1, &sub_out_1);
  dag::Edge *op_1_1_out = sub_graph_1->createEdge("op_1_1_out");
  dag::Node *op_1_1 =
      sub_graph_1->createNode<OpNode>("op_1_1", &sub_in_1, op_1_1_out);
  dag::Node *op_1_2 =
      sub_graph_1->createNode<OpNode>("op_1_2", op_1_1_out, &sub_out_1);
  status = sub_graph_1->setParallelType(dag::kParallelTypePipeline);

  {
    dag::Graph *sep_graph = new dag::Graph("sep_graph", &sub_in_0, &sub_out_1);
    sep_graph->addNode(sub_graph_0);
    dag::Node *op_link =
        sep_graph->createNode<OpNode>("op_link", &sub_out_0, &sub_in_1);
    sep_graph->addNode(sub_graph_1);

    status = sep_graph->setParallelType(dag::kParallelTypePipeline);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("graph setParallelType failed");
      return -1;
    }

    sep_graph->init();

    sep_graph->run();
    NNDEPLOY_LOGE("sep_graph->run();\n");

    sep_graph->dump();

    int count = 2;
    for (int i = 0; i < count; ++i) {
      device::Device *device = device::getDefaultHostDevice();
      device::TensorDesc desc;
      desc.data_type_ = base::dataTypeOf<float>();
      desc.data_format_ = base::DataFormat::kDataFormatNCHW;
      desc.shape_ = {1, 3, 512, 512};
      device::Tensor *input_tensor =
          new device::Tensor(device, desc, "sub_in_0 ");

      sub_in_0.set(input_tensor, i, false);

      NNDEPLOY_LOGE("sub_in_0.set(input_tensor, i, false);\n");
    }

    for (int i = 0; i < count; ++i) {
      device::Tensor *result = sub_out_1.getGraphOutputTensor();
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }

      NNDEPLOY_LOGE(
          "device::Tensor *result[%p] = sub_out_1.getGraphOutputTensor();\n",
          result);
    }

    // 有向无环图graph反初始化
    status = sep_graph->deinit();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("graph deinit failed");
      return -1;
    }

    delete sep_graph;
  }

  //{
  //  dag::Graph *par_graph = new dag::Graph("par_graph", {&sub_in_0,
  //  &sub_in_1},
  //                                         {&sub_out_0, &sub_out_1});
  //  par_graph->addNode(sub_graph_0);
  //  par_graph->addNode(sub_graph_1);

  //  par_graph->init();

  //  par_graph->dump();

  //   delete par_graph;
  //}

  // 有向无环图graph销毁
  delete sub_graph_0;
  delete sub_graph_1;

  NNDEPLOY_LOGE("hello world!\n");

  return 0;
}