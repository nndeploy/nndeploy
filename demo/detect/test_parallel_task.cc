#include "nndeploy/base/common.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
using namespace nndeploy;

std::mutex mutex;

class NNDEPLOY_CC_API Test : public dag::Node {
 public:
  Test(const std::string& name, dag::Edge* input, dag::Edge* output)
      : Node(name, input, output) {}

  Test(const std::string& name, std::initializer_list<dag::Edge*> inputs,
       std::initializer_list<dag::Edge*> outputs)
      : Node(name, inputs, outputs) {}

  virtual ~Test() {}

  virtual base::Status run() { 
    std::lock_guard<std::mutex> lock(mutex);
    std::cout << name_ << std::endl; }

};

int main() {
  dag::Edge input("in");
  dag::Edge output("out");
  std::string name = "parellel_task";
  dag::Graph* graph = new dag::Graph(name, &input, &output);
  dag::Edge* A_out = graph->createEdge("A_out");
  dag::Edge* B_out = graph->createEdge("B_out");
  dag::Edge* C_out = graph->createEdge("C_out");
  dag::Edge* D_out = graph->createEdge("D_out");
  dag::Edge* E_out = graph->createEdge("E_out");
  dag::Edge* F_out = graph->createEdge("F_out");
  dag::Edge* G_out = graph->createEdge("G_out");
  dag::Edge* H_out = graph->createEdge("H_out");

  dag::Node* A = graph->createNode<Test>("A", &input, A_out);
  dag::Node* B = graph->createNode<Test>("B", &input, B_out);
  dag::Node* C = graph->createNode<Test>("C", A_out, C_out);
  dag::Node* D = graph->createNode<Test>("D", B_out, D_out);
  dag::Node* E = graph->createNode<Test>("E", {C_out, D_out}, {E_out});
  dag::Node* F = graph->createNode<Test>("F", D_out, F_out);
  dag::Node* G = graph->createNode<Test>("G", E_out, G_out);
  dag::Node* H = graph->createNode<Test>("H", {E_out, F_out}, {H_out});
  dag::Node* out = graph->createNode<Test>("out", {G_out, H_out}, {&output});

//   graph->dump();
  graph->setParallelType(nndeploy::dag::kParallelTypeTask);

  NNDEPLOY_TIME_POINT_START("graph->init()");
  base::Status status = graph->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph init failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->init()");

  //   std::string input_path =
  //       "/data/sjx/code/nndeploy_resource/nndeploy/test_data/detect/sample.jpg";
  //   // opencv读图
  //   cv::Mat input_mat = cv::imread(input_path);
  //   // 将图片写入有向无环图graph输入边
  //   input.set(input_mat, 0);

  // 有向无环图Graphz运行
  NNDEPLOY_TIME_POINT_START("graph->run()");
  status = graph->run();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph run failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->run()");
}