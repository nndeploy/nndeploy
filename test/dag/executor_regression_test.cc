#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/executor/sequential_executor.h"
#include "nndeploy/dag/executor/parallel_task_executor.h"
#include "nndeploy/dag/executor/parallel_pipeline_executor.h"
#include "nndeploy/dag/executor/parallel_pipeline_condition_executor.h"
#include "nndeploy/net/runtime/pipeline_runtime.h"
#include "nndeploy/net/net.h"
#include "nndeploy/op/op_relu.h"

using namespace nndeploy;
using namespace std::chrono_literals;
namespace {
struct Probe : dag::Node {
  using Node::Node;
  std::atomic<int> calls{0};
  bool fail = false;
  base::Status run() override {
    ++calls;
    return fail ? base::kStatusCodeErrorDag : base::kStatusCodeOk;
  }
};
struct GraphFixture {
  Probe node{"probe"};
  dag::NodeWrapper wrapper{};
  std::vector<dag::NodeWrapper *> nodes{&wrapper};
  std::vector<dag::EdgeWrapper *> edges;
  GraphFixture() { wrapper.node_ = &node; }
};

TEST(ExecutorRegression, CyclesAndRepeatedSort) {
  dag::NodeWrapper a{}, b{}, c{};
  a.successors_ = {&b}; b.predecessors_ = {&a, &c};
  b.successors_ = {&c}; c.predecessors_ = {&b}; c.successors_ = {&b};
  std::vector<dag::NodeWrapper *> nodes{&a, &b, &c}, sorted;
  EXPECT_FALSE(dag::topoSortDFS(nodes, sorted));
  c.successors_.clear(); b.predecessors_ = {&a};
  ASSERT_TRUE(dag::topoSortDFS(nodes, sorted));
  EXPECT_EQ(sorted.size(), 3);
  ASSERT_TRUE(dag::topoSortDFS(nodes, sorted));
  EXPECT_EQ(sorted.size(), 3);
  net::OpWrapper x{}, y{}, z{};
  x.successors_ = {&y}; y.predecessors_ = {&x, &z};
  y.successors_ = {&z}; z.predecessors_ = {&y}; z.successors_ = {&y};
  std::vector<net::OpWrapper *> ops{&x, &y, &z}, order;
  EXPECT_FALSE(net::topoSortDFS(ops, order));
}

TEST(ExecutorRegression, SharedStreamGraphCanRestart) {
  dag::Edge output("output");
  Probe node("node"); node.setOutput(&output);
  dag::Graph graph("graph"); graph.addNode(&node);
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(graph.init()); ASSERT_TRUE(graph.run()); ASSERT_TRUE(graph.deinit());
  }
  EXPECT_EQ(node.calls, 2);
}

TEST(ExecutorRegression, SequentialFailureClearsRunning) {
  GraphFixture f; f.node.fail = true;
  dag::SequentialExecutor executor;
  ASSERT_TRUE(executor.init(f.edges, f.nodes));
  EXPECT_FALSE(executor.run()); EXPECT_FALSE(f.node.isRunning());
  EXPECT_TRUE(executor.deinit());
}

TEST(ExecutorRegression, TaskCancellationIsNotSuccess) {
  GraphFixture f; dag::ParallelTaskExecutor executor;
  ASSERT_TRUE(executor.init(f.edges, f.nodes));
  ASSERT_TRUE(executor.interrupt());
  EXPECT_EQ(executor.run(), base::kStatusCodeNodeInterrupt);
  EXPECT_EQ(f.node.calls, 0);
  EXPECT_TRUE(executor.deinit());
}

TEST(ExecutorRegression, PipelineSourceWaitsForSubmissionAndFailureCloses) {
  GraphFixture f; dag::ParallelPipelineExecutor executor;
  ASSERT_TRUE(executor.init(f.edges, f.nodes));
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(f.node.calls, 0);
  f.node.fail = true;
  ASSERT_TRUE(executor.run());
  EXPECT_FALSE(executor.synchronize());
  EXPECT_FALSE(f.node.isRunning());
  EXPECT_FALSE(executor.deinit());
  executor.deinit(); // Repeated cleanup must be safe, including after failure.
}

TEST(ExecutorRegression, PipelineDroppedFramesSettle) {
  GraphFixture f; dag::Edge input("input");
  f.node.setInput(&input); input.setParallelType(base::kParallelTypePipeline);
  std::vector<dag::Node *> consumers{&f.node}; input.increaseConsumers(consumers);
  input.construct(); input.setQueueMaxSize(1);
  input.setQueueOverflowPolicy(base::kQueueOverflowPolicyDropOldest, 1);
  int a = 1, b = 2; ASSERT_TRUE(input.set(&a, true)); ASSERT_TRUE(input.set(&b, true));
  dag::EdgeWrapper edge{}; edge.edge_ = &input; f.edges.push_back(&edge);
  dag::ParallelPipelineExecutor executor;
  ASSERT_TRUE(executor.init(f.edges, f.nodes));
  ASSERT_TRUE(executor.run()); ASSERT_TRUE(executor.run());
  EXPECT_TRUE(executor.synchronize()); EXPECT_EQ(f.node.calls, 1);
  EXPECT_TRUE(executor.deinit());
}

TEST(ExecutorRegression, DroppedFrameAlignsJoinAndPropagatesToSuccessor) {
  dag::Edge left("left"), right("right"), middle("middle");
  struct Join : Probe {
    using Probe::Probe;
    int value = 0;
    base::Status run() override {
      ++calls;
      auto inputs = getAllInput();
      value = *inputs[0]->get<int>(this) + *inputs[1]->get<int>(this);
      return getAllOutput()[0]->set(new int(value), false);
    }
  } join("join");
  Probe sink("sink");
  ASSERT_TRUE(join.setInput(&left)); ASSERT_TRUE(join.setInput(&right)); ASSERT_TRUE(join.setOutput(&middle));
  sink.setInput(&middle);
  for (auto *edge : {&left, &right, &middle}) edge->setParallelType(base::kParallelTypePipeline);
  std::vector<dag::Node *> joins{&join}, sinks{&sink};
  left.increaseConsumers(joins); right.increaseConsumers(joins); middle.increaseConsumers(sinks);
  left.construct(); right.construct(); middle.construct();
  left.setQueueMaxSize(1); left.setQueueOverflowPolicy(base::kQueueOverflowPolicyDropOldest);
  int a=1, b=2, c=10, d=20;
  left.set(&a, true); left.set(&b, true); right.set(&c, true); right.set(&d, true);
  dag::NodeWrapper x{}, y{}; x.node_=&join; y.node_=&sink; x.successors_={&y}; y.predecessors_={&x};
  dag::EdgeWrapper l{}, r{}, m{}; l.edge_=&left; r.edge_=&right; m.edge_=&middle;
  std::vector<dag::NodeWrapper *> nodes{&x,&y}; std::vector<dag::EdgeWrapper *> edges{&l,&r,&m};
  dag::ParallelPipelineExecutor executor;
  ASSERT_TRUE(executor.init(edges,nodes)); executor.run(); executor.run();
  ASSERT_TRUE(executor.synchronize());
  EXPECT_EQ(join.calls,1); EXPECT_EQ(sink.calls,1); EXPECT_EQ(join.value,22);
  EXPECT_TRUE(executor.deinit());
}

TEST(ExecutorRegression, PipelineCanRestartAndDrainAfterEarlierSync) {
  dag::Edge input("input"); Probe node("node"); node.setInput(&input);
  dag::Graph graph("graph"); graph.setParallelType(base::kParallelTypePipeline); graph.addNode(&node);
  for (int round=0; round<2; ++round) {
    ASSERT_TRUE(graph.init()); ASSERT_TRUE(graph.synchronize());
    int value=1;
    ASSERT_TRUE(input.set(&value,true)); ASSERT_TRUE(graph.run());
    ASSERT_TRUE(input.set(&value,true)); ASSERT_TRUE(graph.run());
    ASSERT_TRUE(graph.deinit()); ASSERT_TRUE(graph.deinit());
  }
  EXPECT_EQ(node.calls,4);
}

TEST(ExecutorRegression, FailedInitializationCanBeCleanedUp) {
  struct FailInit : Probe {
    using Probe::Probe;
    base::Status init() override { return base::kStatusCodeErrorDag; }
  } node("fail");
  dag::NodeWrapper wrapper{}; wrapper.node_=&node;
  std::vector<dag::NodeWrapper *> nodes{&wrapper}; std::vector<dag::EdgeWrapper *> edges;
  dag::ParallelTaskExecutor task;
  EXPECT_FALSE(task.init(edges,nodes)); task.deinit();
  dag::ParallelPipelineExecutor pipeline;
  EXPECT_FALSE(pipeline.init(edges,nodes)); pipeline.deinit(); pipeline.deinit();
  dag::ParallelPipelineConditionExecutor condition;
  EXPECT_FALSE(condition.init(edges,nodes)); condition.deinit();
}

TEST(ExecutorRegression, ConditionCapturesSelectionAndWaitsForWork) {
  struct Block : Probe {
    using Probe::Probe;
    std::shared_future<void> release;
    base::Status run() override { ++calls; release.wait(); return base::kStatusCodeOk; }
  } a("a");
  Probe b("b"); std::promise<void> gate; a.release = gate.get_future().share();
  dag::NodeWrapper x{}, y{}; x.node_ = &a; y.node_ = &b;
  std::vector<dag::NodeWrapper *> nodes{&x, &y}; std::vector<dag::EdgeWrapper *> edges;
  dag::ParallelPipelineConditionExecutor executor;
  ASSERT_TRUE(executor.init(edges, nodes)); executor.select(0);
  ASSERT_TRUE(executor.run()); ASSERT_TRUE(executor.run()); ASSERT_TRUE(executor.run());
  executor.select(1);
  auto sync = std::async(std::launch::async, [&] { return executor.synchronize(); });
  EXPECT_EQ(sync.wait_for(30ms), std::future_status::timeout);
  gate.set_value(); EXPECT_TRUE(sync.get());
  EXPECT_EQ(a.calls, 3); EXPECT_EQ(b.calls, 0); EXPECT_TRUE(executor.deinit());
}

TEST(ExecutorRegression, ConditionDeinitUnblocksMissingInput) {
  dag::Edge input("input");
  GraphFixture f;
  ASSERT_TRUE(f.node.setInput(&input));
  input.setParallelType(base::kParallelTypePipeline);
  std::vector<dag::Node *> consumers{&f.node};
  input.increaseConsumers(consumers);
  input.construct();
  dag::EdgeWrapper edge{};
  edge.edge_ = &input;
  f.edges.push_back(&edge);
  dag::ParallelPipelineConditionExecutor executor;
  ASSERT_TRUE(executor.init(f.edges, f.nodes));
  executor.select(0);
  ASSERT_TRUE(executor.run());

  auto close = std::async(std::launch::async, [&] { return executor.deinit(); });
  const auto state = close.wait_for(2s);
  // Release a regressed implementation so this test fails without hanging.
  if (state == std::future_status::timeout) executor.interrupt();
  close.get();
  EXPECT_EQ(state, std::future_status::ready);
  EXPECT_EQ(f.node.calls, 0);
  EXPECT_FALSE(f.node.getInitialized());
  EXPECT_FALSE(executor.run());
}

TEST(ExecutorRegression, ConditionDeinitUnblocksFullOutput) {
  dag::Edge output("output");
  struct Source : Probe {
    using Probe::Probe;
    int value = 42;
    base::Status run() override {
      ++calls;
      return getOutput(0)->set(&value, true);
    }
  } node("source", {}, {&output});
  output.setParallelType(base::kParallelTypePipeline);
  std::vector<dag::Node *> consumers{nullptr}, producers{&node};
  output.increaseConsumers(consumers);
  output.increaseProducers(producers);
  output.construct();
  output.setQueueMaxSize(1);
  output.setQueueOverflowPolicy(base::kQueueOverflowPolicyAllBackpressure);
  dag::NodeWrapper wrapper{};
  wrapper.node_ = &node;
  dag::EdgeWrapper edge{};
  edge.edge_ = &output;
  std::vector<dag::NodeWrapper *> nodes{&wrapper};
  std::vector<dag::EdgeWrapper *> edges{&edge};
  dag::ParallelPipelineConditionExecutor executor;
  ASSERT_TRUE(executor.init(edges, nodes));
  executor.select(0);
  ASSERT_TRUE(executor.run());
  ASSERT_TRUE(executor.synchronize());
  ASSERT_TRUE(executor.run());

  auto close = std::async(std::launch::async, [&] { return executor.deinit(); });
  const auto state = close.wait_for(2s);
  if (state == std::future_status::timeout) executor.interrupt();
  close.get();
  EXPECT_EQ(state, std::future_status::ready);
  EXPECT_EQ(node.calls, 2);
  EXPECT_FALSE(node.getInitialized());
  EXPECT_FALSE(node.isRunning());
}

struct Relu : op::OpRelu {
  int calls = 0; bool fail = false, dynamic = false, fail_init = false;
  base::Status init() override {
    return fail_init ? base::Status(base::kStatusCodeErrorDag) : OpRelu::init();
  }
  uint64_t getWorkspaceSize() override { return dynamic && calls == 0 ? 32 : 0; }
  base::Status run() override {
    ++calls;
    return fail ? base::kStatusCodeErrorDag : OpRelu::run();
  }
};
struct NetFixture {
  base::DeviceType cpu{base::kDeviceTypeCodeCpu};
  device::TensorDesc desc{base::dataTypeOf<float>(), base::kDataFormatN, {1}};
  device::Tensor input{desc, "x"}, output{desc, "y"};
  Relu relu;
  net::OpWrapper op{};
  net::TensorWrapper x{}, y{};
  std::vector<net::OpWrapper *> ops{&op};
  std::vector<net::TensorWrapper *> tensors{&x, &y};
  std::vector<device::Tensor *> inputs{&input}, outputs{&output};
  NetFixture() {
    relu.setName("relu"); relu.setOpType(ir::kOpTypeRelu);
    relu.setInput(&input, 0); relu.setOutput(&output, 0); op.op_ = &relu; op.name_ = "relu";
    x.name_ = "x"; x.tensor_ = &input; x.consumers_ = {&op}; x.input_output_type_ = net::kInput;
    y.name_ = "y"; y.tensor_ = &output; y.producers_ = {&op}; y.input_output_type_ = net::kOutput;
  }
  base::Status init(net::Runtime &runtime, bool dynamic = false) {
    return runtime.init(tensors, ops, inputs, outputs, dynamic, {{"x", {1}}});
  }
};

TEST(ExecutorRegression, DynamicWorkspaceCanShrinkToZero) {
  NetFixture f; f.relu.dynamic = true; net::SequentialRuntime runtime(f.cpu);
  ASSERT_TRUE(f.init(runtime, true));
  for (int i = 0; i < 3; ++i) ASSERT_TRUE(runtime.run());
  EXPECT_TRUE(runtime.deinit());
}

TEST(ExecutorRegression, PipelineOwnsEachInputAndOutputFrame) {
  NetFixture f; net::PipelineRuntime runtime(f.cpu); ASSERT_TRUE(f.init(runtime));
  device::Tensor input(device::getDevice(f.cpu), f.desc, "x");
  for (int value : {1, 2}) {
    *static_cast<float *>(input.getData()) = value;
    ASSERT_TRUE(runtime.copyToInputTensor(&input)); ASSERT_TRUE(runtime.run());
  }
  // Mutating the caller's buffer must not mutate either submitted frame.
  *static_cast<float *>(input.getData()) = 99;
  ASSERT_TRUE(runtime.synchronize());
  for (int value : {1, 2}) {
    std::unique_ptr<device::Tensor> result(runtime.getOutputTensorAfterRun("y", f.cpu, true, base::kDataFormatN));
    ASSERT_NE(result, nullptr); EXPECT_FLOAT_EQ(*static_cast<float *>(result->getData()), value);
  }
  EXPECT_TRUE(runtime.deinit());
}

TEST(ExecutorRegression, PipelineRejectsBadInputAndUnsupportedInterfaces) {
  NetFixture f; net::PipelineRuntime runtime(f.cpu); ASSERT_TRUE(f.init(runtime));
  device::TensorDesc wrong(base::dataTypeOf<int32_t>(), base::kDataFormatN, {1});
  device::Tensor input(device::getDevice(f.cpu), wrong, "x");
  EXPECT_FALSE(runtime.copyToInputTensor(&input));
  base::ShapeMap shape{{"x", {2}}};
  EXPECT_EQ(runtime.reshape(shape), base::kStatusCodeErrorNotSupport);
  EXPECT_EQ(runtime.getMemorySize(), -1);
  EXPECT_EQ(runtime.setMemory(nullptr), base::kStatusCodeErrorNotSupport);
  EXPECT_TRUE(runtime.deinit());
}

TEST(ExecutorRegression, PipelineFailureReachesSynchronizeAndDeinit) {
  NetFixture f; f.relu.fail = true; net::PipelineRuntime runtime(f.cpu);
  ASSERT_TRUE(f.init(runtime));
  device::Tensor input(device::getDevice(f.cpu), f.desc, "x");
  ASSERT_TRUE(runtime.copyToInputTensor(&input)); ASSERT_TRUE(runtime.run());
  EXPECT_FALSE(runtime.synchronize());
  EXPECT_EQ(runtime.getOutputTensorAfterRun("y", f.cpu, true, base::kDataFormatN), nullptr);
  EXPECT_FALSE(runtime.deinit()); runtime.deinit();
}

TEST(ExecutorRegression, PipelineStageInitFailureRollsBackAndCanRetry) {
  NetFixture f; f.relu.fail_init=true; net::PipelineRuntime runtime(f.cpu);
  EXPECT_FALSE(f.init(runtime));
  EXPECT_EQ(f.relu.getInput(),&f.input);
  f.relu.fail_init=false;
  ASSERT_TRUE(f.init(runtime));
  EXPECT_TRUE(runtime.deinit());
  EXPECT_EQ(f.relu.getInput(),&f.input);
}

TEST(ExecutorRegression, FailingStageWakesDependentStage) {
  NetFixture f; f.relu.fail=true;
  device::Tensor output(f.desc,"z");
  Relu second; second.setName("second"); second.setOpType(ir::kOpTypeRelu);
  second.setInput(&f.output,0); second.setOutput(&output,0);
  net::OpWrapper op{}; op.op_=&second; op.name_="second";
  net::TensorWrapper z{}; z.name_="z"; z.tensor_=&output; z.producers_={&op}; z.input_output_type_=net::kOutput;
  f.y.consumers_={&op}; f.y.input_output_type_=net::kNone;
  f.op.successors_={&op}; op.predecessors_={&f.op};
  f.ops.push_back(&op); f.tensors.push_back(&z); f.outputs={&output};
  net::PipelineRuntime runtime(f.cpu); runtime.setWorkers(2,{f.cpu,f.cpu});
  ASSERT_TRUE(f.init(runtime));
  device::Tensor input(device::getDevice(f.cpu),f.desc,"x");
  ASSERT_TRUE(runtime.copyToInputTensor(&input)); ASSERT_TRUE(runtime.run());
  EXPECT_FALSE(runtime.synchronize()); EXPECT_FALSE(runtime.deinit());
  EXPECT_EQ(second.calls,0);
}

TEST(ExecutorRegression, PipelineQueuesReleaseHistoryAfterAllConsumers) {
  net::PipelineTensor queue;
  net::SequentialRuntime first(base::kDeviceTypeCodeCpu), second(base::kDeviceTypeCodeCpu);
  queue.current_index_[&first]=0; queue.current_index_[&second]=0;
  auto frame=std::make_shared<device::Tensor>("frame");
  std::weak_ptr<device::Tensor> lifetime=frame;
  queue.push(frame); frame.reset();
  auto lease=queue.popShared(&first);
  EXPECT_EQ(queue.tensors_.size(),1);
  auto other=queue.popShared(&second);
  EXPECT_TRUE(queue.tensors_.empty()); EXPECT_FALSE(lifetime.expired());
  lease.reset(); other.reset(); EXPECT_TRUE(lifetime.expired());
}

TEST(ExecutorRegression, PipelineValidatesDeviceList) {
  NetFixture f;
  net::PipelineRuntime runtime(f.cpu);
  runtime.setWorkers(0);
  EXPECT_FALSE(f.init(runtime));
}

TEST(ExecutorRegression, InvalidReshapePreservesExistingBuffer) {
  NetFixture f; net::SequentialRuntime runtime(f.cpu); ASSERT_TRUE(f.init(runtime,true));
  auto *buffer=f.input.getBuffer();
  base::ShapeMap invalid{{"x",{}}}; EXPECT_FALSE(runtime.reshape(invalid));
  EXPECT_EQ(f.input.getBuffer(),buffer); EXPECT_EQ(f.input.getShape(),base::IntVector({1}));
  EXPECT_TRUE(runtime.deinit());
}

TEST(ExecutorRegression, MultiStagePipelinePreservesIntermediateOutputsAndFanout) {
  for (int workers : {1,2,3}) {
    ir::ModelDesc model;
    auto value=[](const char *name) { return std::make_shared<ir::ValueDesc>(name,base::dataTypeOf<float>(),base::IntVector{1}); };
    model.inputs_={value("x")}; model.outputs_={value("mid"),value("y"),value("other")};
    for (auto names : {std::vector<std::string>{"a","x","mid"}, {"b","mid","y"}, {"c","mid","other"}}) {
      std::vector<std::string> inputs{names[1]},outputs{names[2]};
      model.op_descs_.push_back(std::make_shared<ir::OpDesc>(names[0],ir::kOpTypeRelu,inputs,outputs));
    }
    net::Net net; net.setDeviceType(base::kDeviceTypeCodeCpu); net.setModelDesc(&model); net.enableOpt(false);
    net.setParallelType(base::kParallelTypePipeline);
    net.setWorkers(workers,std::vector<base::DeviceType>(workers,base::kDeviceTypeCodeCpu));
    if (workers > 1) {
      net::Net invalid; invalid.setDeviceType(base::kDeviceTypeCodeCpu); invalid.setModelDesc(&model); invalid.enableOpt(false);
      invalid.setParallelType(base::kParallelTypePipeline);
      invalid.setWorkers(workers,{base::kDeviceTypeCodeCpu});
      EXPECT_EQ(invalid.init(),base::kStatusCodeErrorInvalidParam);
      invalid.deinit();
    }
    ASSERT_TRUE(net.init());
    device::TensorDesc desc(base::dataTypeOf<float>(),base::kDataFormatN,{1});
    device::Tensor input(device::getDevice(base::kDeviceTypeCodeCpu),desc,"x");
    for (int v : {3,7}) {
      *static_cast<float *>(input.getData())=v;
      ASSERT_TRUE(net.copyToInputTensor(&input)); ASSERT_TRUE(net.preRun()); ASSERT_TRUE(net.run()); ASSERT_TRUE(net.postRun());
    }
    ASSERT_TRUE(net.synchronize());
    for (int v : {3,7}) for (const char *name : {"mid","y","other"}) {
      std::unique_ptr<device::Tensor> result(net.getOutputTensorAfterRun(name,base::kDeviceTypeCodeCpu,true,base::kDataFormatN));
      ASSERT_NE(result,nullptr); EXPECT_FLOAT_EQ(*static_cast<float *>(result->getData()),v);
    }
    EXPECT_TRUE(net.deinit());
  }
}
} // namespace
