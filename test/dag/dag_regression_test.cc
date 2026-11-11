#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include "nndeploy/dag/edge/data_packet.h"
#include "nndeploy/dag/edge/fixed_edge.h"
#include "nndeploy/dag/edge/pipeline_edge.h"
#include "nndeploy/dag/executor/parallel_task_executor.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/device/cpu/cpu_device.h"

namespace nndeploy {
namespace dag {
namespace {

class ProbeNode : public Node {
 public:
  using Node::Node;
  base::Status result;
  int calls = 0;
  base::Status run() override { ++calls; return result; }
};

TEST(DagRegression, FailureDrainsWorkersAndAllowsRetry) {
  ProbeNode first("first"), last("last");
  NodeWrapper a, b;
  a.node_ = &first;
  b.node_ = &last;
  a.successors_ = {&b};
  b.predecessors_ = {&a};
  std::vector<NodeWrapper *> nodes{&a, &b};
  std::vector<EdgeWrapper *> edges;
  ParallelTaskExecutor executor;
  ASSERT_TRUE(executor.init(edges, nodes));
  first.result = base::kStatusCodeErrorInvalidParam;
  EXPECT_EQ(executor.run(), base::kStatusCodeErrorInvalidParam);
  EXPECT_EQ(last.calls, 0);
  first.result = base::kStatusCodeOk;
  EXPECT_TRUE(executor.run());
  EXPECT_EQ(first.calls, 2);
  EXPECT_EQ(last.calls, 1);
  EXPECT_TRUE(executor.deinit());
}

TEST(DagRegression, RemoveNodeFromBothSidesOfTopology) {
  ProbeNode first("first"), middle("middle"), last("last");
  Graph graph("chain");
  ASSERT_TRUE(graph.addNode(&first));
  ASSERT_TRUE(graph.addNode(&middle));
  ASSERT_TRUE(graph.addNode(&last));
  auto *a = graph.getNodeWrapper(&first);
  auto *b = graph.getNodeWrapper(&middle);
  auto *c = graph.getNodeWrapper(&last);
  a->successors_ = {b};
  b->predecessors_ = {a};
  b->successors_ = {c};
  c->predecessors_ = {b};
  EXPECT_TRUE(graph.deleteNode(&middle));
  EXPECT_TRUE(a->successors_.empty());
  EXPECT_TRUE(c->predecessors_.empty());
  EXPECT_EQ(graph.getNodeCount(), 2);
}

class CountingCpu : public device::CpuDevice {
 public:
  CountingCpu() : CpuDevice(base::kDeviceTypeCodeCpu, "") {}
  int live = 0;
  void *allocate(const device::BufferDesc &desc) override {
    ++live;
    return CpuDevice::allocate(desc);
  }
  void deallocate(void *ptr) override {
    --live;
    CpuDevice::deallocate(ptr);
  }
};

TEST(DagRegression, PacketReleasesAllocatedDeviceMemory) {
  CountingCpu cpu;
  {
    DataPacket packet;
    ASSERT_NE(packet.create(&cpu, device::BufferDesc(16)), nullptr);
    ASSERT_NE(packet.create(&cpu, device::BufferDesc(32)), nullptr);
    EXPECT_EQ(cpu.live, 1);
  }
  EXPECT_EQ(cpu.live, 0);
}
TEST(DagRegression, CancelledNativeSubmissionKeepsCallerOwnership) {
  using namespace std::chrono_literals;
  CountingCpu cpu;
  Edge edge("bounded");
  ASSERT_TRUE(edge.setParallelType(base::kParallelTypePipeline));
  ASSERT_TRUE(edge.setQueueMaxSize(1));
  ASSERT_TRUE(edge.setQueueOverflowPolicy(base::kQueueOverflowPolicyAllBackpressure));
  ASSERT_TRUE(edge.markGraphOutput());
  auto first = std::make_unique<device::Buffer>(&cpu, 4);
  ASSERT_TRUE(edge.set(first.get(), false));
  first.release();
  auto rejected = std::make_unique<device::Buffer>(&cpu, 4);
  auto writer = std::async(std::launch::async, [&] { return edge.set(rejected.get(), false); });
  EXPECT_EQ(writer.wait_for(20ms), std::future_status::timeout);
  EXPECT_TRUE(edge.requestTerminate());
  ASSERT_EQ(writer.wait_for(1s), std::future_status::ready);
  EXPECT_FALSE(writer.get());
  EXPECT_EQ(cpu.live, 2);
  rejected.reset();
  EXPECT_EQ(cpu.live, 1);
}

TEST(DagRegression, NativeTypedProbePreservesMismatchedPacket) {
  CountingCpu cpu;
  Edge edge("mixed");
  ASSERT_TRUE(edge.setParallelType(base::kParallelTypePipeline));
  ASSERT_TRUE(edge.markGraphOutput());
  auto buffer = std::make_unique<device::Buffer>(&cpu, 4);
  auto *ptr = buffer.get();
  ASSERT_TRUE(edge.set(ptr, false));
  buffer.release();
  EXPECT_EQ(edge.getGraphOutputTensor(), nullptr);
  EXPECT_EQ(edge.getGraphOutputBuffer(), ptr);
}

TEST(DagRegression, AcquiredPacketKeepsOldNativeOutputAlive) {
  CountingCpu cpu;
  Edge edge("fixed");
  auto *buffer = edge.create(&cpu, device::BufferDesc(4));
  auto old = edge.acquirePacket(nullptr, false);
  ASSERT_TRUE(edge.notifyWritten(buffer));
  ASSERT_NE(edge.create(&cpu, device::BufferDesc(4)), buffer);
  EXPECT_EQ(old->getBuffer(), buffer);
  EXPECT_EQ(cpu.live, 2);
  old.reset();
  EXPECT_EQ(cpu.live, 1);
}


struct CountedPayload {
  explicit CountedPayload(std::atomic<int> *destroyed) : destroyed(destroyed) {}
  ~CountedPayload() { ++*destroyed; }
  std::atomic<int> *destroyed;
  int value = 0;
};

TEST(DagRegression, DropOldestWaitsForUnfinishedProducer) {
  using namespace std::chrono_literals;
  std::atomic<int> destroyed{0};
  {
    Edge edge("writing");
    edge.setParallelType(base::kParallelTypePipeline);
    edge.setQueueMaxSize(1);
    edge.setQueueOverflowPolicy(base::kQueueOverflowPolicyDropOldest);
    edge.markGraphOutput();
    auto *first = edge.create<CountedPayload>(&destroyed);
    auto writer = std::async(std::launch::async, [&] {
      return edge.create<CountedPayload>(&destroyed);
    });
    EXPECT_EQ(writer.wait_for(50ms), std::future_status::timeout);
    EXPECT_EQ(destroyed.load(), 0);
    // Avoid dereferencing an already freed object when testing the old code.
    if (destroyed.load() == 0) first->value = 42;
    EXPECT_TRUE(edge.notifyWritten(first));
    if (writer.wait_for(1s) != std::future_status::ready) edge.requestTerminate();
    auto *second = writer.get();
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(destroyed.load(), 1);
    second->value = 7;
    ASSERT_TRUE(edge.notifyWritten(second));
    EXPECT_EQ(edge.getGraphOutput<CountedPayload>()->value, 7);
  }
  EXPECT_EQ(destroyed.load(), 2);
}

TEST(DagRegression, CancelKeepsUnfinishedProducerAlive) {
  using namespace std::chrono_literals;
  std::atomic<int> destroyed{0};
  {
    Edge edge("cancel_writing");
    edge.setParallelType(base::kParallelTypePipeline);
    edge.setQueueMaxSize(1);
    edge.setQueueOverflowPolicy(base::kQueueOverflowPolicyDropOldest);
    edge.markGraphOutput();
    auto *first = edge.create<CountedPayload>(&destroyed);
    auto writer = std::async(std::launch::async, [&] {
      return edge.create<CountedPayload>(&destroyed);
    });
    EXPECT_EQ(writer.wait_for(50ms), std::future_status::timeout);
    edge.requestTerminate();
    ASSERT_EQ(writer.wait_for(1s), std::future_status::ready);
    auto *rejected = writer.get();
    EXPECT_EQ(rejected, nullptr);
    // Only the rejected allocation is released; the first writer can finish.
    EXPECT_EQ(destroyed.load(), 1);
    if (!rejected) first->value = 42;
    EXPECT_TRUE(edge.notifyWritten(first));
    EXPECT_EQ(edge.getGraphOutput<CountedPayload>(), nullptr);
  }
  EXPECT_EQ(destroyed.load(), 2);
}

TEST(DagRegression, SkippingUnwrittenPacketDoesNotStrandBackpressure) {
  using namespace std::chrono_literals;
  std::atomic<int> destroyed{0};
  Edge edge("skip_writing");
  edge.setParallelType(base::kParallelTypePipeline);
  edge.setQueueMaxSize(1);
  edge.setQueueOverflowPolicy(base::kQueueOverflowPolicyAllBackpressure);
  edge.markGraphOutput();
  auto *first = edge.create<CountedPayload>(&destroyed);
  ASSERT_EQ(edge.update(nullptr), base::kEdgeUpdateFlagComplete);
  int next = 1;
  auto writer = std::async(std::launch::async, [&] { return edge.set(&next, true); });
  auto reader = std::async(std::launch::async, [&] { return edge.update(nullptr); });
  EXPECT_EQ(writer.wait_for(50ms), std::future_status::timeout);
  EXPECT_EQ(reader.wait_for(50ms), std::future_status::timeout);
  ASSERT_EQ(destroyed.load(), 0);
  first->value = 42;
  EXPECT_TRUE(edge.notifyWritten(first));
  bool completed = writer.wait_for(1s) == std::future_status::ready;
  if (!completed) edge.requestTerminate();
  EXPECT_TRUE(completed);
  EXPECT_TRUE(writer.get());
  EXPECT_EQ(reader.get(), base::kEdgeUpdateFlagComplete);
}

TEST(DagRegression, RequeuedPayloadKeepsOwnerForAllConsumers) {
  std::atomic<int> destroyed{0};
  {
    Edge edge("fanout");
    edge.setParallelType(base::kParallelTypePipeline);
    edge.markGraphOutput();
    ProbeNode consumer("consumer");
    std::vector<Node *> consumers{&consumer};
    edge.increaseConsumers(consumers);
    edge.construct();
    auto *payload = new CountedPayload(&destroyed);
    ASSERT_TRUE(edge.set(payload, false));
    EXPECT_EQ(edge.getGraphOutput<CountedPayload>(), payload);
    ASSERT_EQ(edge.update(&consumer), base::kEdgeUpdateFlagComplete);
    EXPECT_EQ(edge.get<CountedPayload>(&consumer), payload);
    ASSERT_TRUE(edge.set(payload, true));
    EXPECT_EQ(edge.getGraphOutput<CountedPayload>(), payload);
    ASSERT_EQ(edge.update(&consumer), base::kEdgeUpdateFlagComplete);
    EXPECT_EQ(edge.get<CountedPayload>(&consumer), payload);
    EXPECT_EQ(destroyed.load(), 0);
  }
  EXPECT_EQ(destroyed.load(), 1);
}

TEST(DagRegression, RequeuedPayloadSurvivesDropAndBackpressure) {
  using namespace std::chrono_literals;
  for (auto policy : {base::kQueueOverflowPolicyDropOldest,
                      base::kQueueOverflowPolicyAllBackpressure}) {
    std::atomic<int> destroyed{0};
    {
      Edge edge("requeue");
      edge.setParallelType(base::kParallelTypePipeline);
      edge.setQueueMaxSize(1);
      edge.setQueueOverflowPolicy(policy);
      edge.markGraphOutput();
      auto *payload = new CountedPayload(&destroyed);
      ASSERT_TRUE(edge.set(payload, false));
      if (policy == base::kQueueOverflowPolicyDropOldest) {
        ASSERT_TRUE(edge.set(payload, true));
      } else {
        EXPECT_EQ(edge.getGraphOutput<CountedPayload>(), payload);
        auto writer = std::async(std::launch::async, [&] { return edge.set(payload, true); });
        EXPECT_EQ(writer.wait_for(50ms), std::future_status::timeout);
        EXPECT_EQ(edge.getGraphOutput<CountedPayload>(), payload);
        EXPECT_TRUE(writer.get());
      }
      EXPECT_EQ(destroyed.load(), 0);
      edge.requestTerminate();
    }
    EXPECT_EQ(destroyed.load(), 1);
  }
}

TEST(DagRegression, RejectsNullPacketWithoutReplacingOrEnqueueing) {
  // The child isolates the former FixedEdge null dereference from other tests.
  EXPECT_EXIT({
    FixedEdge edge(base::kParallelTypeSequential);
    int value = 42;
    edge.AbstractEdge::set(&value);
    bool rejected = edge.takeDataPacket(nullptr) == base::kStatusCodeErrorInvalidParam;
    std::_Exit(rejected && edge.AbstractEdge::get<int>(nullptr) == &value ? 0 : 1);
  }, ::testing::ExitedWithCode(0), "");
  PipelineEdge edge(base::kParallelTypePipeline);
  edge.markGraphOutput();
  EXPECT_EQ(edge.takeDataPacket(nullptr), base::kStatusCodeErrorInvalidParam);
  EXPECT_TRUE(edge.empty());
}

}  // namespace
}  // namespace dag
}  // namespace nndeploy
