#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <functional>
#include <stdexcept>
#include <thread>

#include "nndeploy/dag/condition.h"
#include "nndeploy/dag/composite_node.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/loop.h"
#include "nndeploy/dag/state.h"

namespace nndeploy {
namespace dag {
namespace {

class StateNode : public Node {
 public:
  StateNode(const std::string& name, std::vector<Edge*> inputs = {},
            std::vector<Edge*> outputs = {}) : Node(name, inputs, outputs) {
    key_ = "test::StateNode";
  }
  std::function<base::Status(StateNode&)> action;
  int calls = 0;
  base::Status run() override {
    ++calls;
    return action ? action(*this) : base::Status::Ok();
  }
};
REGISTER_NODE("test::StateNode", StateNode);

TEST(StateControl, StorageCopiesSlotsAndSharesObjects) {
  State state;
  base::Any value;
  EXPECT_FALSE(state.get("missing", value));
  auto object = std::make_shared<int>(7);
  ASSERT_TRUE(state.set("object", base::Any(object)));
  ASSERT_TRUE(state.get("object", value));
  EXPECT_EQ(base::get<std::shared_ptr<int>>(value), object);
  EXPECT_TRUE(state.erase("object"));
  EXPECT_FALSE(state.contains("object"));
  EXPECT_FALSE(state.erase("object"));
  EXPECT_EQ(*base::get<std::shared_ptr<int>>(value), 7);
}

class OrderedStateTest : public testing::TestWithParam<base::ParallelType> {};
TEST_P(OrderedStateTest, ZeroEdgeNodesShareRequestStateInOrder) {
  StateNode a("a"), b("b"), c("c");
  Graph graph("ordered");
  graph.setParallelType(GetParam());
  ASSERT_TRUE(graph.addNode(&c));
  ASSERT_TRUE(graph.addNode(&a));
  ASSERT_TRUE(graph.addNode(&b));
  ASSERT_TRUE(graph.setExecutionOrder({"a", "b", "c"}));
  a.action = [](StateNode& node) { return node.getState()->set("value", base::Any(1)); };
  b.action = [](StateNode& node) {
    base::Any value;
    EXPECT_TRUE(node.getState()->get("value", value));
    EXPECT_EQ(base::get<int>(value), 1);
    return node.getState()->set("value", base::Any(2));
  };
  c.action = [](StateNode& node) {
    base::Any value;
    EXPECT_TRUE(node.getState()->get("value", value));
    EXPECT_EQ(base::get<int>(value), 2);
    return node.getState()->set("value", base::Any(3));
  };
  ASSERT_TRUE(graph.init());
  EXPECT_FALSE(graph.setExecutionOrder({"c", "b"}));
  for (int i = 0; i < 2; ++i) {
    auto state = std::make_shared<State>();
    ASSERT_TRUE(graph.runWithState(state));
    base::Any value;
    ASSERT_TRUE(state->get("value", value));
    EXPECT_EQ(base::get<int>(value), 3);
    EXPECT_EQ(a.getState(), nullptr);
  }
  ASSERT_TRUE(graph.deinit());
  ASSERT_TRUE(graph.init());
  EXPECT_TRUE(graph.runWithState(std::make_shared<State>()));
  EXPECT_TRUE(graph.deinit());
}
INSTANTIATE_TEST_SUITE_P(Executors, OrderedStateTest,
                        testing::Values(base::kParallelTypeSequential, base::kParallelTypeTask));

TEST(StateControl, MixedDependenciesAndRemovingOneSource) {
  Edge edge("data");
  StateNode a("a", {}, {&edge}), b("b"), c("c", {&edge}, {});
  Graph graph("mixed");
  graph.setParallelType(base::kParallelTypeTask);
  ASSERT_TRUE(graph.addNode(&a));
  ASSERT_TRUE(graph.addNode(&b));
  ASSERT_TRUE(graph.addNode(&c));
  ASSERT_TRUE(graph.addControlDependency("a", "c"));
  ASSERT_TRUE(graph.addControlDependency("b", "c"));
  a.action = [&](StateNode&) { return edge.set<int>(new int(7), false); };
  b.action = [](StateNode& node) { return node.getState()->set("prepared", base::Any(true)); };
  c.action = [&](StateNode& node) {
    base::Any value;
    EXPECT_TRUE(node.getState()->get("prepared", value));
    EXPECT_TRUE(base::get<bool>(value));
    EXPECT_EQ(*edge.get<int>(&node), 7);
    return base::Status::Ok();
  };
  ASSERT_TRUE(graph.init());
  EXPECT_EQ(graph.getNodeWrapper(&c)->predecessors_.size(), 2);
  ASSERT_TRUE(graph.runWithState(std::make_shared<State>()));
  ASSERT_TRUE(graph.deinit());
  ASSERT_TRUE(graph.removeControlDependency("a", "c"));
  ASSERT_TRUE(graph.init());
  EXPECT_EQ(graph.getNodeWrapper(&c)->predecessors_.size(), 2);
  EXPECT_TRUE(graph.runWithState(std::make_shared<State>()));
  EXPECT_TRUE(graph.deinit());
}

TEST(StateControl, SingleNodeAndValidation) {
  StateNode a("a"), b("b"), unused("unused");
  Graph graph("validate");
  graph.addNode(&a); graph.addNode(&b); graph.addNode(&unused);
  EXPECT_FALSE(graph.setExecutionOrder({"a", "a"}));
  EXPECT_FALSE(graph.setExecutionOrder({"missing"}));
  EXPECT_FALSE(graph.addControlDependency("a", "a"));
  EXPECT_FALSE(graph.addControlDependency("a", "missing"));
  ASSERT_TRUE(graph.setExecutionOrder({"a"}));
  ASSERT_TRUE(graph.init());
  ASSERT_TRUE(graph.run());
  EXPECT_EQ(a.calls, 1);
  EXPECT_EQ(unused.calls, 0);
  ASSERT_TRUE(graph.deinit());
  ASSERT_TRUE(graph.setExecutionOrder({"a", "b"}));
  ASSERT_TRUE(graph.addControlDependency("b", "a"));
  EXPECT_FALSE(graph.init());
  ASSERT_TRUE(graph.removeControlDependency("b", "a"));
  EXPECT_TRUE(graph.init());
  EXPECT_TRUE(graph.deinit());
}

TEST(StateControl, JsonRoundTripAndDeletion) {
  StateNode a("a"), b("b"), c("c");
  Graph graph("source");
  graph.addNode(&a); graph.addNode(&b); graph.addNode(&c);
  graph.setExecutionOrder({"a", "b", "c"});
  graph.addControlDependency("a", "c");
  Graph copy("copy");
  ASSERT_TRUE(copy.deserialize(graph.serialize()));
  ASSERT_TRUE(copy.deleteNode(copy.getNode("b")));
  ASSERT_TRUE(copy.init());
  EXPECT_EQ(copy.getNodeWrapper("c")->predecessors_.size(), 1);
  EXPECT_TRUE(copy.run());
  EXPECT_TRUE(copy.deinit());
  Graph roundtrip("roundtrip");
  ASSERT_TRUE(roundtrip.deserialize(copy.serialize()));
  EXPECT_TRUE(roundtrip.init());
  EXPECT_TRUE(roundtrip.run());
  EXPECT_TRUE(roundtrip.deinit());
}

TEST(StateControl, FailureAndExceptionUnbindState) {
  StateNode a("a"), b("b");
  Graph graph("failure");
  graph.addNode(&a); graph.addNode(&b);
  graph.setExecutionOrder({"a", "b"});
  a.action = [](StateNode& node) -> base::Status {
    node.getState()->set("partial", base::Any(42));
    throw std::runtime_error("failed");
  };
  ASSERT_TRUE(graph.init());
  auto state = std::make_shared<State>();
  EXPECT_FALSE(graph.runWithState(state));
  EXPECT_EQ(b.calls, 0);
  EXPECT_EQ(graph.getState(), nullptr);
  EXPECT_EQ(a.getState(), nullptr);
  EXPECT_TRUE(state->contains("partial"));
  a.action = nullptr;
  EXPECT_TRUE(graph.runWithState(std::make_shared<State>()));
  EXPECT_TRUE(graph.deinit());
}

TEST(StateControl, ReentrantAndConcurrentCallsCannotReplaceState) {
  StateNode node("node");
  Graph graph("exclusive");
  graph.addNode(&node); graph.setExecutionOrder({"node"});
  std::promise<void> entered, release;
  auto wait = release.get_future().share();
  node.action = [&](StateNode& n) {
    EXPECT_FALSE(graph.run());
    EXPECT_FALSE(graph.runWithState(std::make_shared<State>()));
    entered.set_value();
    EXPECT_EQ(wait.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    return base::Status::Ok();
  };
  ASSERT_TRUE(graph.init());
  auto state = std::make_shared<State>();
  auto worker = std::async(std::launch::async, [&] { return graph.runWithState(state); });
  EXPECT_EQ(entered.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_FALSE(graph.run());
  EXPECT_FALSE(graph.runWithState(std::make_shared<State>()));
  EXPECT_EQ(graph.getState(), state);
  release.set_value();
  EXPECT_TRUE(worker.get());
  EXPECT_EQ(graph.getState(), nullptr);
  EXPECT_TRUE(graph.deinit());
}

TEST(StateControl, NestedGraphInheritsState) {
  StateNode leaf("leaf");
  Graph child("child"), parent("parent");
  child.addNode(&leaf); child.setExecutionOrder({"leaf"});
  parent.addNode(&child); parent.setExecutionOrder({"child"});
  leaf.action = [](StateNode& node) { return node.getState()->set("nested", base::Any(true)); };
  ASSERT_TRUE(parent.init());
  auto state = std::make_shared<State>();
  EXPECT_TRUE(parent.runWithState(state));
  EXPECT_TRUE(state->contains("nested"));
  EXPECT_EQ(leaf.getState(), nullptr);
  EXPECT_TRUE(parent.deinit());
}

TEST(StateControl, PipelineIsRejectedBeforeStateExecution) {
  StateNode node("node");
  Graph graph("pipeline");
  graph.setParallelType(base::kParallelTypePipeline);
  graph.addNode(&node); graph.setExecutionOrder({"node"});
  EXPECT_EQ(graph.init(), base::kStatusCodeErrorNotSupport);
  ASSERT_TRUE(graph.setExecutionOrder({}));
  ASSERT_TRUE(graph.init());
  EXPECT_EQ(graph.runWithState(std::make_shared<State>()), base::kStatusCodeErrorNotSupport);
  EXPECT_EQ(graph.getState(), nullptr);
  EXPECT_TRUE(graph.deinit());
}

TEST(StateControl, DomDeserializationAndMalformedControlDefinitions) {
  StateNode a("a"), b("b");
  Graph source("source");
  source.addNode(&a); source.addNode(&b);
  source.setExecutionOrder({"a", "b"});
  rapidjson::Document json;
  json.Parse(source.serialize().c_str());
  Graph restored("restored");
  ASSERT_TRUE(restored.deserialize(json));
  ASSERT_EQ(restored.getNodeCount(), 2);
  EXPECT_TRUE(restored.init());
  EXPECT_TRUE(restored.run());
  EXPECT_TRUE(restored.deinit());
  json["execution_order"].SetString("not an array", json.GetAllocator());
  Graph invalid("invalid");
  EXPECT_FALSE(invalid.deserialize(json));
  EXPECT_FALSE(invalid.deserialize("[]"));
}

TEST(StateControl, ParentPipelineRejectsNestedControlBeforeStartingWorkers) {
  StateNode leaf("leaf");
  Graph child("child"), parent("parent");
  child.setParallelType(base::kParallelTypeSequential);
  child.addNode(&leaf); child.setExecutionOrder({"leaf"});
  parent.addNode(&child);
  parent.setParallelType(base::kParallelTypePipeline);
  EXPECT_EQ(parent.init(), base::kStatusCodeErrorNotSupport);
  EXPECT_FALSE(leaf.getInitialized());
}

TEST(StateControl, StateRunReservesSubgraphsBeforeTheyAreScheduled) {
  StateNode gate("gate"), leaf("leaf");
  Graph child("child"), parent("parent");
  child.addNode(&leaf); child.setExecutionOrder({"leaf"});
  parent.addNode(&gate); parent.addNode(&child);
  parent.setExecutionOrder({"gate", "child"});
  parent.setParallelType(base::kParallelTypeTask);
  std::promise<void> entered, release;
  auto wait = release.get_future().share();
  gate.action = [&](StateNode&) {
    entered.set_value();
    EXPECT_EQ(wait.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    return base::Status::Ok();
  };
  leaf.action = [](StateNode& node) { return node.getState()->set("leaf", base::Any(true)); };
  ASSERT_TRUE(parent.init());
  auto state = std::make_shared<State>();
  auto worker = std::async(std::launch::async, [&] { return parent.runWithState(state); });
  EXPECT_EQ(entered.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_FALSE(child.run());
  EXPECT_FALSE(child.runWithState(std::make_shared<State>()));
  release.set_value();
  EXPECT_TRUE(worker.get());
  EXPECT_TRUE(state->contains("leaf"));
  EXPECT_TRUE(parent.deinit());
}

TEST(StateControl, ParentCannotJoinIndependentlyRunningChildState) {
  StateNode leaf("leaf");
  Graph child("child"), parent("parent");
  child.addNode(&leaf); child.setExecutionOrder({"leaf"});
  parent.addNode(&child); parent.setExecutionOrder({"child"});
  std::promise<void> entered, release;
  auto wait = release.get_future().share();
  leaf.action = [&](StateNode&) {
    entered.set_value();
    EXPECT_EQ(wait.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    return base::Status::Ok();
  };
  ASSERT_TRUE(parent.init());
  auto worker = std::async(std::launch::async, [&] {
    return child.runWithState(std::make_shared<State>());
  });
  EXPECT_EQ(entered.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_FALSE(parent.run());
  release.set_value();
  EXPECT_TRUE(worker.get());
  EXPECT_EQ(leaf.calls, 1);
  EXPECT_TRUE(parent.deinit());
}

class StateComposite : public CompositeNode {
 public:
  using CompositeNode::CompositeNode;
  base::Status run() override { return getNode("leaf")->run(); }
};

TEST(StateControl, CompositeNodesUseTheSameState) {
  StateNode leaf("leaf");
  StateComposite composite("composite");
  Graph graph("outer");
  composite.addNode(&leaf);
  graph.addNode(&composite); graph.setExecutionOrder({"composite"});
  leaf.action = [](StateNode& node) { return node.getState()->set("value", base::Any(5)); };
  ASSERT_TRUE(graph.init());
  auto state = std::make_shared<State>();
  EXPECT_TRUE(graph.runWithState(state));
  EXPECT_TRUE(state->contains("value"));
  EXPECT_EQ(leaf.getState(), nullptr);
  EXPECT_TRUE(graph.deinit());
}

class StateChoice : public Condition {
 public:
  using Condition::Condition;
  std::function<int()> select;
  int choose() override { return select ? select() : 0; }
};

class StateLoop : public Loop {
 public:
  using Loop::Loop;
  int bound = 3;
  std::function<int()> limit;
  std::function<base::Status()> before, after;
  int loops() override { return limit ? limit() : bound; }
  base::Status iterBefore() override { return before ? before() : base::Status::Ok(); }
  base::Status iterAfter() override { return after ? after() : base::Status::Ok(); }
};

int stateInt(const std::shared_ptr<State>& state, const std::string& key) {
  base::Any value;
  if (!state || state->get(key, value) != base::kStatusCodeOk)
    throw std::runtime_error("missing state field: " + key);
  return base::get<int>(value);
}

class ControlFlowStateTest : public testing::TestWithParam<base::ParallelType> {};

TEST_P(ControlFlowStateTest, ConditionSelectsOnlyOneZeroEdgeBranchAndRecovers) {
  StateNode a("a"), b("b");
  StateChoice choice("choice");
  choice.setParallelType(GetParam());
  choice.addNode(&a); choice.addNode(&b);
  choice.select = [&] { return stateInt(choice.getState(), "branch"); };
  a.action = [](StateNode& node) { return node.getState()->set("selected", base::Any(0)); };
  b.action = [](StateNode& node) { return node.getState()->set("selected", base::Any(1)); };
  // Alternatives are not a sequence. Put a Graph inside a branch for multiple steps.
  EXPECT_EQ(choice.setExecutionOrder({"a", "b"}), base::kStatusCodeErrorNotSupport);
  ASSERT_TRUE(choice.init());
  auto state = std::make_shared<State>();
  state->set("branch", base::Any(-1));
  EXPECT_EQ(choice.runWithState(state), base::kStatusCodeErrorInvalidValue);
  EXPECT_FALSE(choice.isRunning());
  EXPECT_EQ(choice.getState(), nullptr);
  for (int branch : {1, 0}) {
    state = std::make_shared<State>();
    state->set("branch", base::Any(branch));
    EXPECT_TRUE(choice.runWithState(state));
    EXPECT_EQ(stateInt(state, "selected"), branch);
  }
  EXPECT_EQ(a.calls, 1);
  EXPECT_EQ(b.calls, 1);
  EXPECT_TRUE(choice.deinit());
}

TEST_P(ControlFlowStateTest, LoopConditionAndBranchGraphShareStateEveryIteration) {
  StateNode one("one"), ten("ten");
  Graph branch("branch");
  StateChoice choice("choice");
  StateLoop loop("loop");
  branch.addNode(&one); branch.setExecutionOrder({"one"});
  choice.addNode(&branch); choice.addNode(&ten);
  loop.addNode(&choice); loop.setExecutionOrder({"choice"});
  loop.setParallelType(GetParam());
  one.action = [](StateNode& n) {
    return n.getState()->set("sum", base::Any(stateInt(n.getState(), "sum") + 1));
  };
  ten.action = [](StateNode& n) {
    return n.getState()->set("sum", base::Any(stateInt(n.getState(), "sum") + 10));
  };
  choice.select = [&] { return stateInt(choice.getState(), "step") % 2; };
  loop.limit = [&] { return stateInt(loop.getState(), "limit"); };
  loop.after = [&] {
    return loop.getState()->set("step", base::Any(stateInt(loop.getState(), "step") + 1));
  };
  ASSERT_TRUE(loop.init());
  for (int limit : {3, 0, 2}) {
    auto state = std::make_shared<State>();
    state->set("sum", base::Any(0)); state->set("step", base::Any(0));
    state->set("limit", base::Any(limit));
    EXPECT_TRUE(loop.runWithState(state));
    EXPECT_EQ(stateInt(state, "step"), limit);
    EXPECT_EQ(stateInt(state, "sum"), limit == 3 ? 12 : (limit == 2 ? 11 : 0));
    EXPECT_EQ(choice.getState(), nullptr);
    EXPECT_EQ(branch.getState(), nullptr);
    EXPECT_EQ(one.getState(), nullptr);
  }
  EXPECT_TRUE(loop.deinit());
}

TEST_P(ControlFlowStateTest, LoopOrderHooksDynamicBoundAndFailure) {
  StateNode a("a"), b("b");
  StateLoop loop("loop");
  loop.setParallelType(GetParam());
  loop.addNode(&b); loop.addNode(&a);
  ASSERT_TRUE(loop.setExecutionOrder({"a", "b"}));
  std::vector<std::string> trace;
  loop.limit = [&] { return stateInt(loop.getState(), "limit"); };
  loop.before = [&] { trace.push_back("before"); return base::Status::Ok(); };
  a.action = [&](StateNode&) { trace.push_back("a"); return base::Status::Ok(); };
  b.action = [&](StateNode&) { trace.push_back("b"); return base::Status::Ok(); };
  loop.after = [&] {
    trace.push_back("after");
    return loop.getState()->set("limit", base::Any(2));
  };
  ASSERT_TRUE(loop.init());
  auto state = std::make_shared<State>(); state->set("limit", base::Any(20));
  EXPECT_TRUE(loop.runWithState(state));
  EXPECT_EQ(trace, (std::vector<std::string>{"before", "a", "b", "after",
                                           "before", "a", "b", "after"}));
  loop.before = []() -> base::Status { throw std::runtime_error("before failed"); };
  EXPECT_FALSE(loop.runWithState(state));
  EXPECT_EQ(a.calls, 2);
  EXPECT_FALSE(loop.isRunning());
  EXPECT_EQ(loop.getState(), nullptr);
  loop.before = nullptr;
  a.action = [](StateNode&) -> base::Status { throw std::runtime_error("body failed"); };
  EXPECT_FALSE(loop.runWithState(state));
  EXPECT_EQ(b.calls, 2);
  EXPECT_FALSE(loop.isRunning());
  EXPECT_EQ(loop.getState(), nullptr);
  EXPECT_TRUE(loop.deinit());
}

TEST_P(ControlFlowStateTest, InterruptionStopsLoopBeforeNextIteration) {
  StateNode node("body");
  StateLoop loop("loop");
  loop.setParallelType(GetParam());
  loop.addNode(&node); loop.setExecutionOrder({"body"});
  node.action = [&](StateNode&) {
    EXPECT_TRUE(loop.interrupt());
    return base::Status::Ok();
  };
  ASSERT_TRUE(loop.init());
  EXPECT_EQ(loop.runWithState(std::make_shared<State>()), base::kStatusCodeNodeInterrupt);
  EXPECT_EQ(node.calls, 1);
  EXPECT_FALSE(loop.isRunning());
  EXPECT_EQ(loop.getState(), nullptr);
  EXPECT_TRUE(loop.deinit());
}

INSTANTIATE_TEST_SUITE_P(Executors, ControlFlowStateTest,
                        testing::Values(base::kParallelTypeSequential, base::kParallelTypeTask));

TEST(StateControl, FixedLoopJsonRetainsBodyOrderAndIterationCount) {
  StateNode a("a"), b("b");
  FixedLoop loop("loop");
  loop.setLoops(3); loop.addNode(&b); loop.addNode(&a);
  loop.setExecutionOrder({"a", "b"});
  Graph parent("parent");
  parent.addNode(&loop); parent.setExecutionOrder({"loop"});
  Graph copy("copy");
  ASSERT_TRUE(copy.deserialize(parent.serialize()));
  auto* restored = dynamic_cast<FixedLoop*>(copy.getNode("loop"));
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->loops(), 3);
  ASSERT_TRUE(copy.init());
  EXPECT_TRUE(copy.runWithState(std::make_shared<State>()));
  EXPECT_EQ(dynamic_cast<StateNode*>(restored->getNode("a"))->calls, 3);
  EXPECT_TRUE(copy.deinit());
}

TEST(StateControl, ConditionReservesEvenUnselectedZeroEdgeGraphBranch) {
  StateNode leaf("leaf"), other("other");
  Graph branch("branch");
  StateChoice choice("choice");
  branch.addNode(&leaf); branch.setExecutionOrder({"leaf"});
  choice.addNode(&branch); choice.addNode(&other);
  std::promise<void> entered, release;
  auto wait = release.get_future().share();
  choice.select = [&] {
    EXPECT_FALSE(choice.run());
    entered.set_value();
    EXPECT_EQ(wait.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    return 1;
  };
  ASSERT_TRUE(choice.init());
  auto worker = std::async(std::launch::async, [&] {
    return choice.runWithState(std::make_shared<State>());
  });
  EXPECT_EQ(entered.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_FALSE(choice.run());
  EXPECT_FALSE(branch.run());
  EXPECT_FALSE(branch.runWithState(std::make_shared<State>()));
  release.set_value();
  EXPECT_TRUE(worker.get());
  EXPECT_EQ(leaf.calls, 0);
  EXPECT_EQ(other.calls, 1);
  EXPECT_TRUE(choice.deinit());
}

TEST(StateControl, LegacyConditionAndLoopStillRunWithoutStateOrEdges) {
  StateNode leaf("leaf");
  StateChoice choice("choice");
  FixedLoop loop("loop");
  choice.addNode(&leaf);
  loop.addNode(&choice); loop.setLoops(2);
  EXPECT_FALSE(loop.run());
  EXPECT_TRUE(loop.deinit());
  ASSERT_TRUE(loop.init());
  EXPECT_TRUE(loop.run());
  EXPECT_EQ(leaf.calls, 2);
  EXPECT_TRUE(loop.deinit());
}

}  // namespace
}  // namespace dag
}  // namespace nndeploy
