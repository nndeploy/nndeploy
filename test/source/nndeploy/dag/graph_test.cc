#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/dag/graph.h"

using namespace nndeploy::dag;
using namespace nndeploy::base;

class GraphTest : public testing::Test {
    protected:
    std::unique_ptr<Graph> ConstructGraph(const std::string &name, 
                                          Edge *input, Edge *output) {
        return std::make_unique<Graph>(name, input, output);
    }

    std::unique_ptr<Graph> ConstructGraphWithVecArgs(const std::string &name, 
                                                     std::vector<Edge *> inputs,
                                                     std::vector<Edge *> outputs) {
        return std::make_unique<Graph>(name, inputs, outputs);
    }

    std::unique_ptr<Graph> ConstructGraphWithInitLsArgs(const std::string &name, 
                                                     std::initializer_list<Edge *> inputs,
                                                     std::initializer_list<Edge *> outputs) {
        return std::make_unique<Graph>(name, inputs, outputs);
    }
};
class ProcessNode : public Node {
 public:
  ProcessNode(const std::string &name, std::vector<Edge *> inputs, std::vector<Edge *> outputs)
      : Node(name, inputs, outputs) {}
  ProcessNode(const std::string &name, Edge* input, Edge* output)
      : Node(name, input, output) {}
  
  virtual ~ProcessNode() {}

  virtual Status run() {
    return kStatusCodeOk;
  }
};

TEST_F(GraphTest, GraphWithOneInputOutputEdge) {
    auto edge_in = std::make_unique<Edge>("edge_in");
    auto edge_out = std::make_unique<Edge>("edge_out");
    auto graph = ConstructGraph("@@!!##$$", edge_in.get(), edge_out.get());
    ASSERT_TRUE(graph->getConstructed());
    EXPECT_TRUE(graph->getName() == "@@!!##$$");
    EXPECT_TRUE(graph->getEdge("edge_in") == edge_in.get());
    EXPECT_TRUE(graph->getEdge("edge_out") == edge_out.get());
}


TEST_F(GraphTest, GraphWithDuplicateOutputEdge) {
    auto edge_in =  std::make_unique<Edge>("edge_in");
    auto graph = ConstructGraph("3.141@@!!", edge_in.get(), edge_in.get());
    EXPECT_FALSE(graph->getConstructed());
    EXPECT_TRUE(graph->getAllOutput().size() == 0);
    EXPECT_TRUE(graph->getAllInput().size() == 0);
}

TEST_F(GraphTest, GraphWithVectorInputOutputEdge) {
    auto inputs = std::vector<Edge *>();
    auto outputs = std::vector<Edge *>();
    inputs.emplace_back(new Edge("in_1"));
    inputs.emplace_back(new Edge("in_2"));
    inputs.emplace_back(new Edge("in_3"));
    inputs.emplace_back(new Edge("in_4"));
    outputs.emplace_back(new Edge("out_1"));
    outputs.emplace_back(new Edge("out_2"));
    outputs.emplace_back(new Edge("out_3"));
    outputs.emplace_back(new Edge("out_4"));
    auto graph = ConstructGraphWithVecArgs("@@\n\t!!##$$", inputs, outputs);
    ASSERT_TRUE(graph->getConstructed());
    ASSERT_EQ(graph->getAllInput().size(), 4);
    ASSERT_EQ(graph->getAllOutput().size(), 4);
    
    //TODO: do this in ~GraphTest()
    for(auto in_edge: inputs) {delete in_edge;}
    for(auto out_edge: outputs) {delete out_edge;}
}

TEST_F(GraphTest, GraphAddItselfAsNode) {
    auto inputs = std::vector<Edge *>();
    auto outputs = std::vector<Edge *>();
    auto edge_in =  std::make_unique<Edge>("edge_in");
    auto edge_out =  std::make_unique<Edge>("edge_out");
    inputs.emplace_back(edge_in.get());
    outputs.emplace_back(edge_out.get());
    auto graph = ConstructGraphWithVecArgs("AddItselfAsNode", inputs, outputs);
    ASSERT_TRUE(graph->addNode(graph.get()) != kStatusCodeOk);
}

TEST_F(GraphTest, GraphCreateNode) {
    auto inputs = std::vector<Edge *>();
    auto outputs = std::vector<Edge *>();
    auto edge_in =  std::make_unique<Edge>("edge_in");
    auto edge_out =  std::make_unique<Edge>("edge_out");
    auto node_edge_in =  std::make_unique<Edge>("node_edge_in");
    auto node_edge_out =  std::make_unique<Edge>("node_edge_out");
    inputs.emplace_back(edge_in.get());
    outputs.emplace_back(edge_out.get());
    auto graph = ConstructGraphWithVecArgs("CreateNode", inputs, outputs);
    auto node = graph->createNode<ProcessNode>("test_node", node_edge_in.get(), node_edge_out.get());
    ASSERT_EQ(node->getName(), "test_node");
    ASSERT_EQ(node->getAllInput().size(), 1);
    ASSERT_EQ(node->getAllOutput().size(), 1);
    ASSERT_EQ(node->getInput(), node_edge_in.get());
    ASSERT_EQ(node->getOutput(), node_edge_out.get());
}

//failing test
TEST_F(GraphTest, GraphCreateNodeWithSameEdges) {
    auto inputs = std::vector<Edge *>();
    auto outputs = std::vector<Edge *>();
    auto edge_in =  std::make_unique<Edge>("edge_in");
    auto edge_out =  std::make_unique<Edge>("edge_out");
    inputs.emplace_back(edge_in.get());
    outputs.emplace_back(edge_out.get());
    auto graph = ConstructGraphWithVecArgs("CreateNodeWithSameEdges", inputs, outputs);
    auto node = graph->createNode<ProcessNode>("test_node", inputs, outputs); //this may be a problem
    ASSERT_TRUE(node == nullptr);
}

TEST_F(GraphTest, GraphAddEdge) {
    auto edge_in =  std::make_unique<Edge>("edge_in");
    auto edge_out =  std::make_unique<Edge>("edge_out");
    auto graph = ConstructGraph("GraphAddEdge", edge_in.get(), edge_out.get());
    auto new_edge =  std::make_unique<Edge>("new_edge");
    auto edge_wrapper = graph->addEdge(new_edge.get()); 
    ASSERT_EQ(edge_wrapper->consumers_.size(), 0);
    ASSERT_EQ(edge_wrapper->producers_.size(), 0);
    ASSERT_EQ(edge_wrapper->edge_, new_edge.get());
    ASSERT_EQ(edge_wrapper->edge_->getName(), new_edge.get()->getName());
}

TEST_F(GraphTest, GraphDuplicateAddEdge) {
    auto edge_in =  std::make_unique<Edge>("edge_in");
    auto edge_out =  std::make_unique<Edge>("edge_out");
    auto graph = ConstructGraph("GraphAddEdge", edge_in.get(), edge_out.get());
    auto edge_wrapper_in = graph->addEdge(edge_in.get());
    auto edge_wrapper_out = graph->addEdge(edge_out.get());
    ASSERT_EQ(edge_wrapper_in, nullptr);
    ASSERT_EQ(edge_wrapper_out, nullptr);
}