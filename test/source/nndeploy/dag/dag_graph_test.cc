#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/base/param.h"

class GraphTest : public testing::Test {
    protected:
    std::unique_ptr<nndeploy::dag::Graph> constructGraph(const std::string &name, 
                                                         nndeploy::dag::Edge *input, 
                                                         nndeploy::dag::Edge *output) {
        return std::make_unique<nndeploy::dag::Graph>(name, input, output);
    }

    std::unique_ptr<nndeploy::dag::Graph> constructGraphWithVecArgs(const std::string &name, 
                                                                    std::vector<nndeploy::dag::Edge *> inputs,
                                                                    std::vector<nndeploy::dag::Edge *> outputs) {
        return std::make_unique<nndeploy::dag::Graph>(name, inputs, outputs);
    }

    std::unique_ptr<nndeploy::dag::Graph> constructGraphWithInitLsArgs(const std::string &name, 
                                                                       std::initializer_list<nndeploy::dag::Edge *> inputs,
                                                                       std::initializer_list<nndeploy::dag::Edge *> outputs) {
        return std::make_unique<nndeploy::dag::Graph>(name, inputs, outputs);
    }
};
class ProcessNode : public nndeploy::dag::Node {
 public:
  ProcessNode(const std::string &name, 
              std::vector<nndeploy::dag::Edge *> inputs, 
              std::vector<nndeploy::dag::Edge *> outputs)
      : Node(name, inputs, outputs) {}

  ProcessNode(const std::string &name, 
              nndeploy::dag::Edge* input, 
              nndeploy::dag::Edge* output)
      : Node(name, input, output) {}
  
  virtual ~ProcessNode() {}

  virtual nndeploy::base::Status run() {
    return nndeploy::base::kStatusCodeOk;
  }
};

TEST_F(GraphTest, GraphWithOneInputOutputEdge) {
    using namespace nndeploy::dag;
    
    auto edge_in = std::make_unique<Edge>("edge_in");
    auto edge_out = std::make_unique<Edge>("edge_out");
    auto graph = constructGraph("@@!!##$$", edge_in.get(), edge_out.get());
    ASSERT_TRUE(graph->getConstructed());
    EXPECT_TRUE(graph->getName() == "@@!!##$$");
    EXPECT_TRUE(graph->getEdge("edge_in") == edge_in.get());
    EXPECT_TRUE(graph->getEdge("edge_out") == edge_out.get());
}


TEST_F(GraphTest, GraphWithDuplicateOutputEdge) {
    using namespace nndeploy::dag;
    
    auto edge_in = std::make_unique<Edge>("edge_in");
    auto graph = constructGraph("3.141@@!!", edge_in.get(), edge_in.get());
    EXPECT_FALSE(graph->getConstructed());
    EXPECT_TRUE(graph->getAllOutput().size() == 0);
    EXPECT_TRUE(graph->getAllInput().size() == 0);
}

TEST_F(GraphTest, GraphWithVectorInputOutputEdges) {
    using namespace nndeploy::dag;

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
    auto graph = constructGraphWithVecArgs("@@\n\t!!##$$", inputs, outputs);
    ASSERT_TRUE(graph->getConstructed());
    ASSERT_EQ(graph->getAllInput().size(), 4);
    ASSERT_EQ(graph->getAllOutput().size(), 4);
    
    //TODO: do this in ~GraphTest()
    for(auto in_edge: inputs) {delete in_edge;}
    for(auto out_edge: outputs) {delete out_edge;}
}

TEST_F(GraphTest, GraphWithInitListInputOutputEdges) {
    using namespace nndeploy::dag;

    auto inputs = std::initializer_list<Edge *>{new Edge("edge_in")};
    auto outputs = std::initializer_list<Edge *>{new Edge("edge_out")};
    auto graph = constructGraphWithInitLsArgs("@@\n\t!!##$$", inputs, outputs);
    ASSERT_TRUE(graph->getConstructed());
    ASSERT_EQ(graph->getAllInput().size(), 1);
    ASSERT_EQ(graph->getAllOutput().size(), 1);
}

TEST_F(GraphTest, GraphAddItselfAsNode) {
    using namespace nndeploy::dag;
    using namespace nndeploy::base;

    auto inputs = std::vector<Edge *>();
    auto outputs = std::vector<Edge *>();
    auto edge_in = std::make_unique<Edge>("edge_in");
    auto edge_out = std::make_unique<Edge>("edge_out");
    inputs.emplace_back(edge_in.get());
    outputs.emplace_back(edge_out.get());
    auto graph = constructGraphWithVecArgs("AddItselfAsNode", inputs, outputs);
    ASSERT_TRUE(graph->addNode(graph.get()) != kStatusCodeOk);
}

TEST_F(GraphTest, GraphCreateNode) {
    using namespace nndeploy::dag;

    auto inputs = std::vector<Edge *>();
    auto outputs = std::vector<Edge *>();
    auto edge_in = std::make_unique<Edge>("edge_in");
    auto edge_out = std::make_unique<Edge>("edge_out");
    auto node_edge_in = std::make_unique<Edge>("node_edge_in");
    auto node_edge_out = std::make_unique<Edge>("node_edge_out");
    inputs.emplace_back(edge_in.get());
    outputs.emplace_back(edge_out.get());
    auto graph = constructGraphWithVecArgs("CreateNode", inputs, outputs);
    auto node = graph->createNode<ProcessNode>("test_node", node_edge_in.get(), node_edge_out.get());
    ASSERT_EQ(node->getName(), "test_node");
    ASSERT_EQ(node->getAllInput().size(), 1);
    ASSERT_EQ(node->getAllOutput().size(), 1);
    ASSERT_EQ(node->getInput(), node_edge_in.get());
    ASSERT_EQ(node->getOutput(), node_edge_out.get());
}

TEST_F(GraphTest, GraphCreateNodeWithSameEdges) {
    using namespace nndeploy::dag;

    auto inputs = std::vector<Edge *>();
    auto outputs = std::vector<Edge *>();
    auto edge_in = std::make_unique<Edge>("edge_in");
    auto edge_out = std::make_unique<Edge>("edge_out");
    inputs.emplace_back(edge_in.get());
    outputs.emplace_back(edge_out.get());
    auto graph = constructGraphWithVecArgs("CreateNodeWithSameEdges", inputs, outputs);
    auto node = graph->createNode<ProcessNode>("test_node", inputs, outputs);
    ASSERT_TRUE(node != nullptr);
}

TEST_F(GraphTest, GraphAddEdge) {
    using namespace nndeploy::dag;

    auto edge_in = std::make_unique<Edge>("edge_in");
    auto edge_out = std::make_unique<Edge>("edge_out");
    auto graph = constructGraph("GraphAddEdge", edge_in.get(), edge_out.get());
    auto new_edge = std::make_unique<Edge>("new_edge");
    auto edge_wrapper = graph->addEdge(new_edge.get()); 
    ASSERT_EQ(edge_wrapper->consumers_.size(), 0);
    ASSERT_EQ(edge_wrapper->producers_.size(), 0);
    ASSERT_EQ(edge_wrapper->edge_, new_edge.get());
    ASSERT_EQ(edge_wrapper->edge_->getName(), new_edge.get()->getName());
}

TEST_F(GraphTest, GraphDuplicateAddEdge) {
    using namespace nndeploy::dag;

    auto edge_in = std::make_unique<Edge>("edge_in");
    auto edge_out = std::make_unique<Edge>("edge_out");
    auto graph = constructGraph("GraphDuplicateAddEdge", edge_in.get(), edge_out.get());
    auto edge_wrapper_in = graph->addEdge(edge_in.get());
    auto edge_wrapper_out = graph->addEdge(edge_out.get());
    ASSERT_EQ(edge_wrapper_in, nullptr);
    ASSERT_EQ(edge_wrapper_out, nullptr);
}


TEST_F(GraphTest, GraphSetNodeParamNullParam) {
    using namespace nndeploy::dag;
    using namespace nndeploy::base;

    auto edge_in = std::make_unique<Edge>("edge_in");
    auto edge_out = std::make_unique<Edge>("edge_out");
    auto graph = constructGraph("GraphNullSetNodeParam", edge_in.get(), edge_out.get());
    Param *param = nullptr;
    ASSERT_EQ(graph->setNodeParam("GraphNullSetNodeParam", param), kStatusCodeErrorNullParam);
}

// TEST_F(GraphTest, GraphSetNodeParamNullNode) {
//     auto edge_in =  std::make_unique<Edge>("edge_in");
//     auto edge_out =  std::make_unique<Edge>("edge_out");
//     auto graph = constructGraph("GraphNullSetNodeParam", edge_in.get(), edge_out.get());
//     Param *param = std::make_unique<Param>().get();
//     ASSERT_EQ(graph->setNodeParam("GraphNullSetNodeParam", param), kStatusCodeErrorNullParam);
// }
    // int i = 1;
    // while(i < 51) {
    //     auto edge_in =  std::make_unique<Edge>(std::to_string(i));
    //     auto edge_out =  std::make_unique<Edge>(std::to_string(i*2));
    //     inputs.emplace_back(edge_in.get());
    //     outputs.emplace_back(edge_out.get());
    // }