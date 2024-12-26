#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/dag/graph.h"
using namespace nndeploy::dag;

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
