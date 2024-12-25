#include <gtest/gtest.h>
#include <memory>
#include "nndeploy/dag/edge.h"

using nndeploy::dag::Edge;

class EdgeTest : public testing::Test {
    protected:
    std::unique_ptr<Edge> ConstructEdge(const std::string &name) {
        return  std::make_unique<Edge>(name);
    }
};

TEST_F(EdgeTest, CreateEdgeWithValidName) {
    std::string name = "edge!@#$%^&*()";
    auto edge = ConstructEdge(name);
    EXPECT_EQ(edge->getName(), name);
}

TEST_F(EdgeTest, CreateEdgeWithNullName) {
    std::string name;
    auto edge = ConstructEdge(name);
    EXPECT_EQ(edge->getName(), ""); //should we do something about this?
}

TEST_F(EdgeTest, CreateEdgeWithEmptyName) {
    auto edge = ConstructEdge("");
    EXPECT_TRUE(edge->getName().empty()); //should we do something about this?
}