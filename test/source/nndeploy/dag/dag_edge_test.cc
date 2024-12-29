#include <gtest/gtest.h>
#include <memory>
#include "nndeploy/dag/edge.h"

class EdgeTest : public testing::Test {
    
    protected:
    std::unique_ptr<nndeploy::dag::Edge> ConstructEdge(const std::string &name) {
        return  std::make_unique<nndeploy::dag::Edge>(name);
    }
};

TEST_F(EdgeTest, EdgeWithValidName) {
    std::string name = "edge!@#$%^&*()";
    auto edge = ConstructEdge(name);
    EXPECT_EQ(edge->getName(), name);
}

TEST_F(EdgeTest, EdgeWithNullName) {
    std::string name;
    auto edge = ConstructEdge(name);
    EXPECT_EQ(edge->getName(), ""); //should we do something about this?
}

TEST_F(EdgeTest, EdgeWithEmptyName) {
    auto edge = ConstructEdge("");
    EXPECT_TRUE(edge->getName().empty()); //should we do something about this?
}