#include "nndeploy/dag/edge.h"

#include <gtest/gtest.h>

#include <memory>

class EdgeTest : public testing::Test {
 protected:
  std::unique_ptr<nndeploy::dag::Edge> constructEdge(const std::string &name) {
    return std::make_unique<nndeploy::dag::Edge>(name);
  }
};

TEST_F(EdgeTest, EdgeWithValidName) {
  std::string name = "edge!@#$%^&*()";
  auto edge = constructEdge(name);
  EXPECT_EQ(edge->getName(), name);
}

TEST_F(EdgeTest, DefaultConstructorGeneratesName) {
  nndeploy::dag::Edge edge;
  EXPECT_EQ(edge.getName().find("edge_"), 0u);
  EXPECT_GT(edge.getName().size(), 5u);
}

TEST_F(EdgeTest, EmptyNamesGenerateDistinctEdgeNames) {
  auto first = constructEdge("");
  auto second = constructEdge("");
  EXPECT_FALSE(first->getName().empty());
  EXPECT_FALSE(second->getName().empty());
  EXPECT_NE(first->getName(), second->getName());
}
