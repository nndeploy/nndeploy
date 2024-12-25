/*
    Analysis: nndeploy::dag::Edge

    happy path tests:
                valid_edge: constructs an edge with a name 


    sad path tests:
                none

    evil path test:
                none
*/
#include <gtest/gtest.h>

#include "nndeploy/dag/graph.h"

#include<string>
using nndeploy::dag::Edge;

Edge* construct_edge(const std::string &name)
{
    Edge* edge_ = new Edge(name);
    return edge_; 
}

TEST(edge_test, valid_edge) {
    std::string name = "test_edge";
    Edge* edge_ = construct_edge(name);
    EXPECT_EQ(edge_->getName(), name);
    delete(edge_);
}

TEST(edge_test, invalid_edge) {
    std::string name;
    Edge* edge_ = construct_edge(name);
    EXPECT_EQ(edge_->getName(), ""); //should we do something about this?
    delete(edge_);
}