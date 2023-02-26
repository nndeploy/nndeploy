/**
 * @file device.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-24
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_GRAPH_GRAPH_
#define _NNDEPLOY_INCLUDE_GRAPH_GRAPH_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/device.h"
#include "nndeploy/include/graph/config.h"
#include "nndeploy/include/graph/node.h"
#include "nndeploy/include/graph/packet.h"

namespace nndeploy {
namespace graph {

class Graph {
 public:
  explicit Graph(const std::string &name, const std::string &config_json);
  virtual ~Graph();

  virtual base::Status setNodeConfig(const std::string &node_name,
                                     const std::string &key, base::Value value);
  virtual base::Status setNodeConfig(const std::string &node_name,
                                     const Config &config);
  virtual base::Status addInput(std::vector<std::string> input_name);
  virtual base::Status addOutput(std::vector<std::string> input_name);
  virtual base::Status addInputNode(Node *node);
  virtual base::Status addOutputNode(Node *node);
  virtual base::Status addNode(Node *node);
  virtual base::Status addSubgraph(Graph *node);

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status preRun();
  virtual base::Status postRun();

  virtual base::Status setInput(std::string name, const Packet &input);
  virtual base::Status getOutput(std::string name, Packet &output);

  virtual base::Status run();
  virtual base::Status asyncRun();

 private:
  std::string name_;

  std::vector<std::string> input_name_;
  std::vector<std::string> output_name_;
  std::vector<std::string, std::shared_ptr<Packet>> input_;
  std::vector<std::string, std::shared_ptr<Packet>> output_;
  std::vector<Node *> nodes_;
  std::vector<Graph *> graphs_;
};

}  // namespace graph
}  // namespace nndeploy

#endif