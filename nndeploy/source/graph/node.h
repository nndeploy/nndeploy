#ifndef _NNDEPLOY_SOURCE_GRAPH_NODE_H_
#define _NNDEPLOY_SOURCE_GRAPH_NODE_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"

namespace nndeploy {
namespace graph {

class Node : public Base {
 public:
  Node(const std::string& guid = "", const std::string& name = "")
      : Base(guid, name){};
  virtual ~Node() = default;

  virtual base::Status addInput(Edge* edge);
  virtual base::Status addInput(std::vector<Edge*> edge);
  virtual base::Status addOutput(Edge* edge);
  virtual base::Status addOutput(std::vector<Edge*> edge);

  base::Status precede(Node* node);
  base::Status precede(std::vector<Node*> node);
  base::Status succeed(Node* node);
  base::Status succeed(std::vector<Node*> node);

 protected:
}

}  // namespace graph
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_GRAPH_NODE_H_ */
