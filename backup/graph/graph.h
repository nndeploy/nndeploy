

#include "nndeploy/include/graph/graph.h"

namespace nndeploy {
namespace graph {

class EdgeWrapper;

class NodeWrapper;

class Graph : public Node {
 public:
  Graph(const std::string& name = "");
  virtual ~Graph();

  template <typename T>
  Edge* createEdge(const std::string& name = "");
  void addEdge(Edge* edge);

  template <typename T>
  Node* createNode(const std::string& name = "", base::Param* param = nullptr,
                   Packet* input = nullptr, Packet* output = nullptr);
  virtual base::Status addNode(Node* node);

  virtual base::Status setName(const std::string& name);
  virtual std::string getName();

  virtual base::Status setParam(base::Param* param);
  virtual base::Param* getParam();

  virtual Packet* getInput();
  virtual Packet* getOutput();

  virtual base::Status setInput(Packet* input);
  virtual base::Status setOutput(Packet* output);

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::ShapeMap inferOuputShape();

  virtual base::Status run();

  virtual base::Status dump(std::ostream& oss = std::cout);

 protected:
  std::vector<EdgeWrapper*> edge_repository_;
  std::vector<NodeWrapper*> node_repository_;
};

}  // namespace graph
}  // namespace nndeploy
