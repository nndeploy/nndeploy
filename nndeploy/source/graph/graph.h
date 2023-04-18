#ifndef _NNDEPLOY_SOURCE_GRAPH_GRAPH_
#define _NNDEPLOY_SOURCE_GRAPH_GRAPH_

namespace nndeploy {
namespace graph {

class Graph : public Base {
  virtual base::Status addNode(Node* node);
  virtual base::Status addNode(Node* node, Node* depend_node);
  virtual base::Status addNode(Node* node, std::vector<Node*> depend_node);

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status preRun();
  virtual base::Status postRun();

  virtual base::Status Run();
  virtual base::Status aysncRun();

 private:
  std::vector<Param*> params_;
  std::vector<Edge*> edges_;

  std::vector<Node*> nodes_;
  std::vector<Pipeline*> pipelines_;
  std::vector<Graph*> subgraphs_;

  std::vector<Element*> executors_;
}

}  // namespace graph
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_GRAPH_GRAPH_ */
