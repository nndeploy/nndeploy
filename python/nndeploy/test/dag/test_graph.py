
import nndeploy._nndeploy_internal as _C

# from nndeploy._nndeploy_internal import Node, NodeDesc, Graph

import nndeploy.base
import nndeploy.device
import nndeploy.dag
import torch


# python3 nndeploy/test/dag/test_graph.py


class CustomNode(nndeploy.dag.Node):
    def __init__(self, name, inputs: list[_C.dag.Edge] = None, outputs: list[_C.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
        self.inputs_type = [torch.Tensor for _ in inputs] if inputs else []
        self.outputs_type = [torch.Tensor for _ in outputs] if outputs else []
    
    def init(self):
        print("CustomNode init")
        # return super().init()
        return nndeploy.base.Status(nndeploy.base.StatusCode.Ok)

    def deinit(self):
        print("CustomNode deinit") 
        # return super().deinit()
        return nndeploy.base.Status(nndeploy.base.StatusCode.Ok)
        
    def run(self):
        # 实现自定义的run方法
        import torch
        print("CustomNode run start")
        input_edge_1 = self.get_input(0)
        input_edge_2 = self.get_input(1)
        add_result = input_edge_1.get() + input_edge_2.get()
        output_edge = self.get_output(0)
        if isinstance(output_edge, nndeploy.dag.Edge):
            print("output_edge is nndeploy.dag.Edge")
        elif isinstance(output_edge, _C.dag.Edge):
            print("output_edge is _C.dag.Edge")
        else:
            print("output_edge is not nndeploy.dag.Edge")
        output_edge.set(add_result)
        print("CustomNode run end")
        return nndeploy.base.Status(nndeploy.base.StatusCode.Ok)
      
      
class CustomGraph(nndeploy.dag.Graph):
    def __init__(self, name, inputs: list[_C.dag.Edge] = None, outputs: list[_C.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
        self.edge_1 = nndeploy.dag.Edge("edge_1", nndeploy.base.ParallelType.kParallelTypeNone)
        self.edge_2 = nndeploy.dag.Edge("edge_2", nndeploy.base.ParallelType.kParallelTypeNone)
        self.add_edge(self.edge_1)
        self.add_edge(self.edge_2)
        print("edge_1: ", self.edge_1)
        print("edge_2: ", self.edge_2)
        
        self.node_1 = CustomNode("node_1", inputs, [self.edge_1])
        self.add_node_shared_ptr(self.node_1)
        print("node_1: ", self.node_1)
        
        self.node_2 = CustomNode("node_2", inputs, [self.edge_2])
        self.add_node_shared_ptr(self.node_2)
        print("node_2: ", self.node_2)
        
        self.edge_3 = nndeploy.dag.Edge("edge_3", nndeploy.base.ParallelType.kParallelTypeNone)
        self.node_3 = CustomNode("node_3", [self.edge_1, self.edge_2], [self.edge_3])
        self.add_node_shared_ptr(self.node_3)
        print("node_3: ", self.node_3)
        
        self.edge_4 = nndeploy.dag.Edge("edge_4", nndeploy.base.ParallelType.kParallelTypeNone)
        self.node_4 = CustomNode("node_4", [self.edge_1, self.edge_2], [self.edge_4])
        self.add_node_shared_ptr(self.node_4)
        print("node_4: ", self.node_4)
        
        self.node_5 = CustomNode("node_5", [self.edge_3, self.edge_4], outputs)
        self.add_node_shared_ptr(self.node_5)
        print("node_5: ", self.node_5)
        
    
def test_graph():
    edge_1 = nndeploy.dag.Edge("test_edge_1", nndeploy.base.ParallelType.kParallelTypeNone)
    edge_2 = nndeploy.dag.Edge("test_edge_2", nndeploy.base.ParallelType.kParallelTypeNone)
    edge_3 = nndeploy.dag.Edge("test_edge_3", nndeploy.base.ParallelType.kParallelTypeNone)
    graph = CustomGraph("test_graph", [edge_1, edge_2], [edge_3])
    # graph.set_debug_flag(True)
    
    # print(graph.init())
    # graph.dump()
    
    # edge_1.set(torch.ones(1, 3, 64, 64))
    # edge_2.set(torch.ones(1, 3, 64, 64))
    # graph.run()
    # print(edge_3.get())
    
    # graph.deinit()
    
    edge_00 = nndeploy.dag.Edge("edge_00", nndeploy.base.ParallelType.kParallelTypeNone)
    edge_01 = nndeploy.dag.Edge("edge_01", nndeploy.base.ParallelType.kParallelTypeNone)
    edge_10 = nndeploy.dag.Edge("edge_10", nndeploy.base.ParallelType.kParallelTypeNone)
    edge_11 = nndeploy.dag.Edge("edge_11", nndeploy.base.ParallelType.kParallelTypeNone)
    big_graph = nndeploy.dag.Graph("big_graph", [edge_00, edge_01], [edge_3])
    
    node_0 = CustomNode("node_0", [edge_00, edge_01], [edge_1])
    big_graph.add_node_shared_ptr(node_0)
    
    node_1 = CustomNode("node_1", [edge_00, edge_01], [edge_2])
    big_graph.add_node_shared_ptr(node_1)
    
    big_graph.add_node_shared_ptr(graph)
    
    big_graph.set_debug_flag(True)
    
    print(big_graph.init())
    graph.dump()
    big_graph.dump()
    
    edge_00.set(torch.ones(1, 3, 64, 64))
    edge_01.set(torch.ones(1, 3, 64, 64))
    big_graph.run()
    print(edge_3.get())
    
    big_graph.deinit()
    
    
    
if __name__ == "__main__":
    test_graph()
    
    
        
        
        
        
