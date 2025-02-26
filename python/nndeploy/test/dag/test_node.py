
"""
函数形式Op
"""

import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag
import torch


# python3 nndeploy/test/dag/test_node.py


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
        # print(input_edge_1.get())
        # print(input_edge_2.get())
        add_result = input_edge_1.get() + input_edge_2.get()
        output_edge = self.get_output(0)
        # 确保输出边是nndeploy.dag.Edge类型
        # output_edge = nndeploy.dag.Edge.from_internal_edge(output_edge)
        if isinstance(output_edge, nndeploy.dag.Edge):
            print("output_edge is nndeploy.dag.Edge")
        elif isinstance(output_edge, _C.dag.Edge):
            print("output_edge is _C.dag.Edge")
        else:
            print("output_edge is not nndeploy.dag.Edge")
        output_edge.set(add_result)
        print("CustomNode run end")
        return nndeploy.base.Status(nndeploy.base.StatusCode.Ok)


def test_node():  
    # 使用边指针创建节点
    input_edge_1 = nndeploy.dag.Edge("input_1")
    input_edge_2 = nndeploy.dag.Edge("input_2")
    output_edge = nndeploy.dag.Edge("output")
    node = CustomNode("test", [input_edge_1, input_edge_2], [output_edge])
    print(node.init())
    import torch
    input_edge_1.set(torch.ones(1, 3, 64, 64))
    print(input_edge_1.get_type_name())
    input_edge_2.set(torch.ones(1, 3, 64, 64))
    print(input_edge_2.get_type_name())
    node.run()
    # print(node.check_inputs([input_edge_1, input_edge_2]))
    # print(output_edge.get())
    # print(output_edge.get_type_name())    
    # node.deinit()
    
    edge_list = node.get_all_input()
    print(edge_list)
    
    output_edge_v2 = node.test([input_edge_1, input_edge_2], ["output"])
    print("end!!!")
    print(output_edge_v2)
    print(output_edge_v2[0].get_type_name())
    print(output_edge_v2[0].get())
    
    # input_edge_3 = nndeploy.dag.Edge("input_1")
    # input_edge_4 = nndeploy.dag.Edge("input_2")
    # output_edge_1 = nndeploy.dag.Edge("output")
    # input_edge_3.set(torch.ones(1, 3, 64, 64))
    # print(input_edge_1.get_type_name())
    # input_edge_4.set(torch.zeros(1, 3, 64, 64))
    # print(input_edge_4.get_type_name())
    # node.set_inputs([input_edge_3, input_edge_4])
    # node.set_outputs([output_edge_1])
    
    # node.run()
    
    # print(output_edge_1.get_type_name())
    # print(output_edge_1.get())


if __name__ == "__main__":
    test_node()

# 
# export LD_LIBRARY_PATH=/home/ascenduserdg01/github/nndeploy/build:$LD_LIBRARY_PATH
