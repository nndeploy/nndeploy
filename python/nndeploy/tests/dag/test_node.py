
"""
函数形式Op
"""

import nndeploy._nndeploy_internal as _C

class CustomNode(_C.dag.Node):
    def __init__(self, name):
        super().__init__(name)
    
    # def init(self):
    #     print("CustomNode init")
    #     return _C.base.Status

    # def deinit(self):
    #     print("CustomNode deinit") 
    #     return _C.base.Status
        
    def run(self):
        # 实现自定义的run方法
        print("CustomNode run")
        # return base.kStatusCodeOk  # 返回适当的状态码
        return _C.base.Status


def test_node():
    # 创建输入输出边指针
    # input_edge = _C.dag.EdgePtr("input")
    # output_edge = _C.dag.EdgePtr("output")
    
    # 使用边指针创建节点
    node = CustomNode("test")
    node.init()
    node.run()
    node.deinit()


if __name__ == "__main__":
    test_node()

# 
# export LD_LIBRARY_PATH=/home/ascenduserdg01/github/nndeploy/build:$LD_LIBRARY_PATH
