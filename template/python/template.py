import numpy as np
import json

import nndeploy.dag

class TemplatePy(nndeploy.dag.Node):
    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs) 
        super().set_key("nndeploy.template_py.TemplatePy") # 该节点的标志
        super().set_desc("用户自定义节点") # 对于该节点的描述
        self.set_input_type(np.ndarray) # 节点的输入类型
        self.set_output_type(np.ndarray) # 节点的输出类型
        self.frontend_show_param = 0.0 # 前端需要展示的参数，比如数组、bool值、字符串、list、dict等等，前端会根据数据类型选择合适的UI组件
                
    def run(self):
        input_edge = self.get_input(0) # 获取输入的edge
        input_numpy = input_edge.get(self) # 获取输入的numpy
        gray = np.dot(input_numpy[...,:3], [0.114, 0.587, 0.299])# bgr->gray
        gray = gray.astype(np.uint8)
        output_edge = self.get_output(0) # 获取输出边
        output_edge.set(gray) # 将输出写入到输出边中
        return nndeploy.base.Status.ok()
    
    def serialize(self):
        # 如果由需要前端展示的参数，就需要重写serialize和deserialize方法
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["frontend_show_param"] = self.frontend_show_param # 示例参数frontend_show_param，前端会根据数据类型选择合适的UI组件
        return json.dumps(json_obj)

    def deserialize(self, target: str):
        json_obj = json.loads(target)
        self.frontend_show_param = json_obj["frontend_show_param"] # 示例参数frontend_show_param，从前端获得前端调整的参数
        return super().deserialize(target)

# 节点创建器类，用于创建TemplatePy节点实例
class TemplatePyCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        # 创建并返回TemplatePy节点实例
        self.node = TemplatePy(name, inputs, outputs)
        return self.node

# 创建节点创建器实例
template_py_node_creator = TemplatePyCreator()
# 第一个参数是节点的唯一标识符，需要与节点类中set_key()设置的标识符保持一致，第二个参数是节点创建器实例，用于创建该类型的节点
nndeploy.dag.register_node("nndeploy.template_py.TemplatePy", template_py_node_creator)