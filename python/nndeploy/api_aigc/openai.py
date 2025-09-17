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
        self.openai_api_key = "" # 需要用户自己设置
        self.openai_model = "gpt-4o-mini" # 需要用户自己设置
        self.openai_temperature = 0.5 # 需要用户自己设置
        self.openai_max_tokens = 1024 # 需要用户自己设置
        self.openai_top_p = 1.0 # 需要用户自己设置
        self.openai_frequency_penalty = 0.0 # 需要用户自己设置
        self.openai_presence_penalty = 0.0 # 需要用户自己设置
        self.openai_stop = None # 需要用户自己设置
                
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
        json_obj["openai_api_key"] = self.openai_api_key
        json_obj["openai_model"] = self.openai_model
        json_obj["openai_temperature"] = self.openai_temperature
        json_obj["openai_max_tokens"] = self.openai_max_tokens
        json_obj["openai_top_p"] = self.openai_top_p
        json_obj["openai_frequency_penalty"] = self.openai_frequency_penalty
        json_obj["openai_presence_penalty"] = self.openai_presence_penalty
        json_obj["openai_stop"] = self.openai_stop
        return json.dumps(json_obj)

    def deserialize(self, target: str):
        json_obj = json.loads(target)
        self.frontend_show_param = json_obj["frontend_show_param"] # 示例参数frontend_show_param，从前端获得前端调整的参数
        self.openai_api_key = json_obj["openai_api_key"]
        self.openai_model = json_obj["openai_model"]
        self.openai_temperature = json_obj["openai_temperature"]
        self.openai_max_tokens = json_obj["openai_max_tokens"]
        self.openai_top_p = json_obj["openai_top_p"]
        self.openai_frequency_penalty = json_obj["openai_frequency_penalty"]
        self.openai_presence_penalty = json_obj["openai_presence_penalty"]
        self.openai_stop = json_obj["openai_stop"]
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