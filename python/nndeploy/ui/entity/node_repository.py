"""
节点仓库模块

负责:
- 管理所有可用的节点类型
- 提供节点类型的注册机制
- 支持节点的分类管理
- 处理节点的元数据

以分类树形式展示,支持搜索和收藏常用节点
"""

from typing import Dict, List, Optional
from enum import Enum
import json
import os
from pathlib import Path

class NodeCategory(Enum):
    """节点类别"""
    INPUT = "input"           # 输入节点
    OUTPUT = "output"        # 输出节点
    CONTROL = "control"      # 控制流节点
    CODEC = "codec"          # 编码解码节点
    UTIL = "util"            # 工具节点
    PREPROCESS = "preprocess" # 预处理节点
    TOKENIZER = "tokenizer"   # 分词器节点
    POSTPROCESS = "postprocess" # 后处理节点
    SAMPLER = "sampler"   # 采样器节点
    INFER = "infer"   # 推理节点
    DETECT = "detect"   # 检测节点
    CLASSIFY = "classify"   # 分类节点
    SEGMENT = "segment"   # 分割节点
    LLM = "llm"   # 大语言模型节点
    SD = "sd"   # 稳定扩散节点
    VAE = "vae"   # 变量自动编码器节点
    DIFFUSION = "diffusion"   # 扩散模型节点
    CLIP = "clip"   # CLIP节点

class NodeType:
    """节点类型定义"""
    
    def __init__(
        self,
        id: str, # 节点类型
        name: str,
        category: NodeCategory,
        description: str,
        tags: Set[str] = None,
        inputs: List[] = None,
        outputs: List[] = None,
        parameters: List[] = None
    ):
        # id: 节点类型的唯一标识符，用于在系统中唯一引用该节点类型
        # 不同于name，id通常是系统生成的，不会随用户操作而改变
        self.id = id
        
        # name: 节点类型的显示名称，用于在UI界面上展示，可由用户自定义
        # 与id不同，name可以重复，主要用于用户识别和搜索
        self.name = name
        
        # 节点类别（输入、输出、控制流等）
        self.category = category
        
        # 节点类型的详细描述信息
        self.description = description
        
        # tags: 节点类型的标签集合，用于分类和筛选
        # 一个节点类型可以有多个标签，用于对节点进行分组和快速查找
        self.tags = tags or set()
        
        # 节点的输入端口定义列表，描述节点可接收的输入类型
        self.inputs = inputs or []
        
        # 节点的输出端口定义列表，描述节点可产生的输出类型
        self.outputs = outputs or []
        
        # 节点的可配置参数列表，描述节点的可调整属性
        self.parameters = parameters or []
        
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "tags": self.tags,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "parameters": self.parameters
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'NodeType':
        """从字典创建实例"""
        return cls(
            id=data["id"],
            name=data["name"],
            category=NodeCategory(data["category"]),
            description=data["description"],
            tags=data.get("tags", set()),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            parameters=data.get("parameters", [])
        )
        
    def set_input(self, input, index = -1):
        """设置输入端口"""
        if index == -1:
            self.inputs.append(input)
        else:
            # 确保索引在有效范围内
            if 0 <= index <= len(self.inputs):
                self.inputs.insert(index, input)
            else:
                # 如果索引无效，则追加到末尾
                self.inputs.append(input)

    def set_output(self, output, index = -1):
        """设置输出端口"""
        if index == -1:
            self.outputs.append(output)
        else:
            # 确保索引在有效范围内
            if 0 <= index <= len(self.outputs):
                self.outputs.insert(index, output)
            else:
                # 如果索引无效，则追加到末尾
                self.outputs.append(output)
            
    def set_parameter(self, parameter, index = -1):
        """设置参数"""
        if index == -1:
            self.parameters.append(parameter)
        else:
            # 确保索引在有效范围内
            if 0 <= index <= len(self.parameters):
                self.parameters.insert(index, parameter)
            else:
                # 如果索引无效，则追加到末尾
                self.parameters.append(parameter)
            
class NodeRepository:
    """节点仓库类"""
    
    def __init__(self):
        self._nodes: Dict[str, NodeType] = {}
        self._load_builtin_nodes()
        self._load_custom_nodes()
        
    def _load_builtin_nodes(self):
        """加载内置节点类型"""
        
        for node in builtin_nodes:
            self._nodes[node.id] = node
            
    def _load_custom_nodes(self):
        """加载自定义节点类型"""
        custom_nodes_path = Path(os.path.dirname(__file__)) / "../assets/custom_nodes.json"
        
        if custom_nodes_path.exists():
            try:
                with open(custom_nodes_path, "r", encoding="utf-8") as f:
                    custom_nodes = json.load(f)
                    for node_data in custom_nodes:
                        node = NodeType.from_dict(node_data)
                        self._nodes[node.id] = node
            except Exception as e:
                print(f"加载自定义节点失败: {e}")
                
    def get_node(self, node_id: str) -> Optional[NodeType]:
        """获取节点类型定义"""
        return self._nodes.get(node_id)
        
    def get_nodes_by_category(self, category: NodeCategory) -> List[NodeType]:
        """获取指定类别的所有节点"""
        return [
            node for node in self._nodes.values()
            if node.category == category
        ]
        
    def register_node(self, node: NodeType):
        """注册新的节点类型"""
        if node.id in self._nodes:
            raise ValueError(f"节点类型 {node.id} 已存在")
        self._nodes[node.id] = node
        self._save_custom_nodes()
        
    def _save_custom_nodes(self):
        """保存自定义节点配置"""
        custom_nodes_path = Path(os.path.dirname(__file__)) / "../assets/custom_nodes.json"
        
        # 过滤出自定义节点(非内置节点)
        custom_nodes = [
            node.to_dict() for node in self._nodes.values()
            if not node.id.startswith("builtin_")
        ]
        
        try:
            with open(custom_nodes_path, "w", encoding="utf-8") as f:
                json.dump(custom_nodes, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"保存自定义节点失败: {e}")

# 创建全局节点仓库实例
node_repository = NodeRepository() 