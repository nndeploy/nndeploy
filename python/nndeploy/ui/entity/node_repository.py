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
    PROCESS = "process"       # 处理节点
    MODEL = "model"          # 模型节点
    OUTPUT = "output"        # 输出节点
    CONTROL = "control"      # 控制流节点

class NodeType:
    """节点类型定义"""
    
    def __init__(
        self,
        id: str,
        name: str,
        category: NodeCategory,
        description: str,
        icon: str = None,
        inputs: List[Dict] = None,
        outputs: List[Dict] = None,
        parameters: List[Dict] = None
    ):
        self.id = id
        self.name = name
        self.category = category
        self.description = description
        self.icon = icon or "widgets"  # 默认图标
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.parameters = parameters or []
        
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "icon": self.icon,
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
            icon=data.get("icon"),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            parameters=data.get("parameters", [])
        )

class NodeRepository:
    """节点仓库类"""
    
    def __init__(self):
        self._nodes: Dict[str, NodeType] = {}
        self._load_builtin_nodes()
        self._load_custom_nodes()
        
    def _load_builtin_nodes(self):
        """加载内置节点类型"""
        builtin_nodes = [
            # 输入节点
            NodeType(
                id="text_input",
                name="文本输入",
                category=NodeCategory.INPUT,
                description="接收文本输入",
                outputs=[{"name": "text", "type": "str"}]
            ),
            NodeType(
                id="image_input",
                name="图像输入",
                category=NodeCategory.INPUT,
                description="接收图像输入",
                outputs=[{"name": "image", "type": "numpy.ndarray"}]
            ),
            
            # 处理节点
            NodeType(
                id="text_process",
                name="文本处理",
                category=NodeCategory.PROCESS,
                description="处理文本数据",
                inputs=[{"name": "text", "type": "str"}],
                outputs=[{"name": "processed", "type": "str"}],
                parameters=[
                    {"name": "operation", "type": "select", "options": ["大写", "小写", "分词"]}
                ]
            ),
            NodeType(
                id="image_process",
                name="图像处理",
                category=NodeCategory.PROCESS,
                description="处理图像数据",
                inputs=[{"name": "image", "type": "numpy.ndarray"}],
                outputs=[{"name": "processed", "type": "numpy.ndarray"}],
                parameters=[
                    {"name": "operation", "type": "select", "options": ["缩放", "旋转", "滤波"]}
                ]
            ),
            
            # 模型节点
            NodeType(
                id="llm_model",
                name="大语言模型",
                category=NodeCategory.MODEL,
                description="调用大语言模型",
                inputs=[{"name": "prompt", "type": "str"}],
                outputs=[{"name": "response", "type": "str"}],
                parameters=[
                    {"name": "model", "type": "select", "options": ["GPT-3.5", "GPT-4"]},
                    {"name": "temperature", "type": "float", "min": 0, "max": 1}
                ]
            ),
            
            # 输出节点
            NodeType(
                id="text_output",
                name="文本输出",
                category=NodeCategory.OUTPUT,
                description="显示文本输出",
                inputs=[{"name": "text", "type": "str"}]
            ),
            NodeType(
                id="image_output",
                name="图像输出",
                category=NodeCategory.OUTPUT,
                description="显示图像输出",
                inputs=[{"name": "image", "type": "numpy.ndarray"}]
            ),
            
            # 控制流节点
            NodeType(
                id="condition",
                name="条件判断",
                category=NodeCategory.CONTROL,
                description="条件分支控制",
                inputs=[{"name": "condition", "type": "bool"}],
                outputs=[
                    {"name": "true", "type": "any"},
                    {"name": "false", "type": "any"}
                ]
            ),
            NodeType(
                id="loop",
                name="循环",
                category=NodeCategory.CONTROL,
                description="循环控制",
                inputs=[{"name": "items", "type": "list"}],
                outputs=[{"name": "item", "type": "any"}],
                parameters=[
                    {"name": "max_iterations", "type": "int", "min": 1}
                ]
            )
        ]
        
        for node in builtin_nodes:
            self._nodes[node.id] = node
            
    def _load_custom_nodes(self):
        """加载自定义节点类型"""
        custom_nodes_path = Path(os.path.dirname(__file__)) / "../config/custom_nodes.json"
        
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
        custom_nodes_path = Path(os.path.dirname(__file__)) / "../config/custom_nodes.json"
        
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