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
        icon: str = None,
        inputs: List[] = None,
        outputs: List[] = None,
        parameters: List[] = None
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