"""
节点类型定义模块

负责:
- 实现各种功能节点
- 定义节点的输入输出
- 管理节点的处理逻辑
- 提供节点的配置接口

不同类型节点采用不同配色方案,图标直观表示功能
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class NodeCategory(Enum):
    """节点类别"""
    INPUT = "input"       # 输入节点
    PROCESS = "process"   # 处理节点
    MODEL = "model"       # 模型节点
    OUTPUT = "output"     # 输出节点
    CONTROL = "control"   # 控制节点

@dataclass
class PortDefinition:
    """端口定义"""
    name: str                    # 端口名称
    port_type: str              # 端口类型
    description: str = ""       # 端口描述
    required: bool = True       # 是否必需
    multiple: bool = False      # 是否允许多个连接
    default_value: Any = None   # 默认值

@dataclass
class NodeType:
    """节点类型定义"""
    id: str                                  # 节点类型ID
    name: str                                # 节点名称
    category: NodeCategory                   # 节点类别
    description: str = ""                    # 节点描述
    icon: str = "widgets"                    # 节点图标
    color: str = "#666666"                  # 节点颜色
    inputs: List[PortDefinition] = field(default_factory=list)   # 输入端口
    outputs: List[PortDefinition] = field(default_factory=list)  # 输出端口
    config: Dict[str, Dict] = field(default_factory=dict)        # 配置字段
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "icon": self.icon,
            "color": self.color,
            "inputs": [
                {
                    "name": port.name,
                    "type": port.port_type,
                    "description": port.description,
                    "required": port.required,
                    "multiple": port.multiple,
                    "default": port.default_value
                }
                for port in self.inputs
            ],
            "outputs": [
                {
                    "name": port.name,
                    "type": port.port_type,
                    "description": port.description,
                    "required": port.required,
                    "multiple": port.multiple,
                    "default": port.default_value
                }
                for port in self.outputs
            ],
            "config": self.config
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'NodeType':
        """从字典创建实例"""
        return cls(
            id=data["id"],
            name=data["name"],
            category=NodeCategory(data["category"]),
            description=data.get("description", ""),
            icon=data.get("icon", "widgets"),
            color=data.get("color", "#666666"),
            inputs=[
                PortDefinition(
                    name=p["name"],
                    port_type=p["type"],
                    description=p.get("description", ""),
                    required=p.get("required", True),
                    multiple=p.get("multiple", False),
                    default_value=p.get("default")
                )
                for p in data.get("inputs", [])
            ],
            outputs=[
                PortDefinition(
                    name=p["name"],
                    port_type=p["type"],
                    description=p.get("description", ""),
                    required=p.get("required", True),
                    multiple=p.get("multiple", False),
                    default_value=p.get("default")
                )
                for p in data.get("outputs", [])
            ],
            config=data.get("config", {})
        )

# 预定义的节点类型
BUILTIN_NODE_TYPES = {
    # 输入节点
    "image_input": NodeType(
        id="image_input",
        name="图像输入",
        category=NodeCategory.INPUT,
        description="从文件加载图像",
        icon="image",
        color="#2196F3",
        outputs=[
            PortDefinition(
                name="image",
                port_type="image",
                description="输出图像"
            )
        ],
        config={
            "path": {
                "type": "string",
                "label": "文件路径",
                "default": ""
            }
        }
    ),
    
    # 处理节点
    "image_resize": NodeType(
        id="image_resize",
        name="图像缩放",
        category=NodeCategory.PROCESS,
        description="调整图像大小",
        icon="photo_size_select_large",
        color="#4CAF50",
        inputs=[
            PortDefinition(
                name="image",
                port_type="image",
                description="输入图像"
            )
        ],
        outputs=[
            PortDefinition(
                name="image",
                port_type="image",
                description="输出图像"
            )
        ],
        config={
            "width": {
                "type": "number",
                "label": "宽度",
                "min": 1,
                "default": 224
            },
            "height": {
                "type": "number",
                "label": "高度",
                "min": 1,
                "default": 224
            }
        }
    ),
    
    # 模型节点
    "classification": NodeType(
        id="classification",
        name="图像分类",
        category=NodeCategory.MODEL,
        description="使用深度学习模型进行图像分类",
        icon="category",
        color="#9C27B0",
        inputs=[
            PortDefinition(
                name="image",
                port_type="image",
                description="输入图像"
            )
        ],
        outputs=[
            PortDefinition(
                name="class",
                port_type="string",
                description="分类结果"
            ),
            PortDefinition(
                name="confidence",
                port_type="number",
                description="置信度"
            )
        ],
        config={
            "model": {
                "type": "select",
                "label": "模型",
                "options": {
                    "resnet50": "ResNet-50",
                    "mobilenet": "MobileNet"
                },
                "default": "resnet50"
            }
        }
    )
} 