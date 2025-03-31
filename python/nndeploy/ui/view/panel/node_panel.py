"""
节点面板模块

负责:
- 展示可用的节点列表
- 管理节点分类和搜索
- 处理节点的拖放操作
- 提供节点预览功能

节点以卡片形式展示,支持分类折叠和搜索过滤
"""

"""
节点仓库模块

负责:
- 管理所有可用的节点类型
- 提供节点类型的注册机制
- 支持节点的分类管理
- 处理节点的元数据

以分类树形式展示,支持搜索和收藏常用节点
"""

from typing import Dict, List, Optional, Set
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
        inputs: List = None,
        outputs: List = None,
        parameters: List = None
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
        builtin_nodes = []
        
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

from typing import Dict, List, Optional, Callable
import flet as ft
from ...config.language import get_text
from ..node.node_types import NodeType, NodeCategory, BUILTIN_NODE_TYPES

class NodeCard(ft.UserControl):
    """节点卡片"""
    
    def __init__(
        self,
        node_type: NodeType,
        on_drag_start: Optional[Callable[[str, float, float], None]] = None
    ):
        super().__init__()
        self.node_type = node_type
        self.on_drag_start = on_drag_start
        
    def build(self):
        return ft.GestureDetector(
            content=ft.Container(
                content=ft.Column(
                    [
                        # 图标和名称
                        ft.Row(
                            [
                                ft.Icon(
                                    name=self.node_type.icon,
                                    color=self.node_type.color
                                ),
                                ft.Text(
                                    self.node_type.name,
                                    size=14,
                                    weight=ft.FontWeight.BOLD
                                )
                            ],
                            spacing=10
                        ),
                        
                        # 描述
                        ft.Text(
                            self.node_type.description,
                            size=12,
                            color=ft.colors.BLACK54,
                            max_lines=2,
                            overflow=ft.TextOverflow.ELLIPSIS
                        )
                    ],
                    spacing=5
                ),
                padding=10,
                bgcolor=ft.colors.WHITE,
                border_radius=5,
                shadow=ft.BoxShadow(
                    spread_radius=1,
                    blur_radius=1,
                    color=ft.colors.BLACK12
                )
            ),
            mouse_cursor=ft.MouseCursor.MOVE,
            on_pan_start=self._on_drag_start
        )
        
    def _on_drag_start(self, e: ft.DragStartEvent):
        """开始拖动"""
        if self.on_drag_start:
            self.on_drag_start(self.node_type.id, e.local_x, e.local_y)

class NodeCategory(ft.UserControl):
    """节点分类组"""
    
    def __init__(
        self,
        category: str,
        nodes: List[NodeType],
        on_node_drag_start: Optional[Callable[[str, float, float], None]] = None
    ):
        super().__init__()
        self.category = category
        self.nodes = nodes
        self.on_node_drag_start = on_node_drag_start
        self._expanded = True
        
    def build(self):
        return ft.Column(
            [
                # 分类标题
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Icon(
                                name=ft.icons.EXPAND_MORE if self._expanded
                                else ft.icons.CHEVRON_RIGHT,
                                size=20
                            ),
                            ft.Text(
                                get_text(f"node.category.{self.category}"),
                                size=16,
                                weight=ft.FontWeight.BOLD
                            )
                        ],
                        spacing=5
                    ),
                    padding=ft.padding.only(left=10, right=10),
                    on_click=self._toggle_expand
                ),
                
                # 节点列表
                ft.AnimatedSwitcher(
                    content=ft.Column(
                        [
                            NodeCard(
                                node_type=node,
                                on_drag_start=self.on_node_drag_start
                            )
                            for node in self.nodes
                        ],
                        spacing=10,
                        scroll=ft.ScrollMode.AUTO
                    ) if self._expanded else None,
                    transition=ft.AnimatedSwitcherTransition.SCALE,
                    duration=300,
                    switch_in_curve=ft.AnimationCurve.EASE_OUT,
                    switch_out_curve=ft.AnimationCurve.EASE_IN
                )
            ],
            spacing=10
        )
        
    def _toggle_expand(self, _):
        """切换展开状态"""
        self._expanded = not self._expanded
        self.update()

class NodePanel(ft.UserControl):
    """节点面板"""
    
    def __init__(
        self,
        on_node_drag_start: Optional[Callable[[str, float, float], None]] = None
    ):
        super().__init__()
        self.on_node_drag_start = on_node_drag_start
        self._search_text = ""
        self._nodes = BUILTIN_NODE_TYPES
        
    def build(self):
        return ft.Column(
            [
                # 搜索框
                ft.TextField(
                    prefix_icon=ft.icons.SEARCH,
                    hint_text=get_text("node.search"),
                    on_change=self._on_search_change
                ),
                
                # 节点分类列表
                ft.Column(
                    [
                        NodeCategory(
                            category=category.value,
                            nodes=self._filter_nodes(category),
                            on_node_drag_start=self.on_node_drag_start
                        )
                        for category in NodeCategory
                    ],
                    spacing=20,
                    scroll=ft.ScrollMode.AUTO
                )
            ],
            spacing=20
        )
        
    def _on_search_change(self, e):
        """搜索文本变化"""
        self._search_text = e.control.value.lower()
        self.update()
        
    def _filter_nodes(self, category: NodeCategory) -> List[NodeType]:
        """过滤节点列表
        
        Args:
            category: 节点类别
            
        Returns:
            过滤后的节点列表
        """
        return [
            node for node in self._nodes.values()
            if node.category == category and (
                not self._search_text or
                self._search_text in node.name.lower() or
                self._search_text in node.description.lower()
            )
        ]
        
    def add_node_type(self, node_type: NodeType):
        """添加节点类型"""
        self._nodes[node_type.id] = node_type
        self.update()
        
    def remove_node_type(self, node_id: str):
        """移除节点类型"""
        if node_id in self._nodes:
            del self._nodes[node_id]
            self.update() 