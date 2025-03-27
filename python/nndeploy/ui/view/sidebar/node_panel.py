"""
节点面板模块

负责:
- 展示可用的节点列表
- 管理节点分类和搜索
- 处理节点的拖放操作
- 提供节点预览功能

节点以卡片形式展示,支持分类折叠和搜索过滤
"""

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