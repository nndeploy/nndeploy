"""
连线组件模块

负责:
- 连接节点间的数据流
- 管理连线的样式和类型
- 处理数据的传递
- 提供连线的交互功能

采用贝塞尔曲线,支持不同类型的连线样式
"""

from typing import Tuple, Optional, Callable
import flet as ft
import math

"""
端口组件模块

负责:
- 定义节点的输入输出端口
- 管理端口的数据类型
- 处理端口的连接状态
- 提供端口的交互功能

端口采用半圆形设计,颜色根据数据类型区分
"""

from typing import Optional, Callable, Dict, Any
import flet as ft

class PortType:
    """端口类型"""
    ANY = "any"          # 任意类型
    NUMBER = "number"    # 数值类型
    STRING = "string"    # 字符串类型
    BOOLEAN = "boolean"  # 布尔类型
    ARRAY = "array"      # 数组类型
    OBJECT = "object"    # 对象类型
    IMAGE = "image"      # 图像类型
    AUDIO = "audio"      # 音频类型
    VIDEO = "video"      # 视频类型
    
    @staticmethod
    def can_connect(source_type: str, target_type: str) -> bool:
        """检查端口类型是否可以连接"""
        if source_type == PortType.ANY or target_type == PortType.ANY:
            return True
        return source_type == target_type

class PortDirection:
    """端口方向"""
    INPUT = "input"   # 输入端口
    OUTPUT = "output" # 输出端口

class Port(ft.UserControl):
    """端口组件"""
    
    # 端口类型对应的颜色
    TYPE_COLORS = {
        PortType.ANY: "#666666",
        PortType.NUMBER: "#2196F3",
        PortType.STRING: "#4CAF50",
        PortType.BOOLEAN: "#FF9800",
        PortType.ARRAY: "#9C27B0",
        PortType.OBJECT: "#795548",
        PortType.IMAGE: "#E91E63",
        PortType.AUDIO: "#00BCD4",
        PortType.VIDEO: "#FF5722"
    }
    
    def __init__(
        self,
        id: str,
        name: str,
        direction: str,
        port_type: str = PortType.ANY,
        on_connect: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[], None]] = None,
        on_drag_start: Optional[Callable[[float, float], None]] = None,
        on_drag_update: Optional[Callable[[float, float], None]] = None,
        on_drag_end: Optional[Callable[[], None]] = None
    ):
        super().__init__()
        self.id = id
        self.name = name
        self.direction = direction
        self.port_type = port_type
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_drag_start = on_drag_start
        self.on_drag_update = on_drag_update
        self.on_drag_end = on_drag_end
        
        # 连接状态
        self._connected = False
        self._dragging = False
        self._hover = False
        
    def build(self):
        return ft.GestureDetector(
            content=ft.Container(
                content=ft.Row(
                    [
                        # 端口图标
                        ft.Container(
                            width=12,
                            height=12,
                            border_radius=6,
                            bgcolor=self.TYPE_COLORS.get(
                                self.port_type,
                                self.TYPE_COLORS[PortType.ANY]
                            ),
                            border=ft.border.all(
                                2,
                                ft.colors.with_opacity(
                                    0.5 if self._hover else 0.3,
                                    ft.colors.WHITE
                                )
                            ) if self._connected else None,
                            animate=ft.animation.Animation(300, "easeOut")
                        ),
                        
                        # 端口名称
                        ft.Text(
                            self.name,
                            size=12,
                            color=ft.colors.BLACK54,
                            opacity=1.0 if self._hover else 0.7,
                            animate_opacity=300
                        )
                    ],
                    spacing=5,
                    alignment=ft.MainAxisAlignment.START
                    if self.direction == PortDirection.INPUT
                    else ft.MainAxisAlignment.END
                ),
                padding=ft.padding.all(5),
                border_radius=3,
                bgcolor=ft.colors.with_opacity(
                    0.1 if self._hover else 0,
                    ft.colors.BLACK
                ),
                animate=ft.animation.Animation(300, "easeOut")
            ),
            mouse_cursor=ft.MouseCursor.POINTER,
            on_hover=self._on_hover,
            on_pan_start=self._on_drag_start,
            on_pan_update=self._on_drag_update,
            on_pan_end=self._on_drag_end
        )
        
    def _on_hover(self, e: ft.HoverEvent):
        """鼠标悬停处理"""
        self._hover = e.hovered
        self.update()
        
    def _on_drag_start(self, e: ft.DragStartEvent):
        """开始拖动"""
        if self._connected and self.direction == PortDirection.INPUT:
            # 输入端口已连接时不允许拖动
            return
            
        self._dragging = True
        if self.on_drag_start:
            self.on_drag_start(e.local_x, e.local_y)
            
    def _on_drag_update(self, e: ft.DragUpdateEvent):
        """拖动更新"""
        if not self._dragging:
            return
            
        if self.on_drag_update:
            self.on_drag_update(e.local_x, e.local_y)
            
    def _on_drag_end(self, e: ft.DragEndEvent):
        """结束拖动"""
        if not self._dragging:
            return
            
        self._dragging = False
        if self.on_drag_end:
            self.on_drag_end()
            
    def set_connected(self, connected: bool):
        """设置连接状态"""
        if self._connected == connected:
            return
            
        self._connected = connected
        if connected:
            if self.on_connect:
                self.on_connect()
        else:
            if self.on_disconnect:
                self.on_disconnect()
                
        self.update()
        
    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected
        
    def can_connect_to(self, other: 'Port') -> bool:
        """检查是否可以连接到目标端口
        
        Args:
            other: 目标端口
            
        Returns:
            是否可以连接
        """
        # 检查方向
        if self.direction == other.direction:
            return False
            
        # 检查类型
        return PortType.can_connect(self.port_type, other.port_type) 

class Edge(ft.UserControl):
    """连线组件"""
    
    def __init__(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        color: str = "#666666",
        width: float = 2,
        selected: bool = False,
        on_select: Optional[Callable[[], None]] = None
    ):
        super().__init__()
        self.start = start
        self.end = end
        self.color = color
        self.width = width
        self.selected = selected
        self.on_select = on_select
        
    def build(self):
        # 计算控制点
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        
        cp1 = (self.start[0] + dx * 0.5, self.start[1])
        cp2 = (self.end[0] - dx * 0.5, self.end[1])
        
        # 构建路径
        path = ft.Path(
            data=[
                ft.PathMove(*self.start),
                ft.PathCubicBezier(
                    *cp1,
                    *cp2,
                    *self.end
                )
            ],
            stroke=self.color,
            stroke_width=self.width + (2 if self.selected else 0),
            selected=self.selected
        )
        
        # 添加点击区域
        return ft.GestureDetector(
            content=path,
            on_tap=lambda _: self.on_select and self.on_select()
        )
        
    def update_position(
        self,
        start: Optional[Tuple[float, float]] = None,
        end: Optional[Tuple[float, float]] = None
    ):
        """更新位置
        
        Args:
            start: 起点坐标
            end: 终点坐标
        """
        if start:
            self.start = start
        if end:
            self.end = end
        self.update()
        
    def set_selected(self, selected: bool):
        """设置选中状态"""
        self.selected = selected
        self.update()

class EdgeBuilder:
    """连线构建器"""
    
    def __init__(
        self,
        on_complete: Optional[Callable[[str, str], None]] = None
    ):
        self.on_complete = on_complete
        self._building = False
        self._start_port: Optional[str] = None
        self._preview: Optional[Edge] = None
        
    def start_edge(self, port_id: str, x: float, y: float):
        """开始构建连线
        
        Args:
            port_id: 起始端口ID
            x: 起始X坐标
            y: 起始Y坐标
        """
        self._building = True
        self._start_port = port_id
        self._preview = Edge(
            start=(x, y),
            end=(x, y),
            color="#999999",
            width=1
        )
        
    def update_edge(self, x: float, y: float):
        """更新连线位置
        
        Args:
            x: 当前X坐标
            y: 当前Y坐标
        """
        if not self._building or not self._preview:
            return
            
        self._preview.update_position(end=(x, y))
        
    def complete_edge(self, port_id: str):
        """完成连线
        
        Args:
            port_id: 目标端口ID
        """
        if not self._building or not self._start_port:
            return
            
        # 通知完成
        if self.on_complete:
            self.on_complete(self._start_port, port_id)
            
        # 清理状态
        self._building = False
        self._start_port = None
        if self._preview:
            self._preview.update()
        self._preview = None
        
    def cancel_edge(self):
        """取消连线"""
        if not self._building:
            return
            
        # 清理状态
        self._building = False
        self._start_port = None
        if self._preview:
            self._preview.update()
        self._preview = None
        
    @property
    def is_building(self) -> bool:
        """是否正在构建连线"""
        return self._building 