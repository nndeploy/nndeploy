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