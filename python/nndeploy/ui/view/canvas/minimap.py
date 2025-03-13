"""
小地图导航模块

负责:
- 提供工作流的缩略预览
- 显示当前视图区域
- 处理视图区域拖动
- 支持快速定位功能

采用半透明背景,节点简化显示,当前视图区域高亮
"""

from typing import Optional, Tuple, Callable
import flet as ft

class MinimapView(ft.UserControl):
    """小地图视图"""
    
    def __init__(
        self,
        width: float = 200,
        height: float = 150,
        on_view_change: Optional[Callable[[float, float], None]] = None
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.on_view_change = on_view_change
        
        # 视图状态
        self._scale = 1.0
        self._view_rect = (0, 0, width, height)
        self._dragging = False
        self._drag_start = None
        
    def build(self):
        return ft.Container(
            width=self.width,
            height=self.height,
            bgcolor=ft.colors.BLACK12,
            border_radius=5,
            padding=5,
            content=ft.Stack([
                # 节点预览层
                self._build_nodes_layer(),
                
                # 视图区域层
                self._build_view_layer(),
            ])
        )
        
    def _build_nodes_layer(self) -> ft.Control:
        """构建节点预览层"""
        return ft.Container(
            # TODO: 绘制简化的节点预览
        )
        
    def _build_view_layer(self) -> ft.Control:
        """构建视图区域层"""
        return ft.Container(
            left=self._view_rect[0],
            top=self._view_rect[1],
            width=self._view_rect[2],
            height=self._view_rect[3],
            border=ft.border.all(2, ft.colors.BLUE),
            border_radius=3,
            opacity=0.3,
            on_pan_start=self._on_view_drag_start,
            on_pan_update=self._on_view_drag_update,
            on_pan_end=self._on_view_drag_end,
        )
        
    def update_view(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        scale: float
    ):
        """更新视图区域
        
        Args:
            x: 视图左上角X坐标
            y: 视图左上角Y坐标
            width: 视图宽度
            height: 视图高度
            scale: 缩放比例
        """
        self._scale = scale
        self._view_rect = (x, y, width, height)
        self.update()
        
    def _on_view_drag_start(self, e: ft.DragStartEvent):
        """开始拖动视图区域"""
        self._dragging = True
        self._drag_start = (e.local_x, e.local_y)
        
    def _on_view_drag_update(self, e: ft.DragUpdateEvent):
        """拖动视图区域"""
        if not self._dragging or not self._drag_start:
            return
            
        # 计算偏移量
        dx = e.local_x - self._drag_start[0]
        dy = e.local_y - self._drag_start[1]
        
        # 更新视图位置
        x = max(0, min(self.width - self._view_rect[2],
                      self._view_rect[0] + dx))
        y = max(0, min(self.height - self._view_rect[3],
                      self._view_rect[1] + dy))
                      
        self._view_rect = (x, y, self._view_rect[2], self._view_rect[3])
        
        # 通知视图变化
        if self.on_view_change:
            self.on_view_change(x, y)
            
        self.update()
        
    def _on_view_drag_end(self, e: ft.DragEndEvent):
        """结束拖动视图区域"""
        self._dragging = False
        self._drag_start = None 