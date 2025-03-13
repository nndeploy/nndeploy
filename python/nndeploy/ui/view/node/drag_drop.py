"""
拖放操作模块

负责:
- 处理节点的拖放操作
- 管理拖放预览效果
- 处理节点的放置位置
- 支持批量拖放功能

拖动时节点半透明,放置位置显示辅助线
"""

from typing import Optional, Tuple, Callable
import flet as ft

class DragDropManager:
    """拖放管理器"""
    
    def __init__(
        self,
        on_drop: Optional[Callable[[str, float, float], None]] = None,
        grid_size: int = 20
    ):
        self.on_drop = on_drop
        self.grid_size = grid_size
        self._dragging = False
        self._drag_node: Optional[str] = None
        self._drag_pos: Optional[Tuple[float, float]] = None
        self._preview: Optional[ft.Control] = None
        
    def start_drag(
        self,
        node_type: str,
        x: float,
        y: float,
        preview: ft.Control
    ):
        """开始拖动
        
        Args:
            node_type: 节点类型
            x: 起始X坐标
            y: 起始Y坐标
            preview: 预览控件
        """
        self._dragging = True
        self._drag_node = node_type
        self._drag_pos = (x, y)
        self._preview = preview
        self._preview.opacity = 0.5
        
    def update_drag(self, x: float, y: float):
        """更新拖动位置
        
        Args:
            x: 当前X坐标
            y: 当前Y坐标
        """
        if not self._dragging:
            return
            
        # 对齐到网格
        x = round(x / self.grid_size) * self.grid_size
        y = round(y / self.grid_size) * self.grid_size
        
        # 更新预览位置
        if self._preview:
            self._preview.left = x
            self._preview.top = y
            self._preview.update()
            
    def end_drag(self, x: float, y: float):
        """结束拖动
        
        Args:
            x: 结束X坐标
            y: 结束Y坐标
        """
        if not self._dragging:
            return
            
        # 对齐到网格
        x = round(x / self.grid_size) * self.grid_size
        y = round(y / self.grid_size) * self.grid_size
        
        # 通知放置
        if self.on_drop and self._drag_node:
            self.on_drop(self._drag_node, x, y)
            
        # 清理状态
        self._dragging = False
        self._drag_node = None
        self._drag_pos = None
        if self._preview:
            self._preview.opacity = 1
            self._preview.update()
        self._preview = None
        
    def cancel_drag(self):
        """取消拖动"""
        if not self._dragging:
            return
            
        # 清理状态
        self._dragging = False
        self._drag_node = None
        self._drag_pos = None
        if self._preview:
            self._preview.opacity = 1
            self._preview.update()
        self._preview = None
        
    @property
    def is_dragging(self) -> bool:
        """是否正在拖动"""
        return self._dragging 