"""
画布拖拽模块

负责:
- 处理画布的平移操作
- 管理拖拽状态
- 处理键盘和鼠标事件
- 提供平滑的动画效果

拖动时鼠标指针变为手形,有平滑的动画效果
"""

from typing import Optional, Tuple, Callable
import flet as ft

class CanvasDrag:
    """画布拖拽控制器"""
    
    def __init__(
        self,
        on_drag: Optional[Callable[[float, float], None]] = None
    ):
        self.on_drag = on_drag
        self._dragging = False
        self._start_pos: Optional[Tuple[float, float]] = None
        self._last_pos: Optional[Tuple[float, float]] = None
        
    def on_pan_start(self, e: ft.DragStartEvent):
        """开始拖拽"""
        self._dragging = True
        self._start_pos = (e.local_x, e.local_y)
        self._last_pos = self._start_pos
        
        # 修改鼠标指针
        e.control.cursor = ft.CursorStyle.MOVE
        e.control.update()
        
    def on_pan_update(self, e: ft.DragUpdateEvent):
        """拖拽更新"""
        if not self._dragging or not self._last_pos:
            return
            
        # 计算偏移量
        dx = e.local_x - self._last_pos[0]
        dy = e.local_y - self._last_pos[1]
        
        # 更新位置
        self._last_pos = (e.local_x, e.local_y)
        
        # 通知拖拽回调
        if self.on_drag:
            self.on_drag(dx, dy)
            
    def on_pan_end(self, e: ft.DragEndEvent):
        """结束拖拽"""
        self._dragging = False
        self._start_pos = None
        self._last_pos = None
        
        # 恢复鼠标指针
        e.control.cursor = ft.CursorStyle.DEFAULT
        e.control.update()
        
    def on_keyboard(self, e: ft.KeyboardEvent) -> bool:
        """处理键盘事件
        
        Returns:
            是否处理了事件
        """
        if e.key == "Space":
            # 空格键临时启用拖拽模式
            if e.type == ft.KeyboardEventType.KEY_DOWN:
                e.control.cursor = ft.CursorStyle.MOVE
            else:
                e.control.cursor = ft.CursorStyle.DEFAULT
            e.control.update()
            return True
            
        return False 