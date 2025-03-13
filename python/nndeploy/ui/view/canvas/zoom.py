"""
缩放控制模块

负责:
- 处理画布的缩放操作
- 管理缩放级别
- 提供缩放控制界面
- 支持鼠标滚轮缩放

缩放控件采用半透明设计,不干扰画布内容
"""

from typing import Optional, Callable
import flet as ft

class ZoomControl(ft.UserControl):
    """缩放控制器"""
    
    def __init__(
        self,
        min_scale: float = 0.1,
        max_scale: float = 5.0,
        step: float = 0.1,
        on_scale_change: Optional[Callable[[float], None]] = None
    ):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.step = step
        self.on_scale_change = on_scale_change
        self._scale = 1.0
        
    def build(self):
        return ft.Container(
            width=40,
            padding=5,
            bgcolor=ft.colors.BLACK12,
            border_radius=20,
            content=ft.Column(
                [
                    ft.IconButton(
                        icon=ft.icons.ADD,
                        icon_size=20,
                        on_click=lambda _: self.zoom_in()
                    ),
                    ft.Container(
                        content=ft.Text(
                            f"{int(self._scale * 100)}%",
                            size=12,
                            text_align=ft.TextAlign.CENTER
                        ),
                        margin=ft.margin.symmetric(vertical=5)
                    ),
                    ft.IconButton(
                        icon=ft.icons.REMOVE,
                        icon_size=20,
                        on_click=lambda _: self.zoom_out()
                    ),
                    ft.IconButton(
                        icon=ft.icons.CROP_FREE,
                        icon_size=20,
                        on_click=lambda _: self.reset_zoom()
                    ),
                ],
                spacing=0,
                alignment=ft.MainAxisAlignment.CENTER,
            )
        )
        
    def zoom_in(self):
        """放大"""
        self.set_scale(self._scale + self.step)
        
    def zoom_out(self):
        """缩小"""
        self.set_scale(self._scale - self.step)
        
    def reset_zoom(self):
        """重置缩放"""
        self.set_scale(1.0)
        
    def set_scale(self, scale: float):
        """设置缩放比例"""
        # 限制缩放范围
        scale = max(self.min_scale, min(self.max_scale, scale))
        
        if scale != self._scale:
            self._scale = scale
            
            # 通知缩放变化
            if self.on_scale_change:
                self.on_scale_change(scale)
                
            self.update()
            
    def on_wheel(self, e: ft.MouseEvent) -> bool:
        """处理鼠标滚轮事件
        
        Returns:
            是否处理了事件
        """
        if e.ctrl:
            # Ctrl + 滚轮缩放
            delta = self.step if e.delta_y < 0 else -self.step
            self.set_scale(self._scale + delta)
            return True
            
        return False 