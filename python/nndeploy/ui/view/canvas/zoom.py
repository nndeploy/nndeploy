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
import flet.canvas

class ZoomControl:
    """缩放控制器"""
    
    def __init__(
        self,
        min_scale: float = 0.1,
        max_scale: float = 5.0,
        step: float = 0.1,
        on_scale_change: Optional[Callable[[float], None]] = None
    ):
        self.min_scale = min_scale  # 最小缩放比例
        self.max_scale = max_scale  # 最大缩放比例
        self.step = step  # 缩放步长
        self.on_scale_change = on_scale_change  # 缩放变化回调函数
        self._scale = 1.0  # 当前缩放比例
        self.control = self._build()  # 构建控件
        
    def _build(self) -> ft.Control:
        """构建缩放控制器界面"""
        return ft.Container(
            width=40,
            padding=5,
            bgcolor=ft.colors.BLACK12,
            border_radius=20,
            content=ft.Column(
                [
                    ft.IconButton(
                        icon=ft.Icons.ADD,
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
                        icon=ft.Icons.REMOVE,
                        icon_size=20,
                        on_click=lambda _: self.zoom_out()
                    ),
                    ft.IconButton(
                        icon=ft.Icons.CROP_FREE,
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
            
            # 更新显示的缩放百分比
            if isinstance(self.control.content, ft.Column):
                column = self.control.content
                if len(column.controls) > 1:
                    percentage_container = column.controls[1]
                    if isinstance(percentage_container, ft.Container) and isinstance(percentage_container.content, ft.Text):
                        percentage_container.content.value = f"{int(self._scale * 100)}%"
            
            # 通知缩放变化
            if self.on_scale_change:
                self.on_scale_change(scale)
                
            # 更新控件显示
            self.control.update()
            
    def on_wheel(self, e) -> bool:
        """处理鼠标滚轮事件
        
        Returns:
            是否处理了事件
        """
        # 检查滚轮事件是否有效
        if not hasattr(e, 'scroll_delta_y') or e.scroll_delta_y is None:
            return False
            
        # 确保delta_y是数值并且不为零
        try:
            print(e.scroll_delta_y)
            if abs(e.scroll_delta_y) < 0.001:  # 避免过小的值
                return False
                
            # 根据滚轮方向确定缩放方向
            delta = self.step if e.scroll_delta_y < 0 else -self.step
            self.set_scale(self._scale + delta)
            return True
        except (TypeError, AttributeError):
            return False
    

if __name__ == "__main__":
    def main(page: ft.Page):
        # 设置页面属性
        page.title = "缩放控制器演示"
        page.padding = 20
        page.bgcolor = "#f0f0f0"
        
        # 创建一个示例内容
        demo_content = ft.Container(
            width=400,
            height=300,
            bgcolor="#ffffff",
            border_radius=10,
            border=ft.border.all(1, "#cccccc"),
            padding=20,
            content=ft.Column([
                ft.Text("缩放演示", size=24, weight=ft.FontWeight.BOLD),
                ft.Container(height=10),
                ft.Text("这是一个可缩放的内容区域。使用右侧的缩放控制器或直接滚动鼠标滚轮来缩放此内容。"),
                ft.Container(height=20),
                ft.Row([
                    ft.Icon(ft.Icons.ZOOM_IN, size=40, color="#2563EB"),
                    ft.Container(width=10),
                    ft.Text("当前缩放比例:", size=16),
                    ft.Text("100%", size=16, weight=ft.FontWeight.BOLD, color="#2563EB"),
                ], alignment=ft.MainAxisAlignment.CENTER),
            ])
        )
        
        # 创建一个容器来模拟画布
        # 注意：ft.Container 不直接支持 on_wheel 事件，需要使用 GestureDetector 包装
        canvas = flet.GestureDetector(
            content=ft.Container(
                width=600,
                height=400,
                bgcolor="#e0e0e0",
                border_radius=5,
                content=demo_content,
                alignment=ft.alignment.center
            ),
            # on_scroll=on_canvas_wheel  # 使用 on_scroll 而不是 on_wheel
        )
        
        # 创建缩放控制器
        def on_scale_change(scale):
            # 更新内容的缩放
            demo_content.scale = scale
            demo_content.update()
            
            # 更新显示的缩放比例文本
            scale_text.value = f"{int(scale * 100)}%"
            scale_text.update()
            
        zoom_control = ZoomControl(
            min_scale=0.5,
            max_scale=2.0,
            step=0.1,
            on_scale_change=on_scale_change
        )
        
        # 获取缩放比例显示文本的引用
        scale_text = None
        for row in demo_content.content.controls:
            if isinstance(row, ft.Row):
                for control in row.controls:
                    if isinstance(control, ft.Text) and control.color == "#2563EB":
                        scale_text = control
                        break
        
        # 处理鼠标滚轮事件
        def on_canvas_wheel(e):
            # 确保事件对象有效
            print(e)
            if hasattr(e, 'scroll_delta_y') and e.scroll_delta_y is not None:
                print(e.scroll_delta_y)
                if zoom_control.on_wheel(e):
                    e.prevent_default = True
        
        canvas.on_scroll = on_canvas_wheel
        
        # 创建布局
        layout = ft.Row([
            canvas,
            ft.Container(width=20),  # 间距
            zoom_control.control,
        ], alignment=ft.MainAxisAlignment.CENTER)
        
        # 添加到页面
        page.add(
            ft.Container(
                content=ft.Text("缩放控制器演示", size=30, weight=ft.FontWeight.BOLD),
                margin=ft.margin.only(bottom=20)
            ),
            layout,
            ft.Container(
                content=ft.Text("提示: 直接滚动鼠标滚轮即可缩放", italic=True),
                margin=ft.margin.only(top=20)
            )
        )
    
    ft.app(target=main, view=ft.WEB_BROWSER, port=8080)
