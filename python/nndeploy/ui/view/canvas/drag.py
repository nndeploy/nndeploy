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
import flet.canvas  # 导入Flet画布模块

class Drag:
    """拖拽控制器
    
    用于管理画布的拖拽行为，包括拖拽状态跟踪、事件处理和回调通知。
    支持鼠标拖拽和键盘控制两种交互方式。
    """
    
    def __init__(
        self,
        on_drag: Optional[Callable[[float, float], None]] = None
    ):
        """初始化拖拽控制器
        
        Args:
            on_drag: 拖拽回调函数，接收dx和dy参数表示拖拽的偏移量
        """
        self.on_drag = on_drag  # 存储拖拽回调函数
        self._dragging = False  # 拖拽状态标志
        self._start_pos: Optional[Tuple[float, float]] = None  # 拖拽起始位置
        self._last_pos: Optional[Tuple[float, float]] = None  # 上次拖拽位置
        
    def on_pan_start(self, e: ft.DragStartEvent):
        """开始拖拽
        
        当用户按下鼠标并开始拖拽时触发
        
        Args:
            e: 拖拽开始事件对象
        """
        self._dragging = True  # 设置拖拽状态为真
        self._start_pos = (e.local_x, e.local_y)  # 记录起始位置
        self._last_pos = self._start_pos  # 初始化上次位置
        
        # 修改鼠标指针为移动样式
        e.control.cursor = ft.MouseCursor.MOVE
        e.control.update()  # 更新控件显示
        
    def on_pan_update(self, e: ft.DragUpdateEvent):
        """拖拽更新
        
        当用户拖拽鼠标移动时持续触发
        
        Args:
            e: 拖拽更新事件对象
        """
        if not self._dragging or not self._last_pos:
            return  # 如果未处于拖拽状态则退出
            
        # 计算相对于上次位置的偏移量
        dx = e.local_x - self._last_pos[0]  # 水平偏移
        dy = e.local_y - self._last_pos[1]  # 垂直偏移
        
        # 更新上次位置记录
        self._last_pos = (e.local_x, e.local_y)
        
        # 通知拖拽回调
        if self.on_drag:
            self.on_drag(dx, dy)  # 调用回调函数传递偏移量
            
    def on_pan_end(self, e: ft.DragEndEvent):
        """结束拖拽
        
        当用户释放鼠标结束拖拽时触发
        
        Args:
            e: 拖拽结束事件对象
        """
        self._dragging = False  # 重置拖拽状态
        self._start_pos = None  # 清除起始位置
        self._last_pos = None  # 清除上次位置
        
        # 恢复鼠标指针为默认样式
        e.control.cursor = ft.MouseCursor.BASIC
        e.control.update()  # 更新控件显示


# 以下是预览Demo代码
if __name__ == "__main__":
    def main(page: ft.Page):
        # 设置页面属性
        page.title = "画布拖拽演示"
        page.padding = 0
        page.bgcolor = "#f0f0f0"
        
        # 创建一个示例画布
        canvas_width = 800
        canvas_height = 600
        canvas = flet.canvas.Canvas(
            width=canvas_width,
            height=canvas_height,
            content=ft.Container(
                width=canvas_width,
                height=canvas_height,
                bgcolor="#ffffff",
                border=ft.border.all(1, "#cccccc")
            )
        )
        
        # 创建一些示例内容
        content_container = ft.Container(
            width=canvas_width,
            height=canvas_height,
            content=ft.Stack([
                # 添加一些示例元素
                ft.Container(
                    left=100, top=100,
                    width=150, height=100,
                    bgcolor="#2563EB",
                    border_radius=10,
                    padding=10,
                    content=ft.Text("拖拽画布可以移动视图", color="white")
                ),
                ft.Container(
                    left=400, top=200,
                    width=200, height=150,
                    bgcolor="#10B981",
                    border_radius=10,
                    padding=10,
                    content=ft.Text("使用空格键也可以临时启用拖拽", color="white")
                ),
            ])
        )
        
        # 创建拖拽控制器
        offset_x = 0
        offset_y = 0
        
        def on_canvas_drag(dx, dy):
            nonlocal offset_x, offset_y
            # 更新偏移量
            offset_x += dx
            offset_y += dy
            # 更新内容位置
            content_container.left = offset_x
            content_container.top = offset_y
            page.update()
        
        drag_controller = Drag(on_drag=on_canvas_drag)
        
        # 创建手势检测器
        canvas_container = ft.GestureDetector(
            # mouse_cursor=ft.MouseCursor.MOVE,
            drag_interval=16,  # 约60fps的更新率
            on_pan_start=drag_controller.on_pan_start,
            on_pan_update=drag_controller.on_pan_update,
            on_pan_end=drag_controller.on_pan_end,
            content=ft.Stack([
                ft.Container(
                    width=canvas_width,
                    height=canvas_height,
                    bgcolor="#f5f5f5"
                ),
                content_container
            ])
        )
        
        # 添加说明文本
        instructions = ft.Text(
            "拖拽画布移动视图，或按住空格键临时启用拖拽模式",
            size=16,
            weight=ft.FontWeight.BOLD,
            color="#333333"
        )
        
        # 创建页面布局
        page.add(
            ft.Column([
                instructions,
                canvas_container
            ])
        )

        
    ft.app(target=main, view=ft.WEB_BROWSER, port=9090)