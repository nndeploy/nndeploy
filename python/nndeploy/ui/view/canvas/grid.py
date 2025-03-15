"""
画布网格组件

负责:
- 绘制画布背景网格
- 支持无限画布功能，允许用户向任何方向拖拉扩展画布
"""

from typing import Optional, Tuple, List
import flet as ft
import flet.canvas
from nndeploy.ui.config import get_color, get_style, settings

class Grid:
    """画布网格组件
    
    - 绘制画布背景网格
    - 支持无限画布功能，允许用户向任何方向拖拉扩展画布
    """
    
    def __init__(
        self,
        page: ft.Page,
        width: float,
        height: float,
        cell_size: int = 20,
        color: Optional[str] = None,
        opacity: float = 0.3,
    ):
        """初始化网格组件
        
        Args:
            page: Flet页面对象，用于更新UI
            width: 画布宽度，决定网格的水平范围
            height: 画布高度，决定网格的垂直范围
            cell_size: 网格单元格大小(像素)，控制网格的密度
            color: 网格线颜色，默认使用主题的secondary颜色
            opacity: 网格线透明度(0-1)，控制网格线的可见度
        """
        # 存储网格基本属性
        self.page = page
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.color = color or get_color("secondary")
        self.opacity = opacity
        
        # 无限画布相关属性
        self.offset_x = 0  # 画布X轴偏移量
        self.offset_y = 0  # 画布Y轴偏移量
        self.is_dragging = False  # 是否正在拖动画布
        self.drag_start_x = 0  # 拖动起始X坐标
        self.drag_start_y = 0  # 拖动起始Y坐标
        
        # 创建控件
        self.canvas = ft.canvas.Canvas(
            width=self.width,
            height=self.height,
            shapes=[]
        )
        
        # 绘制初始网格
        self._draw_grid()
        
        # 创建可拖动的容器
        self.container = ft.GestureDetector(
            mouse_cursor=ft.MouseCursor.MOVE,
            on_pan_start=self._on_pan_start,
            on_pan_update=self._on_pan_update,
            on_pan_end=self._on_pan_end,
            content=ft.Stack([
                # 背景层 - 纯色背景
                ft.Container(
                    width=self.width,
                    height=self.height,
                    bgcolor=get_color("background")
                ),
                # 网格线层
                self.canvas
            ])
        )
        
    def _draw_grid(self):
        """绘制网格线
        
        根据当前的宽度、高度、单元格大小和偏移量，绘制所有水平和垂直网格线。
        考虑画布偏移量，实现无限画布效果。
        """
        # 清除现有网格线
        self.canvas.shapes.clear()
        
        # 计算可见区域的起始和结束坐标（考虑偏移量）
        start_x = -self.offset_x % self.cell_size
        start_y = -self.offset_y % self.cell_size
        
        # 绘制水平线
        y = start_y
        while y < self.height:
            self.canvas.shapes.append(
                ft.canvas.Line(
                    x1=0, y1=y, x2=self.width, y2=y,
                    paint=ft.Paint(
                        color=self.color,
                        stroke_width=1
                    )
                )
            )
            y += self.cell_size
        
        # 绘制垂直线
        x = start_x
        while x < self.width:
            self.canvas.shapes.append(
                ft.canvas.Line(
                    x1=x, y1=0, x2=x, y2=self.height,
                    paint=ft.Paint(
                        color=self.color,
                        stroke_width=1
                    )
                )
            )
            x += self.cell_size
        
    def _on_pan_start(self, e):
        """开始拖动画布
        
        记录拖动起始位置
        
        Args:
            e: 拖动事件对象
        """
        self.is_dragging = True
        self.drag_start_x = e.local_x
        self.drag_start_y = e.local_y
    
    def _on_pan_update(self, e):
        """更新画布拖动
        
        根据拖动距离更新画布偏移量
        
        Args:
            e: 拖动事件对象
        """
        if not self.is_dragging:
            return
            
        # 计算拖动距离
        delta_x = e.local_x - self.drag_start_x
        delta_y = e.local_y - self.drag_start_y
        
        # 更新偏移量
        self.offset_x += delta_x
        self.offset_y += delta_y
        
        # 更新拖动起始位置
        self.drag_start_x = e.local_x
        self.drag_start_y = e.local_y
        
        # 重新绘制网格线
        self._draw_grid()
        
        # 更新显示
        self.canvas.update()
    
    def _on_pan_end(self, e):
        """结束画布拖动
        
        重置拖动状态
        
        Args:
            e: 拖动事件对象
        """
        self.is_dragging = False
            
    def resize(self, width: float, height: float):
        """调整网格大小
        
        当画布大小变化时调用此方法，重新计算网格线并更新显示。
        
        Args:
            width: 新宽度，单位为像素
            height: 新高度，单位为像素
        """
        # 更新尺寸属性
        self.width = width
        self.height = height
        
        # 更新画布尺寸
        self.canvas.width = width
        self.canvas.height = height
        
        # 更新背景层尺寸
        if isinstance(self.container.content, ft.Stack) and len(self.container.content.controls) > 0:
            background = self.container.content.controls[0]
            if isinstance(background, ft.Container):
                background.width = width
                background.height = height
        
        # 重新绘制网格线
        self._draw_grid()
        
        # 更新显示
        self.canvas.update()
        self.container.content.update()


if __name__ == "__main__":
    def main(page: ft.Page):
        # 设置页面属性
        page.title = "无限画布测试"
        page.padding = 0
        page.bgcolor = get_color("background")
        
        # 获取窗口尺寸
        page.window_width = 1200
        page.window_height = 800
        page.window_resizable = True
        
        # 创建满屏网格
        grid = Grid(page, width=page.window_width, height=page.window_height)
        
        # 添加参考点标记
        center_mark = ft.Container(
            width=10,
            height=10,
            bgcolor=ft.colors.RED,
            border_radius=5,
            left=(page.window_width / 2) - 5,
            top=(page.window_height / 2) - 5,
        )
        
        # 添加坐标信息文本
        coords_text = ft.Text(
            f"偏移量: (0, 0)",
            color=ft.colors.WHITE,
            bgcolor=ft.colors.BLACK54,
            size=14,
            left=10,
            top=10,
        )
        
        # 更新坐标信息
        def update_coords():
            coords_text.value = f"偏移量: ({grid.offset_x:.1f}, {grid.offset_y:.1f})"
            coords_text.update()
            
        # 监听拖动事件
        original_on_pan_update = grid._on_pan_update
        def on_pan_update_with_coords(e):
            original_on_pan_update(e)
            update_coords()
        grid._on_pan_update = on_pan_update_with_coords
        
        # 创建布局
        layout = ft.Stack([
            grid.container,
            center_mark,
            coords_text,
        ])
        
        # 监听窗口大小变化
        def on_resize(e):
            grid.resize(page.window_width, page.window_height)
            center_mark.left = (page.window_width / 2) - 5
            center_mark.top = (page.window_height / 2) - 5
            center_mark.update()
        
        page.on_resize = on_resize
        page.add(layout)
    
    ft.app(target=main, view=ft.WEB_BROWSER, port=9090)
