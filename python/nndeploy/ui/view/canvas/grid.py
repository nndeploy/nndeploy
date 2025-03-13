"""
画布网格组件

负责:
- 绘制画布背景网格
- 管理网格显示和隐藏
- 处理网格大小调整
- 提供网格对齐功能
"""

from typing import Optional, Tuple
import flet as ft
from nndeploy.ui.config import get_color, get_style, settings

class Grid(ft.UserControl):
    """画布网格组件"""
    
    def __init__(
        self,
        width: float,
        height: float,
        cell_size: int = 20,
        enabled: bool = True,
        color: Optional[str] = None,
        opacity: float = 0.2,
        snap_to_grid: bool = True
    ):
        """初始化网格组件
        
        Args:
            width: 画布宽度
            height: 画布高度 
            cell_size: 网格单元格大小(像素)
            enabled: 是否启用网格
            color: 网格线颜色,默认使用主题色
            opacity: 网格线透明度(0-1)
            snap_to_grid: 是否启用网格对齐
        """
        super().__init__()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.enabled = enabled
        self.color = color or get_color("secondary")
        self.opacity = opacity
        self.snap_to_grid = snap_to_grid
        
        # 缓存网格线坐标
        self._horizontal_lines: List[Tuple[float, float, float, float]] = []
        self._vertical_lines: List[Tuple[float, float, float, float]] = []
        self._calculate_grid_lines()
        
    def _calculate_grid_lines(self):
        """计算网格线坐标"""
        # 水平线
        self._horizontal_lines = [
            (0, y, self.width, y)
            for y in range(0, int(self.height), self.cell_size)
        ]
        
        # 垂直线
        self._vertical_lines = [
            (x, 0, x, self.height)
            for x in range(0, int(self.width), self.cell_size)
        ]
        
    def build(self):
        """构建网格组件"""
        if not self.enabled:
            return ft.Container(width=self.width, height=self.height)
            
        return ft.Stack(
            [
                # 背景
                ft.Container(
                    width=self.width,
                    height=self.height,
                    bgcolor=get_color("background")
                ),
                
                # 网格线
                ft.CustomPaint(
                    paint=ft.Paint(
                        stroke_width=1,
                        stroke_color=self.color,
                        style=ft.PaintingStyle.STROKE,
                        opacity=self.opacity
                    ),
                    size=(self.width, self.height),
                    on_paint=self._paint_grid
                )
            ]
        )
        
    def _paint_grid(self, canvas: ft.Canvas):
        """绘制网格线
        
        Args:
            canvas: Flet画布对象
        """
        # 绘制水平线
        for x1, y1, x2, y2 in self._horizontal_lines:
            canvas.draw_line(x1, y1, x2, y2)
            
        # 绘制垂直线
        for x1, y1, x2, y2 in self._vertical_lines:
            canvas.draw_line(x1, y1, x2, y2)
            
    def resize(self, width: float, height: float):
        """调整网格大小
        
        Args:
            width: 新宽度
            height: 新高度
        """
        self.width = width
        self.height = height
        self._calculate_grid_lines()
        self.update()
        
    def set_cell_size(self, size: int):
        """设置网格单元格大小
        
        Args:
            size: 单元格大小(像素)
        """
        self.cell_size = max(10, min(100, size))  # 限制大小范围
        self._calculate_grid_lines()
        self.update()
        
    def toggle(self, enabled: Optional[bool] = None):
        """切换网格显示状态
        
        Args:
            enabled: 是否显示网格,None表示切换当前状态
        """
        if enabled is None:
            enabled = not self.enabled
        self.enabled = enabled
        self.update()
        
    def snap_point(self, x: float, y: float) -> Tuple[float, float]:
        """将坐标对齐到网格
        
        Args:
            x: X坐标
            y: Y坐标
            
        Returns:
            对齐后的坐标元组(x, y)
        """
        if not self.snap_to_grid:
            return x, y
            
        # 计算最近的网格线
        x = round(x / self.cell_size) * self.cell_size
        y = round(y / self.cell_size) * self.cell_size
        
        return x, y
        
    def get_cell_rect(self, row: int, col: int) -> Tuple[float, float, float, float]:
        """获取指定网格单元格的矩形区域
        
        Args:
            row: 行号
            col: 列号
            
        Returns:
            单元格矩形区域(x, y, width, height)
        """
        x = col * self.cell_size
        y = row * self.cell_size
        return x, y, self.cell_size, self.cell_size
        
    def get_cell_at(self, x: float, y: float) -> Tuple[int, int]:
        """获取指定坐标所在的网格单元格
        
        Args:
            x: X坐标
            y: Y坐标
            
        Returns:
            单元格位置(row, col)
        """
        row = int(y / self.cell_size)
        col = int(x / self.cell_size)
        return row, col
