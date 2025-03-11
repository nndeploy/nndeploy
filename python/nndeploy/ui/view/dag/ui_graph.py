import flet
import nndeploy.dag
from nndeploy.dag import NodeWrapper, EdgeWrapper

from flet.canvas import Canvas, Path, Line


class UiGraph(flet.GestureDetector):
    def __init__(self, page, graph=None):
        super().__init__()
        self.page = page
        self.graph = graph
        self.ui_nodes: list[UiNode] = []
        self.ui_edges: list[UiEdge] = []
        self.ui_nodes_map: dict[NodeWrapper, UiNode] = {}
        self.ui_edges_map: dict[EdgeWrapper, UiEdge] = {}
        
        # 初始化缩放和平移参数
        self.scale = 1.0  # 缩放比例
        self.offset_x = 0  # 水平偏移量
        self.offset_y = 0  # 垂直偏移量
        self.grid_size = 50  # 网格基础大小
        
        # 设置手势事件处理
        self.on_pan_start = self._on_pan_start
        self.on_pan_update = self._on_pan_update
        self.on_scroll = self._on_scroll
        
        # 创建画布
        self.canvas = Canvas()
        self.canvas.shapes = self._create_background_grid()
        
    def _on_pan_start(self, e):
        # 记录拖拽起始位置
        self.start_x = e.local_x
        self.start_y = e.local_y
        
    def _on_pan_update(self, e):
        # 更新偏移量
        delta_x = e.local_x - self.start_x
        delta_y = e.local_y - self.start_y
        self.offset_x += delta_x
        self.offset_y += delta_y
        self.start_x = e.local_x
        self.start_y = e.local_y
        
        # 更新网格
        self.canvas.shapes = self._create_background_grid()
        self.canvas.update()
        
    def _on_scroll(self, e):
        # 获取鼠标位置
        mouse_x = e.local_x
        mouse_y = e.local_y
        
        # 保存缩放前的相对位置
        old_scale = self.scale
        
        # 处理缩放
        if e.scroll_delta_y < 0:
            self.scale *= 1.1  # 放大
        else:
            self.scale *= 0.9  # 缩小
        self.scale = max(0.1, min(5.0, self.scale))  # 限制缩放范围
        
        # 调整偏移量以保持鼠标位置不变
        scale_factor = self.scale / old_scale
        self.offset_x = mouse_x - (mouse_x - self.offset_x) * scale_factor
        self.offset_y = mouse_y - (mouse_y - self.offset_y) * scale_factor
        
        # 更新网格
        self.canvas.shapes = self._create_background_grid()
        self.canvas.update()
        
    def _create_background_grid(self):
        # 创建水平和垂直网格线
        grid_lines = []
        scaled_grid_size = self.grid_size * self.scale  # 网格大小随缩放变化
        
        # 计算可见区域的网格线范围
        visible_width = self.page.width
        visible_height = self.page.height
        
        # 计算需要绘制的网格范围
        left = -visible_width - abs(self.offset_x)
        right = visible_width * 2 + abs(self.offset_x)
        top = -visible_height - abs(self.offset_y)
        bottom = visible_height * 2 + abs(self.offset_y)
        
        # 计算网格线的起始和结束索引
        start_x = int(left / scaled_grid_size) - 1
        end_x = int(right / scaled_grid_size) + 1
        start_y = int(top / scaled_grid_size) - 1
        end_y = int(bottom / scaled_grid_size) + 1
        
        # 水平线
        for y in range(start_y, end_y + 1):
            grid_y = y * scaled_grid_size + self.offset_y
            grid_lines.append(
                Line(
                    x1=left, y1=grid_y,
                    x2=right, y2=grid_y,
                    paint=flet.Paint(
                        color=flet.colors.with_opacity(0.1, flet.colors.WHITE),
                        stroke_width=1,
                    )
                )
            )
            
        # 垂直线    
        for x in range(start_x, end_x + 1):
            grid_x = x * scaled_grid_size + self.offset_x
            grid_lines.append(
                Line(
                    x1=grid_x, y1=top,
                    x2=grid_x, y2=bottom,
                    paint=flet.Paint(
                        color=flet.colors.with_opacity(0.1, flet.colors.WHITE),
                        stroke_width=1,
                    )
                )
            )
            
        return grid_lines
        
    def build(self):
        return flet.Stack([
            self.canvas,
            flet.Container(
                content=flet.Row(
                    controls=[
                        flet.Text("Hello, World!"),
                    ]
                )
            ),
        ])