"""
小地图导航模块

负责:
- 提供工作流的缩略预览
- 显示当前视图区域
- 处理视图区域拖动
- 支持快速定位功能

采用半透明背景,节点简化显示,当前视图区域高亮
"""

from typing import Optional, Tuple, Callable, List, Dict
import flet as ft

class Minimap:
    """小地图视图"""
    
    def __init__(
        self,
        page: ft.Page,
        width: float = 200,
        height: float = 150,
        on_view_change: Optional[Callable[[float, float], None]] = None
    ):
        """初始化小地图
        
        Args:
            page: Flet页面对象
            width: 小地图宽度
            height: 小地图高度
            on_view_change: 视图变化回调函数
        """
        self.page = page
        self.width = width
        self.height = height
        self.on_view_change = on_view_change
        
        # 视图状态
        self._scale = 1.0
        self._view_rect = (0, 0, width, height)
        self._dragging = False
        self._drag_start = None
        
        # 节点列表
        self.nodes = []
        
        # 构建控件
        self.container = self._build()
        
    def _build(self) -> ft.Container:
        """构建小地图控件"""
        # 创建节点预览层
        self.nodes_layer = ft.Container(
            width=self.width,
            height=self.height,
            content=ft.Stack([])  # 初始化为空Stack以便后续添加节点
        )
        
        # 创建视图区域层
        self.view_layer = ft.Container(
            width=self._view_rect[2],
            height=self._view_rect[3],
            border=ft.border.all(2, ft.colors.BLUE),
            border_radius=3,
            opacity=0.3,
        )
        
        # 创建手势检测器用于拖动视图区域
        self.gesture_detector = ft.GestureDetector(
            mouse_cursor=ft.MouseCursor.MOVE,
            drag_interval=10,
            on_pan_start=self._on_view_drag_start,
            on_pan_update=self._on_view_drag_update,
            on_pan_end=self._on_view_drag_end,
            content=self.view_layer
        )
        
        # 创建堆叠布局
        stack = ft.Stack([
            self.nodes_layer,
            ft.Container(
                left=self._view_rect[0],
                top=self._view_rect[1],
                content=self.gesture_detector
            )
        ])
        
        # 创建主容器
        return ft.Container(
            width=self.width,
            height=self.height,
            bgcolor=ft.colors.BLACK12,
            border_radius=5,
            padding=5,
            content=stack
        )
        
    def add_node(self, node_type: str, x: float, y: float, color: str):
        """添加节点到小地图
        
        Args:
            node_type: 节点类型
            x: 节点X坐标(相对于画布)
            y: 节点Y坐标(相对于画布)
            color: 节点颜色
        """
        # 计算节点在小地图上的位置
        # 这里需要根据实际画布大小进行缩放
        map_x = (x / 2000) * self.width  # 使用固定画布宽度2000
        map_y = (y / 1500) * self.height  # 使用固定画布高度1500
        
        # 创建简化的节点表示
        node = ft.Container(
            width=10,
            height=10,
            left=map_x,
            top=map_y,
            bgcolor=color,
            border_radius=5,
        )
        
        # 添加到节点列表和显示层
        self.nodes.append({
            "type": node_type,
            "control": node,
            "x": x,
            "y": y,
            "color": color
        })
        
        # 添加节点到Stack
        self.nodes_layer.content.controls.append(node)
        
        self.update()
        
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
        
        # 计算视图区域在小地图上的位置和大小
        map_x = (x / 2000) * self.width
        map_y = (y / 1500) * self.height
        map_width = (width / 2000) * self.width
        map_height = (height / 1500) * self.height
        
        self._view_rect = (map_x, map_y, map_width, map_height)
        
        # 更新视图区域控件
        self.gesture_detector.content.width = map_width
        self.gesture_detector.content.height = map_height
        
        if isinstance(self.gesture_detector.parent, ft.Container):
            self.gesture_detector.parent.left = map_x
            self.gesture_detector.parent.top = map_y
        
        self.update()
        
    def update(self):
        """更新小地图显示"""
        if self.container.page:
            self.container.update()
        
    def _on_view_drag_start(self, e):
        """开始拖动视图区域"""
        self._dragging = True
        self._drag_start = (e.local_x, e.local_y)
        
    def _on_view_drag_update(self, e):
        """拖动视图区域"""
        if not self._dragging or not self._drag_start:
            return
            
        # 计算偏移量
        dx = e.delta_x
        dy = e.delta_y
        
        # 更新视图位置
        x = max(0, min(self.width - self._view_rect[2],
                      self.view_layer.left + dx))
        y = max(0, min(self.height - self._view_rect[3],
                      self.view_layer.top + dy))
                      
        self.view_layer.left = x
        self.view_layer.top = y
        self._view_rect = (x, y, self._view_rect[2], self._view_rect[3])
        
        # 通知视图变化
        if self.on_view_change:
            # 转换回画布坐标
            canvas_x = (x / self.width) * 2000  # 使用固定画布宽度2000
            canvas_y = (y / self.height) * 1500  # 使用固定画布高度1500
            self.on_view_change(canvas_x, canvas_y)
            
        self.update()
        
    def _on_view_drag_end(self, e):
        """结束拖动视图区域"""
        self._dragging = False
        self._drag_start = None


# 演示代码
if __name__ == "__main__":
    def main(page: ft.Page):
        # 设置页面属性
        page.title = "小地图导航演示"
        page.padding = 0
        page.bgcolor = "#f0f0f0"
        
        # 创建网格对象(模拟)
        grid_offset_x = 0
        grid_offset_y = 0
        
        # 创建小地图
        def on_view_change(x, y):
            nonlocal grid_offset_x, grid_offset_y
            # 更新网格偏移量(反向移动以实现导航效果)
            grid_offset_x = -x
            grid_offset_y = -y
            
            # 更新网格位置显示
            grid_container.left = grid_offset_x
            grid_container.top = grid_offset_y
            grid_container.update()
            
            # 更新坐标信息
            coords_text.value = f"画布偏移: ({grid_offset_x:.1f}, {grid_offset_y:.1f})"
            coords_text.update()
        
        minimap = Minimap(
            page=page,
            width=200,
            height=150,
            on_view_change=on_view_change
        )
        
        # 创建模拟画布内容
        grid_container = ft.Container(
            width=2000,  # 模拟大画布
            height=1500,
            left=grid_offset_x,
            top=grid_offset_y,
            content=ft.Stack([])
        )
        
        # 添加一些节点到画布
        nodes_data = [
            {"type": "start", "x": 200, "y": 200, "color": "#4CAF50", "size": 60},
            {"type": "llm", "x": 500, "y": 300, "color": "#2196F3", "size": 100},
            {"type": "search", "x": 800, "y": 200, "color": "#FF9800", "size": 80},
            {"type": "end", "x": 1200, "y": 400, "color": "#F44336", "size": 60},
            {"type": "process", "x": 700, "y": 600, "color": "#9C27B0", "size": 80},
        ]
        
        # 添加节点到画布和小地图
        for node in nodes_data:
            # 添加到画布
            canvas_node = ft.Container(
                width=node["size"],
                height=node["size"],
                left=node["x"],
                top=node["y"],
                bgcolor=node["color"],
                border_radius=10,
                alignment=ft.alignment.center,
                content=ft.Text(node["type"], color=ft.colors.WHITE)
            )
            grid_container.content.controls.append(canvas_node)
            
            # 添加到小地图
            minimap.add_node(node["type"], node["x"], node["y"], node["color"])
        
        # 创建坐标信息显示
        coords_text = ft.Text(
            f"画布偏移: ({grid_offset_x:.1f}, {grid_offset_y:.1f})",
            color=ft.colors.BLACK,
            size=14,
        )
        
        # 创建主布局
        main_view = ft.Stack([
            grid_container,  # 模拟画布
            ft.Container(  # 小地图容器
                content=minimap.container,
                right=20,
                top=20,
            ),
            ft.Container(  # 坐标信息容器
                content=coords_text,
                left=20,
                top=20,
                bgcolor=ft.colors.WHITE70,
                padding=10,
                border_radius=5,
            )
        ])
        
        # 添加到页面
        page.add(main_view)
        
        # 初始化视图区域
        minimap.update_view(0, 0, page.width, page.height, 1.0)
        
    ft.app(target=main, view=ft.WEB_BROWSER, port=9090)