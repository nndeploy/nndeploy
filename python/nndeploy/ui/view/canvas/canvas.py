"""
画布组件

负责:
- 管理工作流节点和连线
- 处理节点拖放和连线操作
- 管理画布状态(缩放、平移等)
- 提供撤销/重做功能
"""

from typing import Optional, Dict, List, Tuple, Callable
import flet as ft
from nndeploy.ui.config import get_color, get_style, settings
from .grid import Grid
from .drag import DragManager
from .selection import SelectionManager
from .history import HistoryManager
from .zoom import ZoomManager

class Canvas(ft.UserControl):
    """画布组件"""
    
    def __init__(
        self,
        on_selection_change: Optional[Callable[[List[str]], None]] = None,
        on_node_double_click: Optional[Callable[[str], None]] = None
    ):
        """初始化画布组件
        
        Args:
            on_selection_change: 选择变化回调
            on_node_double_click: 节点双击回调
        """
        super().__init__()
        
        # 回调函数
        self.on_selection_change = on_selection_change
        self.on_node_double_click = on_node_double_click
        
        # 画布尺寸
        self.width = settings.get("canvas", "width", 3000)
        self.height = settings.get("canvas", "height", 2000)
        
        # 视口位置和大小
        self._viewport_x = 0
        self._viewport_y = 0
        self._viewport_width = 0
        self._viewport_height = 0
        
        # 初始化管理器
        self._init_managers()
        
        # 节点和连线数据
        self._nodes: Dict[str, Dict] = {}  # 节点数据
        self._edges: Dict[str, Dict] = {}  # 连线数据
        
    def _init_managers(self):
        """初始化各个管理器"""
        # 网格管理
        self.grid = Grid(
            width=self.width,
            height=self.height,
            cell_size=settings.get("canvas", "grid_size", 20),
            enabled=settings.get("canvas", "show_grid", True)
        )
        
        # 拖拽管理
        self.drag_manager = DragManager(
            on_drag_start=self._on_drag_start,
            on_drag_update=self._on_drag_update,
            on_drag_end=self._on_drag_end
        )
        
        # 选择管理
        self.selection_manager = SelectionManager(
            on_selection_change=self._on_selection_change
        )
        
        # 历史管理
        self.history_manager = HistoryManager()
        
        # 缩放管理
        self.zoom_manager = ZoomManager(
            min_scale=0.1,
            max_scale=2.0,
            on_zoom_change=self._on_zoom_change
        )
        
    def build(self):
        """构建画布组件"""
        return ft.GestureDetector(
            content=ft.Stack(
                [
                    # 网格层
                    self.grid,
                    
                    # 连线层
                    self._build_edges_layer(),
                    
                    # 节点层
                    self._build_nodes_layer(),
                    
                    # 选择层
                    self.selection_manager,
                    
                    # 拖拽层
                    self.drag_manager,
                ],
                clip_behavior=ft.ClipBehavior.HARD_EDGE
            ),
            on_pan_start=self._on_pan_start,
            on_pan_update=self._on_pan_update,
            on_pan_end=self._on_pan_end,
            on_scale_start=self._on_scale_start,
            on_scale_update=self._on_scale_update,
            on_scale_end=self._on_scale_end,
        )
        
    def _build_edges_layer(self) -> ft.Control:
        """构建连线层"""
        return ft.CustomPaint(
            paint=ft.Paint(
                style=ft.PaintingStyle.STROKE,
                stroke_width=2,
                stroke_cap=ft.StrokeCap.ROUND,
            ),
            size=(self.width, self.height),
            on_paint=self._paint_edges
        )
        
    def _build_nodes_layer(self) -> ft.Control:
        """构建节点层"""
        return ft.Stack(
            [self._build_node(node_id, data) for node_id, data in self._nodes.items()]
        )
        
    def _build_node(self, node_id: str, data: Dict) -> ft.Control:
        """构建节点组件
        
        Args:
            node_id: 节点ID
            data: 节点数据
            
        Returns:
            节点控件
        """
        return ft.Container(
            content=ft.Column(
                [
                    # 标题栏
                    ft.Container(
                        content=ft.Text(data["title"]),
                        bgcolor=get_color(f"node.{data['type']}.title"),
                        padding=5,
                    ),
                    
                    # 内容区
                    ft.Container(
                        content=ft.Text(data["content"]),
                        bgcolor=get_color(f"node.{data['type']}.content"),
                        padding=10,
                    )
                ]
            ),
            left=data["x"],
            top=data["y"],
            bgcolor=get_color(f"node.{data['type']}.background"),
            border=ft.border.all(1, get_color("border")),
            border_radius=5,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=3,
                color=ft.colors.with_opacity(0.3, ft.colors.BLACK),
            )
        )
        
    def _paint_edges(self, canvas: ft.Canvas):
        """绘制连线
        
        Args:
            canvas: Flet画布对象
        """
        for edge_id, edge in self._edges.items():
            # 获取源节点和目标节点
            source = self._nodes.get(edge["source"])
            target = self._nodes.get(edge["target"])
            if not source or not target:
                continue
                
            # 计算连线路径
            path = self._calculate_edge_path(
                source["x"], source["y"],
                target["x"], target["y"]
            )
            
            # 设置连线样式
            canvas.paint.color = get_color(f"edge.{edge['type']}")
            canvas.paint.stroke_width = edge.get("width", 2)
            
            # 绘制连线
            canvas.draw_path(path)
            
    def _calculate_edge_path(
        self,
        x1: float, y1: float,
        x2: float, y2: float
    ) -> ft.Path:
        """计算连线路径
        
        Args:
            x1: 起点X坐标
            y1: 起点Y坐标
            x2: 终点X坐标
            y2: 终点Y坐标
            
        Returns:
            连线路径
        """
        # 计算控制点
        dx = x2 - x1
        dy = y2 - y1
        
        cp1x = x1 + dx * 0.5
        cp1y = y1
        cp2x = x2 - dx * 0.5
        cp2y = y2
        
        # 创建贝塞尔曲线路径
        path = ft.Path()
        path.move_to(x1, y1)
        path.cubic_to(cp1x, cp1y, cp2x, cp2y, x2, y2)
        
        return path
        
    def create_node(self, node_type: str, position: Dict[str, float]):
        """创建新节点
        
        Args:
            node_type: 节点类型
            position: 节点位置
        """
        # 生成节点ID
        node_id = f"node_{len(self._nodes)}"
        
        # 对齐到网格
        x, y = self.grid.snap_point(position["x"], position["y"])
        
        # 创建节点数据
        node_data = {
            "id": node_id,
            "type": node_type,
            "x": x,
            "y": y,
            "title": f"Node {len(self._nodes) + 1}",
            "content": "Double click to edit"
        }
        
        # 添加到节点集合
        self._nodes[node_id] = node_data
        
        # 记录历史
        self.history_manager.push_state({
            "type": "create_node",
            "node_id": node_id,
            "data": node_data
        })
        
        self.update()
        
    def update_node_config(self, node_id: str, config: Dict):
        """更新节点配置
        
        Args:
            node_id: 节点ID
            config: 配置数据
        """
        if node_id not in self._nodes:
            return
            
        # 记录历史
        old_config = self._nodes[node_id].copy()
        self.history_manager.push_state({
            "type": "update_node",
            "node_id": node_id,
            "old_data": old_config,
            "new_data": config
        })
        
        # 更新配置
        self._nodes[node_id].update(config)
        self.update()
        
    def delete_selected_nodes(self):
        """删除选中的节点"""
        selected = self.selection_manager.get_selected()
        if not selected:
            return
            
        # 记录历史
        deleted_nodes = {}
        deleted_edges = {}
        
        for node_id in selected:
            # 保存节点数据
            deleted_nodes[node_id] = self._nodes[node_id]
            
            # 删除相关连线
            for edge_id, edge in list(self._edges.items()):
                if edge["source"] == node_id or edge["target"] == node_id:
                    deleted_edges[edge_id] = edge
                    del self._edges[edge_id]
                    
            # 删除节点
            del self._nodes[node_id]
            
        self.history_manager.push_state({
            "type": "delete_nodes",
            "nodes": deleted_nodes,
            "edges": deleted_edges
        })
        
        # 清除选择
        self.selection_manager.clear()
        self.update()
        
    def undo(self):
        """撤销操作"""
        state = self.history_manager.undo()
        if not state:
            return
            
        self._apply_history_state(state)
        self.update()
        
    def redo(self):
        """重做操作"""
        state = self.history_manager.redo()
        if not state:
            return
            
        self._apply_history_state(state)
        self.update()
        
    def _apply_history_state(self, state: Dict):
        """应用历史状态
        
        Args:
            state: 历史状态数据
        """
        action = state["type"]
        
        if action == "create_node":
            self._nodes[state["node_id"]] = state["data"]
            
        elif action == "update_node":
            if state["node_id"] in self._nodes:
                self._nodes[state["node_id"]] = state["new_data"]
                
        elif action == "delete_nodes":
            # 恢复删除的节点
            self._nodes.update(state["nodes"])
            self._edges.update(state["edges"])
            
    def _on_selection_change(self, selected: List[str]):
        """选择变化处理
        
        Args:
            selected: 选中的节点ID列表
        """
        if self.on_selection_change:
            self.on_selection_change(selected)
            
    def _on_drag_start(self, e: ft.DragStartEvent):
        """拖拽开始处理"""
        self.drag_manager.start_drag(e)
        
    def _on_drag_update(self, e: ft.DragUpdateEvent):
        """拖拽更新处理"""
        self.drag_manager.update_drag(e)
        
    def _on_drag_end(self, e: ft.DragEndEvent):
        """拖拽结束处理"""
        self.drag_manager.end_drag(e)
        
    def _on_pan_start(self, e: ft.DragStartEvent):
        """平移开始处理"""
        pass
        
    def _on_pan_update(self, e: ft.DragUpdateEvent):
        """平移更新处理"""
        self._viewport_x += e.delta_x
        self._viewport_y += e.delta_y
        self.update()
        
    def _on_pan_end(self, e: ft.DragEndEvent):
        """平移结束处理"""
        pass
        
    def _on_scale_start(self, e: ft.ScaleStartEvent):
        """缩放开始处理"""
        self.zoom_manager.start_zoom(e)
        
    def _on_scale_update(self, e: ft.ScaleUpdateEvent):
        """缩放更新处理"""
        self.zoom_manager.update_zoom(e)
        
    def _on_scale_end(self, e: ft.ScaleEndEvent):
        """缩放结束处理"""
        self.zoom_manager.end_zoom(e)
        
    def _on_zoom_change(self, scale: float):
        """缩放变化处理
        
        Args:
            scale: 缩放比例
        """
        self.update()
