import flet
from flet.canvas import Canvas, Path
from flet.core.painting import Paint, PaintingStyle
from flet.core.types import StrokeCap

from flet import (
    Container,
    Stack,
)

from typing import List, Optional, Tuple
from connection import Connection
from slot import Slot
from flet import GestureDetector
from node import WorkflowNode

class CanvasManager:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.connections: List[Connection] = []
        self.nodes: List[WorkflowNode] = []  # 添加节点列表
        self.temp_connection = None
        self.dragging_slot = None
        
        # 视图变换参数
        self.scale = 1.0  # 缩放比例
        self.offset_x = 0  # 视图水平偏移
        self.offset_y = 0  # 视图垂直偏移
        self.is_panning = False  # 是否正在平移
        self.last_pan_pos = None  # 上次平移位置
        
        # 创建画布
        self.canvas = Canvas(
            width=width,
            height=height,
            expand=True,
        )
        
        # 创建堆叠容器来放置节点和画布
        self.stack = Stack(
            controls=[self.canvas],
            expand=True,
        )
        
        # 创建手势检测器包装画布
        self.gesture_detector = GestureDetector(
            content=self.stack,  # 改为包装 stack
            drag_interval=50,
            on_pan_start=self._handle_pan_start,
            on_pan_update=self._handle_pan_update,
            on_pan_end=self._handle_pan_end,
            on_scroll=self._handle_scroll,
            expand=True,
        )
    
    def add_node(self, node: WorkflowNode):
        """添加节点"""
        self.nodes.append(node)
        self.stack.controls.append(node.container)
        
    def remove_node(self, node: WorkflowNode):
        """移除节点"""
        if node in self.nodes:
            self.nodes.remove(node)
            self.stack.controls.remove(node.container)
            # 移除与该节点相关的所有连接
            connections_to_remove = [
                conn for conn in self.connections
                if conn.from_slot in node.output_slots.values() or 
                   conn.to_slot in node.input_slots.values()
            ]
            for conn in connections_to_remove:
                self.remove_connection(conn)
        
        # 视图变换参数
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.is_panning = False
        self.last_pan_pos = None
        
        # 创建画布
        self.canvas = Canvas(
            width=width,
            height=height,
            expand=True,
        )
        
        # 创建手势检测器包装画布
        self.gesture_detector = GestureDetector(
            content=self.canvas,
            drag_interval=10,
            on_pan_start=self._handle_pan_start,
            on_pan_update=self._handle_pan_update,
            on_pan_end=self._handle_pan_end,
            on_scroll=self._handle_scroll,
        )
    
    def _handle_pan_start(self, e):
        """开始平移"""
        if not self.dragging_slot:
            self.is_panning = True
            # 记录起始点位置
            self.pan_start_pos = (e.local_x, e.local_y)
            self.last_pan_pos = (e.local_x, e.local_y)
            # 记录起始偏移
            self.start_offset_x = self.offset_x
            self.start_offset_y = self.offset_y
    
    def _handle_pan_update(self, e):
        """更新平移"""
        if self.is_panning and self.pan_start_pos:
            # 添加速度因子来减慢平移速度（值越小，移动越慢）
            speed_factor = 0.01
            
            # 计算相对于起始点的总位移，并应用速度因子
            total_dx = (e.local_x - self.pan_start_pos[0]) / self.scale * speed_factor
            total_dy = (e.local_y - self.pan_start_pos[1]) / self.scale * speed_factor
            
            # 直接更新到起始偏移位置加上总位移
            self.offset_x = self.start_offset_x + total_dx
            self.offset_y = self.start_offset_y + total_dy
            
            # 更新上次位置用于下次计算
            self.last_pan_pos = (e.local_x, e.local_y)
            
            # 重绘所有内容
            self._update_view()
    
    def _handle_pan_end(self, e):
        """结束平移"""
        self.is_panning = False
        self.last_pan_pos = None
    
    def _handle_scroll(self, e):
        """处理滚轮缩放"""
        # 获取鼠标位置作为缩放中心
        center_x = e.local_x
        center_y = e.local_y
        
        # 计算新的缩放比例
        old_scale = self.scale
        if e.scroll_delta_y < 0:
            self.scale = min(4.0, self.scale * 1.1)  # 放大限制
        else:
            self.scale = max(0.1, self.scale / 1.1)  # 缩小限制
        
        # 调整偏移以保持鼠标位置不变
        scale_factor = self.scale / old_scale
        self.offset_x = center_x - (center_x - self.offset_x) * scale_factor
        self.offset_y = center_y - (center_y - self.offset_y) * scale_factor
        
        # 更新视图
        self._update_view()
    
    def _update_view(self):
        """更新视图变换"""
        # 清除画布
        self.canvas.shapes.clear()
        
        # 更新所有节点的位置
        for node in self.nodes:
            # 从节点的容器中获取位置
            current_pos = (node.container.left, node.container.top)
            transformed_pos = self._transform_point(current_pos)
            
            # 更新节点容器的位置
            node.container.left = transformed_pos[0]
            node.container.top = transformed_pos[1]
            
            # 应用缩放到节点大小
            if hasattr(node.container, 'width'):
                node.container.width = node.container.width * self.scale
            if hasattr(node.container, 'height'):
                node.container.height = node.container.height * self.scale
            node.container.update()
        
        # 更新所有连接
        self.update_connections()
        
        # 更新临时连接
        if self.temp_connection:
            # 获取临时连接的起点和终点
            start_pos = self.dragging_slot.get_center_position()
            transformed_start = self._transform_point(start_pos)
            # 终点使用鼠标位置，已经是画布坐标系
            end_pos = (self.temp_connection.x2, self.temp_connection.y2)
            
            # 创建新的临时连接路径
            self.temp_connection = Path(
                [
                    Path.MoveTo(transformed_start[0], transformed_start[1]),
                    Path.LineTo(end_pos[0], end_pos[1]),
                ],
                paint=Paint(
                    stroke_width=2,
                    color=flet.colors.with_opacity(0.5, flet.colors.GREEN),
                    style=PaintingStyle.STROKE,
                    stroke_cap=StrokeCap.ROUND,
                ),
            )
            self.canvas.shapes.append(self.temp_connection)
        
        # 更新画布
        self.canvas.update()
    
    def _transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """应用视图变换到点"""
        x, y = point
        return (
            (x + self.offset_x) * self.scale,
            (y + self.offset_y) * self.scale
        )
    
    def add_connection(self, from_slot: Slot, to_slot: Slot) -> Connection:
        """添加新连接"""
        connection = Connection(from_slot, to_slot, self.canvas)
        self.connections.append(connection)
        return connection
    
    def remove_connection(self, connection: Connection):
        """移除连接"""
        if connection in self.connections:
            connection.remove()
            self.connections.remove(connection)
    
    def update_connections(self, moving_node: WorkflowNode = None):
        """更新连接线"""
        self.canvas.shapes.clear()
        
        if moving_node:
            # 只更新与移动节点相关的连接
            for connection in self.connections:
                if (any(slot == connection.from_slot for slot in moving_node.output_slots.values()) or
                    any(slot == connection.to_slot for slot in moving_node.input_slots.values())):
                    connection.update()
        else:
            # 更新所有连接
            for connection in self.connections:
                connection.update()
                
        self.canvas.update()
    
    def get_node_position(self, control) -> tuple:
        """获取控件所属节点的位置
        Args:
            control: 要查找的控件
        Returns:
            tuple: 节点的位置坐标 (x, y)
        """
        # 遍历所有节点
        for node in self.nodes:
            # 检查控件是否属于该节点
            if control in [node.container] or \
               any(slot.drag_target == control or slot.slot_dot == control 
                   for slot in node.input_slots.values()) or \
               any(slot.draggable == control or slot.slot_dot == control 
                   for slot in node.output_slots.values()):
                return (node.container.left or 0, node.container.top or 0)
        
        print(f"Warning: Could not find node for control {control}")
        return (0, 0)

    def start_temp_connection(self, slot: Slot):
        """开始临时连接"""
        self.dragging_slot = slot
    
    def update_temp_connection(self, end_x: float, end_y: float):
        """更新临时连接，使用贝塞尔曲线"""
        if not self.dragging_slot:
            return
            
        if self.temp_connection:
            self.canvas.shapes.remove(self.temp_connection)
            
        start_pos = self.dragging_slot.get_center_position()
        
        # 计算控制点
        # 对于输出插槽，控制点向右偏移
        # 对于输入插槽，控制点向左偏移
        control_distance = 100  # 控制点距离
        
        if self.dragging_slot.type == "output":
            control1 = (start_pos[0] + control_distance, start_pos[1])
            control2 = (end_x - control_distance, end_y)
        else:
            control1 = (start_pos[0] - control_distance, start_pos[1])
            control2 = (end_x + control_distance, end_y)
        
        # 创建三次贝塞尔曲线路径
        self.temp_connection = Path(
            [
                Path.MoveTo(start_pos[0], start_pos[1]),
                Path.CubicTo(
                    control1[0], control1[1],  # 第一个控制点
                    control2[0], control2[1],  # 第二个控制点
                    end_x, end_y,              # 终点
                ),
            ],
            paint=Paint(
                stroke_width=2,
                color=flet.colors.with_opacity(0.5, flet.colors.GREEN),
                style=PaintingStyle.STROKE,
                stroke_cap=StrokeCap.ROUND,
            ),
        )
        self.canvas.shapes.append(self.temp_connection)
        self.canvas.update()
    
    def end_temp_connection(self):
        """结束临时连接"""
        if self.temp_connection:
            self.canvas.shapes.remove(self.temp_connection)
            self.temp_connection = None
        self.dragging_slot = None
        self.canvas.update()