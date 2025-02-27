import flet as ft
from flet.canvas import Canvas, Path
from typing import Optional, Tuple, Callable
from flet import Draggable, DragTarget
from coordinate_mapper import CoordinateMapper

class Slot:
    """节点的输入/输出插槽"""
    def __init__(
        self,
        name: str,
        slot_type: str = "input",  # "input" 或 "output"
        position: tuple = (0, 0),
        on_position_changed: Optional[Callable] = None,
        canvas_manager=None,  # 添加 canvas_manager 参数
    ):
        self.name = name
        self.type = slot_type
        self.position = position
        self.on_position_changed = on_position_changed
        self.canvas_manager = canvas_manager  # 保存 canvas_manager 引用
        
        # 创建插槽的可视化控件
        self.container = self._create_container()
    
    def _create_container(self) -> ft.Container:
        self.is_hovered = False
        
        def on_hover(e):
            self.is_hovered = e.data == "true"
            # 更新插槽的外观
            if self.type == "input":
                self.slot_dot.bgcolor = ft.colors.BLUE_ACCENT if self.is_hovered else ft.colors.BLUE
            else:
                self.slot_dot.bgcolor = ft.colors.GREEN_ACCENT if self.is_hovered else ft.colors.GREEN
            self.slot_dot.update()
        
        # 创建基础的插槽点容器
        self.slot_dot = ft.Container(
            width=10,
            height=10,
            bgcolor=ft.colors.BLUE if self.type == "input" else ft.colors.GREEN,
            border_radius=5,
            on_hover=on_hover,
            animate=ft.animation.Animation(300, "easeOut"),
        )
        
        # 创建名称标签
        name_label = ft.Text(
            self.name,
            size=12,
            color=ft.colors.GREY_700,
        )
        
        # 根据类型创建不同的布局
        if self.type == "input":
            # 输入插槽使用 DragTarget
            self.drag_target = DragTarget(
                content=self.slot_dot,
                on_accept=self._handle_drag_accept,
                on_will_accept=self._handle_will_accept,
            )
            return ft.Container(
                content=ft.Row(
                    [self.drag_target, name_label],
                    spacing=5,
                    alignment=ft.MainAxisAlignment.START,
                ),
            )
        else:  # output
            # 先使用 GestureDetector 包装 slot_dot
            self.gesture_detector = ft.GestureDetector(
                content=self.slot_dot,
                on_pan_start=self._handle_pan_start,
                on_pan_update=self._handle_pan_update,
                on_pan_end=self._handle_pan_end,
            )
            
            # 然后用 Draggable 包装 GestureDetector
            self.draggable = Draggable(
                content=self.gesture_detector,
                data=self,  # 传递 slot 实例
                on_drag_start=self._handle_drag_start,
                on_drag_complete=self._handle_drag_end,
            )
            
            return ft.Container(
                content=ft.Row(
                    [name_label, self.draggable],
                    spacing=5,
                    alignment=ft.MainAxisAlignment.END,
                ),
            )
    
    def _handle_drag_start(self, e):
        """处理拖动开始"""
        if self.canvas_manager:
            self.canvas_manager.start_temp_connection(self)
    
    def _handle_drag_update(self, e):
        """处理拖动更新"""
        if self.canvas_manager:
            self.canvas_manager.update_temp_connection(e.global_x, e.global_y)
    
    def _handle_drag_end(self, e):
        """处理拖动结束"""
        if self.canvas_manager:
            self.canvas_manager.end_temp_connection()
    
    def _handle_will_accept(self, e):
        """检查是否接受拖放"""
        # 只接受来自输出插槽的连接
        return (
            isinstance(e.data, Slot) and 
            e.data.type == "output" and 
            self.type == "input"
        )
    
    def _handle_drag_accept(self, e):
        """处理拖放接受"""
        if self.canvas_manager:
            from_slot = e.data
            self.canvas_manager.add_connection(from_slot, self)
    
    def _handle_pan_start(self, e):
        """处理拖动开始"""
        if self.canvas_manager and self.type == "output":
            self.canvas_manager.start_temp_connection(self)
    
    def _handle_pan_update(self, e):
        """处理拖动更新"""
        if self.canvas_manager and self.type == "output":
            # 获取当前鼠标位置
            self.canvas_manager.update_temp_connection(e.global_x, e.global_y)
    
    def _handle_pan_end(self, e):
        """处理拖动结束"""
        if self.canvas_manager:
            if self.type == "input" and self.canvas_manager.dragging_slot:
                # 如果是输入插槽且有正在拖动的输出插槽，创建连接
                from_slot = self.canvas_manager.dragging_slot
                if from_slot.type == "output":
                    self.canvas_manager.add_connection(from_slot, self)
            self.canvas_manager.end_temp_connection()
    
    def get_center_position(self) -> tuple:
        """获取插槽中心点的绝对坐标（相对于画布）"""
        # 从 slot_dot 开始计算
        total_x = self.slot_dot.width / 2  # slot_dot 的中心点
        total_y = self.slot_dot.height / 2
        current = self.slot_dot
        
        # 记录是否找到了 Column 容器（用于垂直排列的容器）
        found_column = False
        
        # 向上遍历所有父容器，直到找到 Canvas
        while current and not isinstance(current, Canvas):
            # 累加容器自身的位置
            if hasattr(current, 'left') and current.left is not None:
                total_x += current.left
            if hasattr(current, 'top') and current.top is not None:
                total_y += current.top
            
            # 处理垂直布局
            if isinstance(current, ft.Column) and not found_column:
                found_column = True
                # 获取当前 slot_dot 在 Column 中的索引
                if hasattr(current, 'controls'):
                    for i, control in enumerate(current.controls):
                        if self.slot_dot in control.content.controls:
                            # 根据索引计算垂直偏移
                            total_y += i * (self.slot_dot.height + 5)  # 5 是垂直间距
                            break
            
            # 处理水平布局
            if isinstance(current, ft.Row):
                if self.type == "output":
                    # 对于输出插槽，需要加上父容器的宽度
                    parent = current.parent
                    if parent and hasattr(parent, 'width') and parent.width is not None:
                        total_x += parent.width - self.slot_dot.width - 10  # 10 是右边距
                elif self.type == "input":
                    # 对于输入插槽，只需要加上左边距
                    total_x += 10  # 10 是左边距
            
            # 累加容器的外边距
            if hasattr(current, 'margin') and current.margin is not None:
                margin = current.margin
                if hasattr(margin, 'left') and margin.left is not None:
                    total_x += margin.left
                if hasattr(margin, 'top') and margin.top is not None:
                    total_y += margin.top
            
            # 累加父容器的内边距
            parent = current.parent
            if parent and hasattr(parent, 'padding') and parent.padding is not None:
                padding = parent.padding
                if hasattr(padding, 'left') and padding.left is not None:
                    total_x += padding.left
                if hasattr(padding, 'top') and padding.top is not None:
                    total_y += padding.top
            
            current = parent
        
        local_center = (self.slot_dot.width / 2, self.slot_dot.height / 2)
        return CoordinateMapper.map_to_canvas(self.slot_dot, local_center)
    
    def update_position(self, x: float, y: float):
        """更新插槽位置"""
        self.position = (x, y)
        if self.on_position_changed:
            self.on_position_changed()