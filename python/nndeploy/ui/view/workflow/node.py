from typing import List, Optional, Dict, Any
import flet as ft
from slot import Slot  # 添加 Slot 类的导入

class WorkflowNode:
    """工作流节点基类"""
    def __init__(
        self,
        title: str,
        width: float = 200,
        height: float = 300,
        position: tuple = (0, 0),
        page: ft.Page = None,
        canvas_manager=None,  # 添加 canvas_manager 参数
    ):
        self.title = title
        self.width = width
        self.height = height
        self.position = position
        self.page = page
        self.canvas_manager = canvas_manager  # 保存 canvas_manager 引用
        self.inputs = []
        self.outputs = []
        self.properties = {}
        self.input_slots = {}
        self.output_slots = {}
        self.on_position_changed = None  # 添加这行
        self._last_update_time = 0  # 添加时间戳记录
        self._update_interval = 0.01  # 设置更新间隔为100ms
        
        # 允许子类初始化自己的属性
        self.init_properties()
        
        # 创建插槽
        self._create_slots()
        
        # 创建节点容器
        self.container = self._create_container()
        
        # 添加拖动功能
        self._setup_drag_feature()
    
    def _create_slots(self):
        """创建输入输出插槽"""
        # 创建输入插槽
        for name in self.inputs:
            slot = Slot(name, "input", on_position_changed=self.on_position_changed,canvas_manager=self.canvas_manager)
            self.input_slots[name] = slot
        
        # 创建输出插槽
        for name in self.outputs:
            slot = Slot(name, "output", on_position_changed=self.on_position_changed,canvas_manager=self.canvas_manager)
            self.output_slots[name] = slot
    
    def on_slot_position_changed(self):
        """当插槽位置改变时的回调"""
        if hasattr(self, 'on_position_changed'):
            self.on_position_changed()
    
    def _setup_drag_feature(self):
        """设置拖动功能"""
        def on_pan_start(e):
            # 开始拖动时，将节点移到最上层
            if self.container.parent:
                controls = self.container.parent.controls
                controls.remove(self.container)
                controls.append(self.container)
                self.container.parent.update()
        import time 

        def on_pan_update(e):
            self.container.left = self.container.left + e.delta_x
            self.container.top = self.container.top + e.delta_y
            self.position = (self.container.left, self.container.top)
            
            # 通知更新，传入当前移动的节点
            if hasattr(self, 'on_position_changed'):
                self.canvas_manager.update_connections(self)
            self.page.update()
            
            # 检查是否需要更新连接线
            current_time = time.time()
            if current_time - self._last_update_time >= self._update_interval:
                if hasattr(self, 'on_position_changed'):
                    self.on_position_changed()
                self._last_update_time = current_time
            
            # 始终更新节点位置
            self.page.update()
        
        def on_click(e):
            # 点击时，将节点移到最上层
            if self.container.parent:
                controls = self.container.parent.controls
                controls.remove(self.container)
                controls.append(self.container)
                self.container.parent.update()
        
        # 将手势检测器添加到内容中
        self.container.content = ft.GestureDetector(
            content=self.container.content,
            drag_interval=10,
            on_pan_start=on_pan_start,
            on_pan_update=on_pan_update,
            on_tap=on_click,
        )
    
    def init_properties(self):
        """子类重写此方法来初始化自己的属性"""
        pass
    
    def _create_container(self) -> ft.Container:
        """创建节点的基础容器"""
        return ft.Container(
            content=ft.Column(
                controls=[
                    # 标题栏
                    self._create_title_bar(),
                    # 端口区
                    self._create_ports_area(),
                    # 内容区
                    self._create_content_area(),
                ],
                spacing=5,
                tight=True,
            ),
            width=self.width,
            height=self.height,
            border_radius=5,
            padding=0,
            bgcolor=ft.colors.WHITE,
            left=self.position[0],
            top=self.position[1],
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=4,
                color=ft.colors.with_opacity(0.25, ft.colors.BLACK),
                offset=ft.Offset(0, 2),
            ),
        )
    
    def _create_ports_area(self) -> ft.Container:
        """创建输入输出端口区域"""
        return ft.Container(
            content=ft.Row(
                controls=[
                    ft.Container(
                        content=ft.Column(
                            controls=[slot.container for slot in self.input_slots.values()],
                            spacing=5,
                            alignment=ft.MainAxisAlignment.START,
                        ),
                        padding=ft.padding.only(left=5),
                    ),
                    ft.Container(expand=True),
                    ft.Container(
                        content=ft.Column(
                            controls=[slot.container for slot in self.output_slots.values()],
                            spacing=5,
                            alignment=ft.MainAxisAlignment.START,
                        ),
                        padding=ft.padding.only(right=5),
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            height=50,  # 给定固定高度
            padding=5,
        )
    
    def _create_title_bar(self) -> ft.Container:
        """创建标题栏"""
        return ft.Container(
            content=ft.Text(self.title, size=14, weight=ft.FontWeight.BOLD),
            bgcolor=ft.colors.BLUE_GREY_100,
            padding=5,
        )
    
    def _create_content_area(self) -> ft.Container:
        """创建内容区域，子类重写此方法来自定义内容"""
        return ft.Container(
            content=ft.Column(
                controls=[],
                scroll=ft.ScrollMode.AUTO,
            ),
            expand=True,
            padding=5,
        )
    
    def _create_ports_area(self) -> ft.Row:
        """创建输入输出端口区域"""
        input_column = ft.Column(
            controls=[slot.container for slot in self.input_slots.values()],
            spacing=5,
            alignment=ft.MainAxisAlignment.START,
        )
        
        output_column = ft.Column(
            controls=[slot.container for slot in self.output_slots.values()],
            spacing=5,
            alignment=ft.MainAxisAlignment.START,
        )
        
        return ft.Container(
            content=ft.Row(
                controls=[
                    input_column,
                    ft.Container(expand=True),
                    output_column,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            expand=True,
            padding=0,
        )
    
    def _create_input_ports(self) -> ft.Row:
        """创建输入端口"""
        return ft.Row(
            controls=[
                ft.Container(
                    width=10,
                    height=10,
                    bgcolor=ft.colors.BLUE,
                    border_radius=5,
                ) for _ in self.inputs
            ],
            spacing=5,
        )
    
    def _create_output_ports(self) -> ft.Row:
        """创建输出端口"""
        return ft.Row(
            controls=[
                ft.Container(
                    width=10,
                    height=10,
                    bgcolor=ft.colors.GREEN,
                    border_radius=5,
                ) for _ in self.outputs
            ],
            spacing=5,
        )
    
    def update(self):
        """更新节点状态"""
        pass
    
    def to_dict(self) -> Dict:
        """将节点数据序列化为字典"""
        return {
            "title": self.title,
            "position": self.position,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "properties": self.properties,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "WorkflowNode":
        """从字典创建节点"""
        node = cls(
            title=data["title"],
            position=data["position"],
        )
        node.inputs = data["inputs"]
        node.outputs = data["outputs"]
        node.properties = data["properties"]
        return node