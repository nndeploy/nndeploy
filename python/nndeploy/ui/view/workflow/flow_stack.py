from typing import Callable
import flet
from flet.usercontrol import UserControl
from flet import (
    Container,
    Stack,
    DragTarget,
    Draggable,
    alignment,
    border,
    colors,
    Text,
    Column,
    Row,
    IconButton,
    icons,
)

class NodePort(UserControl):
    """节点端口类，用于连接节点"""
    def __init__(self, port_type="input", text=""):
        super().__init__()
        self.port_type = port_type
        self.text = text
        self.connections = []
        
    def build(self):
        return Container(
            content=Row([
                Container(
                    width=12,
                    height=12,
                    border_radius=6,
                    bgcolor=colors.BLUE if self.port_type == "input" else colors.GREEN,
                ),
                Text(self.text, size=12),
            ]),
            on_hover=self._on_hover,
        )
    
    def _on_hover(self, e):
        self.connections

class FlowNode(UserControl):
    """工作流节点类"""
    def __init__(self, title, inputs=None, outputs=None):
        super().__init__()
        self.title = title
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.position = (0, 0)
        
    def build(self):
        return DragTarget(
            content=Draggable(
                content=Container(
                    content=Column([
                        # 节点标题
                        Container(
                            content=Row([
                                Text(self.title, size=14, weight="bold"),
                                IconButton(
                                    icon=icons.CLOSE,
                                    icon_size=14,
                                    on_click=self._delete_node
                                ),
                            ], alignment="spaceBetween"),
                            bgcolor=colors.BLUE_GREY_100,
                            padding=5,
                        ),
                        # 输入端口
                        Column([NodePort("input", text=inp) for inp in self.inputs]),
                        # 输出端口
                        Column([NodePort("output", text=out) for out in self.outputs]),
                    ]),
                    width=200,
                    bgcolor=colors.WHITE,
                    border=border.all(1, colors.BLUE_GREY_300),
                    border_radius=5,
                    shadow=3,
                ),
                group="nodes",
            ),
            on_accept=self._on_drop,
        )
    
    def _delete_node(self, e):
        self.remove()
    
    def _on_drop(self, e):
        self.position = (e.x, e.y)
        self.update()

class FlowStack(UserControl):
    """工作流编辑器主类"""
    def __init__(self):
        super().__init__()
        self.nodes = []
        self.connections = []
        
    def build(self):
        return Container(
            content=Stack(
                controls=self.nodes,
            ),
            width=1000,
            height=600,
            bgcolor=colors.BLUE_GREY_50,
            border_radius=10,
            padding=20,
        )
    
    def add_node(self, node_type, position=(0, 0)):
        """添加新节点"""
        node_configs = {
            "input": FlowNode("输入", outputs=["图片", "参数"]),
            "process": FlowNode("处理", inputs=["输入"], outputs=["输出"]),
            "output": FlowNode("输出", inputs=["结果"]),
        }
        
        if node_type in node_configs:
            node = node_configs[node_type]
            node.position = position
            self.nodes.append(node)
            self.update()
    
    def connect_nodes(self, source_port, target_port):
        """连接节点"""
        if source_port.port_type == "output" and target_port.port_type == "input":
            connection = (source_port, target_port)
            self.connections.append(connection)
            source_port.connections.append(connection)
            target_port.connections.append(connection)
            self.update()

# 使用示例
def main(page: flet.Page):
    flow_editor = FlowStack()
    
    # 添加工具栏
    toolbar = Row(
        controls=[
            IconButton(
                icon=icons.ADD,
                on_click=lambda _: flow_editor.add_node("input", (100, 100))
            ),
            IconButton(
                icon=icons.SETTINGS,
                on_click=lambda _: flow_editor.add_node("process", (300, 100))
            ),
            IconButton(
                icon=icons.OUTPUT,
                on_click=lambda _: flow_editor.add_node("output", (500, 100))
            ),
        ]
    )
    
    page.add(toolbar, flow_editor)

if __name__ == "__main__":
    flet.app(target=main)