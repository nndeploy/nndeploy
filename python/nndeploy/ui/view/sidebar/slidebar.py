"""
侧边栏模块

负责:
- 组织和管理所有侧边面板
- 提供面板切换功能
- 支持面板折叠/展开
"""

from typing import Optional, Callable
import flet as ft
from ...config.language import get_text
from .node_panel import NodePanel
from .model_panel import ModelPanel 
from .material_panel import MaterialPanel

class SideBar(ft.UserControl):
    """侧边栏"""
    
    def __init__(
        self,
        # 节点相关回调
        on_node_drag_start: Optional[Callable[[str, float, float], None]] = None,
        # 模型相关回调
        on_model_select: Optional[Callable[[str], None]] = None,
        on_model_config: Optional[Callable[[str], None]] = None,
        on_model_import: Optional[Callable[[], None]] = None,
        # 素材相关回调
        on_material_select: Optional[Callable[[str], None]] = None,
        on_material_drag_start: Optional[Callable[[str, float, float], None]] = None,
        on_material_import: Optional[Callable[[], None]] = None,
    ):
        super().__init__()
        
        # 保存回调函数
        self.on_node_drag_start = on_node_drag_start
        self.on_model_select = on_model_select
        self.on_model_config = on_model_config
        self.on_model_import = on_model_import
        self.on_material_select = on_material_select
        self.on_material_drag_start = on_material_drag_start
        self.on_material_import = on_material_import

        # 创建面板
        self.node_panel = NodePanel(on_node_drag_start=on_node_drag_start)
        self.model_panel = ModelPanel(
            on_model_select=on_model_select,
            on_model_config=on_model_config,
            on_import=on_model_import
        )
        self.material_panel = MaterialPanel(
            on_material_select=on_material_select,
            on_material_drag_start=on_material_drag_start,
            on_import=on_material_import
        )

        # 当前选中的面板索引
        self._selected_index = 0
        
    def build(self):
        return ft.Container(
            content=ft.Row(
                [
                    # 左侧标签栏
                    ft.NavigationRail(
                        selected_index=self._selected_index,
                        label_type=ft.NavigationRailLabelType.ALL,
                        min_width=100,
                        min_extended_width=150,
                        destinations=[
                            ft.NavigationRailDestination(
                                icon=ft.icons.ACCOUNT_TREE_OUTLINED,
                                selected_icon=ft.icons.ACCOUNT_TREE,
                                label=get_text("sidebar.nodes"),
                            ),
                            ft.NavigationRailDestination(
                                icon=ft.icons.MODEL_TRAINING_OUTLINED,
                                selected_icon=ft.icons.MODEL_TRAINING,
                                label=get_text("sidebar.models"),
                            ),
                            ft.NavigationRailDestination(
                                icon=ft.icons.COLLECTIONS_OUTLINED,
                                selected_icon=ft.icons.COLLECTIONS,
                                label=get_text("sidebar.materials"),
                            ),
                        ],
                        on_change=self._on_tab_change,
                    ),
                    
                    # 右侧面板区域
                    ft.VerticalDivider(width=1),
                    ft.Container(
                        content=ft.AnimatedSwitcher(
                            content=self._get_current_panel(),
                            transition=ft.AnimatedSwitcherTransition.FADE,
                            duration=300,
                            switch_in_curve=ft.AnimationCurve.EASE_OUT,
                            switch_out_curve=ft.AnimationCurve.EASE_IN,
                        ),
                        expand=True,
                        padding=10
                    ),
                ],
                expand=True,
            ),
            bgcolor=ft.colors.BACKGROUND,
        )

    def _get_current_panel(self):
        """获取当前选中的面板"""
        panels = [self.node_panel, self.model_panel, self.material_panel]
        return panels[self._selected_index]

    def _on_tab_change(self, e):
        """标签切换事件"""
        self._selected_index = e.control.selected_index
        self.update()

def main(page: ft.Page):
    """侧边栏演示"""
    page.title = "侧边栏演示"
    page.padding = 0
    
    def handle_node_drag(node_id: str, x: float, y: float):
        page.show_snack_bar(ft.SnackBar(
            content=ft.Text(f"开始拖动节点: {node_id} 位置: ({x}, {y})")
        ))
    
    def handle_model_select(model_id: str):
        page.show_snack_bar(ft.SnackBar(
            content=ft.Text(f"选择模型: {model_id}")
        ))
    
    def handle_model_config(model_id: str):
        page.show_snack_bar(ft.SnackBar(
            content=ft.Text(f"配置模型: {model_id}")
        ))
    
    def handle_model_import():
        page.show_snack_bar(ft.SnackBar(
            content=ft.Text("导入模型")
        ))
    
    def handle_material_select(material_id: str):
        page.show_snack_bar(ft.SnackBar(
            content=ft.Text(f"选择素材: {material_id}")
        ))
    
    def handle_material_drag(material_id: str, x: float, y: float):
        page.show_snack_bar(ft.SnackBar(
            content=ft.Text(f"开始拖动素材: {material_id} 位置: ({x}, {y})")
        ))
    
    def handle_material_import():
        page.show_snack_bar(ft.SnackBar(
            content=ft.Text("导入素材")
        ))

    # 创建主布局
    layout = ft.Row(
        [
            # 左侧边栏
            ft.Container(
                content=SideBar(
                    on_node_drag_start=handle_node_drag,
                    on_model_select=handle_model_select,
                    on_model_config=handle_model_config,
                    on_model_import=handle_model_import,
                    on_material_select=handle_material_select,
                    on_material_drag_start=handle_material_drag,
                    on_material_import=handle_material_import
                ),
                width=400,
                border=ft.border.only(right=ft.BorderSide(1, ft.colors.OUTLINE))
            ),
            
            # 右侧主内容区域
            ft.Container(
                content=ft.Text("主内容区域", size=24),
                expand=True,
                alignment=ft.alignment.center
            )
        ],
        expand=True
    )
    
    # 添加到页面
    page.add(layout)
    
    # 更新一些测试数据
    from ...entity.model_repository import Model, ModelType, ModelStatus
    sidebar = layout.controls[0].content
    sidebar.model_panel.update_models({
        "1": Model(
            id="1",
            name="测试模型1",
            description="这是一个测试模型",
            model_type=ModelType.LLM,
            status=ModelStatus.AVAILABLE
        ),
        "2": Model(
            id="2", 
            name="测试模型2",
            description="这是另一个测试模型",
            model_type=ModelType.IMAGE,
            status=ModelStatus.LOADING
        )
    })
    
    from ...entity.material_repository import Material
    sidebar.material_panel.update_materials({
        "1": Material(
            id="1",
            name="测试图片",
            description="这是一个测试图片素材",
            material_type="image",
            path="https://picsum.photos/200/300",
            thumbnail="https://picsum.photos/200/300"
        ),
        "2": Material(
            id="2",
            name="测试视频",
            description="这是一个测试视频素材",
            material_type="video", 
            path="video.mp4",
            thumbnail="https://picsum.photos/200/300"
        )
    })

if __name__ == "__main__":
    ft.app(target=main, view=ft.WEB_BROWSER)
