"""
素材面板模块

负责:
- 管理素材资源
- 处理素材的导入和预览
- 支持素材拖放到画布
- 提供素材分类管理
- 在左侧边栏展示

素材支持缩略图预览,可切换列表和网格视图
"""

from typing import Dict, List, Optional, Callable
import flet as ft
from nndeploy.ui.config.language import get_text
from nndeploy.ui.entity.material_repository import Material, MaterialType

class MaterialCard(ft.Container):
    """素材卡片"""
    
    def __init__(
        self,
        material: Material,
        on_select: Optional[Callable[[str], None]] = None,
        on_drag_start: Optional[Callable[[str, float, float], None]] = None
    ):
        self.material = material
        self.on_select = on_select
        self.on_drag_start = on_drag_start
        
        # 构建卡片内容
        card_content = ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        # 预览图
                        ft.Container(
                            content=self._build_preview(),
                            height=120,
                            border_radius=ft.border_radius.vertical(5, 0)
                        ),
                        
                        # 信息
                        ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text(
                                        self.material.name,
                                        size=14,
                                        weight=ft.FontWeight.BOLD
                                    ),
                                    ft.Text(
                                        self.material.description,
                                        size=12,
                                        color=ft.colors.BLACK54,
                                        max_lines=2,
                                        overflow=ft.TextOverflow.ELLIPSIS
                                    )
                                ],
                                spacing=5
                            ),
                            padding=10
                        )
                    ],
                    spacing=0
                )
            )
        )
        
        # 使用手势检测器包装卡片
        gesture_detector = ft.GestureDetector(
            content=card_content,
            mouse_cursor=ft.MouseCursor.MOVE,
            on_tap=lambda _: (
                self.on_select and self.on_select(self.material.id)
            ),
            on_pan_start=self._on_drag_start
        )
        
        # 初始化父类
        super().__init__(content=gesture_detector)
        
    def _build_preview(self) -> ft.Control:
        """构建预览控件"""
        if self.material.type == "image":
            return ft.Image(
                src=self.material.path,
                fit=ft.ImageFit.COVER
            )
        elif self.material.type == "video":
            return ft.Stack(
                [
                    ft.Image(
                        src=self.material.thumbnail,
                        fit=ft.ImageFit.COVER
                    ),
                    ft.Container(
                        content=ft.Icon(
                            ft.icons.PLAY_CIRCLE,
                            size=40,
                            color=ft.colors.WHITE
                        ),
                        alignment=ft.alignment.center
                    )
                ]
            )
        else:
            return ft.Icon(
                self._get_type_icon(),
                size=40,
                color=ft.colors.BLACK54
            )
            
    def _get_type_icon(self) -> str:
        """获取类型图标"""
        icons = {
            "text": ft.icons.TEXT_FIELDS,
            "audio": ft.icons.AUDIOTRACK,
            "file": ft.icons.INSERT_DRIVE_FILE
        }
        return icons.get(self.material.type, ft.icons.HELP)
        
    def _on_drag_start(self, e: ft.DragStartEvent):
        """开始拖动"""
        if self.on_drag_start:
            self.on_drag_start(self.material.id, e.local_x, e.local_y)

class MaterialPanel(ft.Column):
    """素材面板"""
    
    def __init__(
        self,
        on_material_select: Optional[Callable[[str], None]] = None,
        on_material_drag_start: Optional[Callable[[str, float, float], None]] = None,
        on_import: Optional[Callable[[], None]] = None
    ):
        self.on_material_select = on_material_select
        self.on_material_drag_start = on_material_drag_start
        self.on_import = on_import
        self._materials: Dict[str, Material] = {}
        self._search_text = ""
        self._grid_view = True
        
        # 工具栏
        self.toolbar = ft.Row(
            [
                ft.TextField(
                    prefix_icon=ft.icons.SEARCH,
                    hint_text=get_text("material.search"),
                    expand=True,
                    on_change=self._on_search_change
                ),
                ft.IconButton(
                    icon=ft.icons.GRID_VIEW if self._grid_view
                    else ft.icons.LIST,
                    tooltip=get_text(
                        "material.gridView" if self._grid_view
                        else "material.listView"
                    ),
                    on_click=self._toggle_view
                ),
                ft.IconButton(
                    icon=ft.icons.ADD,
                    tooltip=get_text("material.import"),
                    on_click=lambda _: (
                        self.on_import and self.on_import()
                    )
                )
            ]
        )
        
        # 素材列表
        self.material_grid = ft.GridView(
            runs_count=2 if self._grid_view else 1,
            max_extent=200 if self._grid_view else 400,
            child_aspect_ratio=1.0 if self._grid_view else 3.0,
            spacing=10,
            run_spacing=10
        )
        
        # 初始化父类
        super().__init__(
            [self.toolbar, self.material_grid],
            spacing=20
        )
        
    def _on_search_change(self, e):
        """搜索文本变化"""
        self._search_text = e.control.value.lower()
        self._update_material_grid()
        self.update()
        
    def _toggle_view(self, _):
        """切换视图模式"""
        self._grid_view = not self._grid_view
        self.material_grid.runs_count = 2 if self._grid_view else 1
        self.material_grid.max_extent = 200 if self._grid_view else 400
        self.material_grid.child_aspect_ratio = 1.0 if self._grid_view else 3.0
        self._update_material_grid()
        self.update()
        
    def _filter_materials(self) -> List[Material]:
        """过滤素材列表"""
        return [
            material for material in self._materials.values()
            if not self._search_text or
            self._search_text in material.name.lower() or
            self._search_text in material.description.lower()
        ]
    
    def _update_material_grid(self):
        """更新素材网格"""
        self.material_grid.controls = [
            MaterialCard(
                material=material,
                on_select=self.on_material_select,
                on_drag_start=self.on_material_drag_start
            )
            for material in self._filter_materials()
        ]
        
    def update_materials(self, materials: Dict[str, Material]):
        """更新素材列表"""
        self._materials = materials
        self._update_material_grid()
        self.update()

def main(page: ft.Page):
    """预览演示"""
    page.title = "素材面板演示"
    
    # 创建测试素材
    materials = {
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
            description="这是一个测试视频素材，描述比较长以测试多行显示效果",
            material_type="video",
            path="video.mp4",
            thumbnail="https://picsum.photos/200/300"
        ),
        "3": Material(
            id="3",
            name="测试文本",
            description="这是一个测试文本素材",
            material_type="text",
            path="text.txt"
        ),
        "4": Material(
            id="4",
            name="测试音频",
            description="这是一个测试音频素材",
            material_type="audio",
            path="audio.mp3"
        )
    }
    
    def on_material_select(material_id: str):
        page.snack_bar = ft.SnackBar(
            content=ft.Text(f"选择了素材: {materials[material_id].name}")
        )
        page.snack_bar.open = True
        page.update()
    
    def on_material_drag_start(material_id: str, x: float, y: float):
        page.snack_bar = ft.SnackBar(
            content=ft.Text(f"开始拖动素材: {materials[material_id].name} 位置: ({x}, {y})")
        )
        page.snack_bar.open = True
        page.update()
    
    def on_import():
        page.snack_bar = ft.SnackBar(
            content=ft.Text("导入素材")
        )
        page.snack_bar.open = True
        page.update()
    
    # 创建素材面板
    panel = MaterialPanel(
        on_material_select=on_material_select,
        on_material_drag_start=on_material_drag_start,
        on_import=on_import
    )
    
    # 先将面板添加到页面，然后再更新素材
    page.add(panel)
    panel.update_materials(materials)

if __name__ == "__main__":
    ft.app(target=main, view=ft.WEB_BROWSER, port=9090)
