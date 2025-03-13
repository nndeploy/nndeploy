"""
素材面板模块

负责:
- 管理素材资源
- 处理素材的导入和预览
- 支持素材拖放到画布
- 提供素材分类管理

素材支持缩略图预览,可切换列表和网格视图
"""

from typing import Dict, List, Optional, Callable
import flet as ft
from ...config.language import get_text
from ...entity.material_repository import Material

class MaterialCard(ft.UserControl):
    """素材卡片"""
    
    def __init__(
        self,
        material: Material,
        on_select: Optional[Callable[[str], None]] = None,
        on_drag_start: Optional[Callable[[str, float, float], None]] = None
    ):
        super().__init__()
        self.material = material
        self.on_select = on_select
        self.on_drag_start = on_drag_start
        
    def build(self):
        return ft.GestureDetector(
            content=ft.Card(
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
            ),
            mouse_cursor=ft.MouseCursor.MOVE,
            on_tap=lambda _: (
                self.on_select and self.on_select(self.material.id)
            ),
            on_pan_start=self._on_drag_start
        )
        
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

class MaterialPanel(ft.UserControl):
    """素材面板"""
    
    def __init__(
        self,
        on_material_select: Optional[Callable[[str], None]] = None,
        on_material_drag_start: Optional[Callable[[str, float, float], None]] = None,
        on_import: Optional[Callable[[], None]] = None
    ):
        super().__init__()
        self.on_material_select = on_material_select
        self.on_material_drag_start = on_material_drag_start
        self.on_import = on_import
        self._materials: Dict[str, Material] = {}
        self._search_text = ""
        self._grid_view = True
        
    def build(self):
        return ft.Column(
            [
                # 工具栏
                ft.Row(
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
                ),
                
                # 素材列表
                ft.GridView(
                    [
                        MaterialCard(
                            material=material,
                            on_select=self.on_material_select,
                            on_drag_start=self.on_material_drag_start
                        )
                        for material in self._filter_materials()
                    ],
                    runs_count=2 if self._grid_view else 1,
                    max_extent=200 if self._grid_view else 400,
                    child_aspect_ratio=1.0 if self._grid_view else 3.0,
                    spacing=10,
                    run_spacing=10
                )
            ],
            spacing=20
        )
        
    def _on_search_change(self, e):
        """搜索文本变化"""
        self._search_text = e.control.value.lower()
        self.update()
        
    def _toggle_view(self, _):
        """切换视图模式"""
        self._grid_view = not self._grid_view
        self.update()
        
    def _filter_materials(self) -> List[Material]:
        """过滤素材列表"""
        return [
            material for material in self._materials.values()
            if not self._search_text or
            self._search_text in material.name.lower() or
            self._search_text in material.description.lower()
        ]
        
    def update_materials(self, materials: Dict[str, Material]):
        """更新素材列表"""
        self._materials = materials
        self.update() 