"""
设置菜单模块

负责:
- 提供应用程序设置菜单
- 管理画布、节点、连线等设置
- 处理设置的保存和加载
- 提供设置面板的显示和隐藏

子菜单采用级联式设计,每种设置配有对应的小图标
"""

from typing import Optional, Callable, List
import flet as ft
from ...config.language import get_text
from ...config.settings import settings

class SettingsMenu(ft.UserControl):
    """设置菜单"""
    
    def __init__(
        self,
        on_canvas_settings: Optional[Callable[[], None]] = None,
        on_node_settings: Optional[Callable[[], None]] = None,
        on_edge_settings: Optional[Callable[[], None]] = None,
        on_performance_settings: Optional[Callable[[], None]] = None,
        on_auto_save_settings: Optional[Callable[[], None]] = None
    ):
        super().__init__()
        self.on_canvas_settings = on_canvas_settings
        self.on_node_settings = on_node_settings
        self.on_edge_settings = on_edge_settings
        self.on_performance_settings = on_performance_settings
        self.on_auto_save_settings = on_auto_save_settings
        
    def build(self):
        return ft.PopupMenuButton(
            content=ft.Text(get_text("menu.settings")),
            items=[
                # 画布设置
                ft.PopupMenuItem(
                    text=get_text("menu.settings.canvas"),
                    icon=ft.icons.GRID_ON,
                    on_click=lambda _: (
                        self.on_canvas_settings and self.on_canvas_settings()
                    ),
                    submenu=ft.PopupMenuButton(
                        items=self._build_canvas_settings()
                    )
                ),
                
                # 节点设置
                ft.PopupMenuItem(
                    text=get_text("menu.settings.node"),
                    icon=ft.icons.WIDGETS,
                    on_click=lambda _: (
                        self.on_node_settings and self.on_node_settings()
                    ),
                    submenu=ft.PopupMenuButton(
                        items=self._build_node_settings()
                    )
                ),
                
                # 连线设置
                ft.PopupMenuItem(
                    text=get_text("menu.settings.edge"),
                    icon=ft.icons.TIMELINE,
                    on_click=lambda _: (
                        self.on_edge_settings and self.on_edge_settings()
                    ),
                    submenu=ft.PopupMenuButton(
                        items=self._build_edge_settings()
                    )
                ),
                
                ft.PopupMenuDivider(),
                
                # 性能设置
                ft.PopupMenuItem(
                    text=get_text("menu.settings.performance"),
                    icon=ft.icons.SPEED,
                    on_click=lambda _: (
                        self.on_performance_settings and 
                        self.on_performance_settings()
                    ),
                    submenu=ft.PopupMenuButton(
                        items=self._build_performance_settings()
                    )
                ),
                
                # 自动保存设置
                ft.PopupMenuItem(
                    text=get_text("menu.settings.autoSave"),
                    icon=ft.icons.SAVE,
                    on_click=lambda _: (
                        self.on_auto_save_settings and 
                        self.on_auto_save_settings()
                    ),
                    submenu=ft.PopupMenuButton(
                        items=self._build_auto_save_settings()
                    )
                ),
            ]
        )
        
    def _build_canvas_settings(self) -> List[ft.PopupMenuItem]:
        """构建画布设置菜单项"""
        return [
            ft.PopupMenuItem(
                text=get_text("menu.settings.canvas.grid"),
                selected=settings.get("canvas", "grid_enabled"),
                on_click=lambda _: settings.set(
                    "canvas", "grid_enabled",
                    not settings.get("canvas", "grid_enabled")
                )
            ),
            ft.PopupMenuItem(
                text=get_text("menu.settings.canvas.snap"),
                selected=settings.get("canvas", "snap_to_grid"),
                on_click=lambda _: settings.set(
                    "canvas", "snap_to_grid",
                    not settings.get("canvas", "snap_to_grid")
                )
            ),
        ]
        
    def _build_node_settings(self) -> List[ft.PopupMenuItem]:
        """构建节点设置菜单项"""
        return [
            # TODO: 添加节点设置项
        ]
        
    def _build_edge_settings(self) -> List[ft.PopupMenuItem]:
        """构建连线设置菜单项"""
        return [
            # TODO: 添加连线设置项
        ]
        
    def _build_performance_settings(self) -> List[ft.PopupMenuItem]:
        """构建性能设置菜单项"""
        return [
            ft.PopupMenuItem(
                text=get_text("menu.settings.performance.animation"),
                selected=settings.get("performance", "animation_enabled"),
                on_click=lambda _: settings.set(
                    "performance", "animation_enabled",
                    not settings.get("performance", "animation_enabled")
                )
            ),
        ]
        
    def _build_auto_save_settings(self) -> List[ft.PopupMenuItem]:
        """构建自动保存设置菜单项"""
        return [
            ft.PopupMenuItem(
                text=get_text("menu.settings.autoSave.enabled"),
                selected=settings.get("auto_save", "enabled"),
                on_click=lambda _: settings.set(
                    "auto_save", "enabled",
                    not settings.get("auto_save", "enabled")
                )
            ),
        ] 