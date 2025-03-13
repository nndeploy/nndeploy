"""
文件菜单模块

负责:
- 提供文件相关操作菜单
- 处理新建、打开、保存等功能
- 管理最近文件列表
- 处理文件导入导出

菜单项采用深色文字配合浅色背景,鼠标悬停时呈现轻微高亮效果
"""

from typing import List, Optional, Callable
import flet as ft
from nndeploy.ui.config import get_text
from nndeploy.ui.service import file_service
from ...config.shortcuts import get_shortcut

class FileMenu(ft.UserControl):
    """文件菜单"""
    
    def __init__(
        self,
        on_new: Optional[Callable[[], None]] = None,
        on_open: Optional[Callable[[], None]] = None,
        on_save: Optional[Callable[[], None]] = None,
        on_save_as: Optional[Callable[[], None]] = None,
        on_import: Optional[Callable[[], None]] = None,
        on_export: Optional[Callable[[], None]] = None,
        recent_files: List[str] = None
    ):
        super().__init__()
        self.on_new = on_new
        self.on_open = on_open
        self.on_save = on_save
        self.on_save_as = on_save_as
        self.on_import = on_import
        self.on_export = on_export
        self._recent_files = recent_files or []
        
    def build(self):
        return ft.PopupMenuButton(
            content=ft.Text(get_text("menu.file")),
            items=[
                # 新建
                ft.PopupMenuItem(
                    text=get_text("menu.file.new"),
                    icon=ft.icons.ADD,
                    on_click=lambda _: self.on_new and self.on_new(),
                    shortcut=str(get_shortcut("new_workflow"))
                ),
                
                # 打开
                ft.PopupMenuItem(
                    text=get_text("menu.file.open"),
                    icon=ft.icons.FOLDER_OPEN,
                    on_click=lambda _: self.on_open and self.on_open(),
                    shortcut=str(get_shortcut("open_workflow"))
                ),
                
                # 最近文件
                self._build_recent_files_menu(),
                
                ft.PopupMenuDivider(),
                
                # 保存
                ft.PopupMenuItem(
                    text=get_text("menu.file.save"),
                    icon=ft.icons.SAVE,
                    on_click=lambda _: self.on_save and self.on_save(),
                    shortcut=str(get_shortcut("save_workflow"))
                ),
                
                # 另存为
                ft.PopupMenuItem(
                    text=get_text("menu.file.saveAs"),
                    icon=ft.icons.SAVE_AS,
                    on_click=lambda _: self.on_save_as and self.on_save_as(),
                    shortcut=str(get_shortcut("save_workflow_as"))
                ),
                
                ft.PopupMenuDivider(),
                
                # 导入
                ft.PopupMenuItem(
                    text=get_text("menu.file.import"),
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=lambda _: self.on_import and self.on_import()
                ),
                
                # 导出
                ft.PopupMenuItem(
                    text=get_text("menu.file.export"),
                    icon=ft.icons.DOWNLOAD,
                    on_click=lambda _: self.on_export and self.on_export()
                ),
            ]
        )
        
    def _build_recent_files_menu(self) -> ft.PopupMenuItem:
        """构建最近文件子菜单"""
        return ft.PopupMenuItem(
            text=get_text("menu.file.recent"),
            icon=ft.icons.HISTORY,
            disabled=len(self._recent_files) == 0,
            submenu=ft.PopupMenuButton(
                items=[
                    ft.PopupMenuItem(
                        text=path,
                        on_click=lambda _, p=path: self.on_open and self.on_open(p)
                    )
                    for path in self._recent_files
                ]
            ) if self._recent_files else None
        )
        
    def update_recent_files(self, files: List[str]):
        """更新最近文件列表"""
        self._recent_files = files
        self.update() 