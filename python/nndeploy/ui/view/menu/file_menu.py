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
from nndeploy.ui.config.shortcuts import get_shortcut

def FileMenu(
    on_new: Optional[Callable[[], None]] = None,
    on_open: Optional[Callable[[], None]] = None,
    on_save: Optional[Callable[[], None]] = None,
    on_save_as: Optional[Callable[[], None]] = None,
    on_import: Optional[Callable[[], None]] = None,
    on_export: Optional[Callable[[], None]] = None,
    recent_files: List[str] = None
):
    recent_files = recent_files or []
    
    def build_recent_files_menu():
        """构建最近文件子菜单"""
        if len(recent_files) == 0:
            return ft.PopupMenuItem(
                text=get_text("menu.file.recent"),
                icon=ft.Icons.HISTORY,
                disabled=True
            )
        
        # 创建最近文件菜单项
        recent_menu_items = [
            ft.PopupMenuItem(
                text=path,
                on_click=lambda _, p=path: on_open and on_open(p)
            )
            for path in recent_files
        ]
        
        # 使用content_menu而不是submenu
        return ft.PopupMenuItem(
            text=get_text("menu.file.recent"),
            icon=ft.Icons.HISTORY,
            # on_click=lambda _, p=path: on_open and on_open(p)
        )

    return ft.PopupMenuButton(
        content=ft.Text(get_text("menu.file")),
        items=[
            # 新建
            ft.PopupMenuItem(
                text=get_text("menu.file.new"),
                icon=ft.Icons.ADD,
                on_click=lambda _: on_new and on_new()
            ),
            
            # 打开
            ft.PopupMenuItem(
                text=get_text("menu.file.open"),
                icon=ft.Icons.FOLDER_OPEN,
                on_click=lambda _: on_open and on_open()
            ),
            
            # 最近文件
            build_recent_files_menu(),
            
            ft.Divider(),
            
            # 保存
            ft.PopupMenuItem(
                text=get_text("menu.file.save"),
                icon=ft.Icons.SAVE,
                on_click=lambda _: on_save and on_save()
            ),
            
            # 另存为
            ft.PopupMenuItem(
                text=get_text("menu.file.saveAs"),
                icon=ft.Icons.SAVE_AS,
                on_click=lambda _: on_save_as and on_save_as()
            ),
            
            ft.Divider(),
            
            # 导入
            ft.PopupMenuItem(
                text=get_text("menu.file.import"),
                icon=ft.Icons.UPLOAD_FILE,
                on_click=lambda _: on_import and on_import()
            ),
            
            # 导出
            ft.PopupMenuItem(
                text=get_text("menu.file.export"),
                icon=ft.Icons.DOWNLOAD,
                on_click=lambda _: on_export and on_export()
            ),
        ],
        menu_position=ft.PopupMenuPosition.UNDER # 设置菜单在按钮下方显示
    )

def update_recent_files(files: List[str]):
    """更新最近文件列表"""
    if not files:
        return
    
    # 更新文件服务中的最近文件列表
    file_service.get_instance().set_recent_files(files)
    
    # 通知UI更新
    if hasattr(update_recent_files, 'callback'):
        update_recent_files.callback(files)

def register_recent_files_callback(callback: Callable[[List[str]], None]):
    """注册最近文件更新回调函数"""
    update_recent_files.callback = callback
    
    
def main(page: ft.Page):
    """主函数，接收page参数"""
    update_recent_files.callback = lambda files: print(f"最近文件更新: {files}")
    register_recent_files_callback(lambda files: print(f"最近文件更新: {files}"))
    
    page.title = "NNDeploy 文件菜单演示"
    # page.theme = ft.Theme(color_scheme="dark")
    # page.theme_mode = ft.ThemeMode.DARK
    page.add(FileMenu())
    
    
if __name__ == "__main__":
    ft.app(target=main, view=ft.WEB_BROWSER, port=9090)