"""
编辑菜单模块

负责:
- 提供编辑相关操作菜单
- 处理撤销、重做、复制、粘贴等功能
- 管理编辑操作状态
- 提供编辑快捷键支持

不可用选项显示为灰色,当前可执行操作高亮显示
"""

from typing import Optional, Callable
import flet as ft
from nndeploy.ui.config.language import get_text
from nndeploy.ui.config.shortcuts import get_shortcut

def EditMenu(
    on_undo: Optional[Callable[[], None]] = None,
    on_redo: Optional[Callable[[], None]] = None,
    on_cut: Optional[Callable[[], None]] = None,
    on_copy: Optional[Callable[[], None]] = None,
    on_paste: Optional[Callable[[], None]] = None,
    on_delete: Optional[Callable[[], None]] = None,
    on_select_all: Optional[Callable[[], None]] = None,
    can_undo: bool = False,
    can_redo: bool = False,
    has_selection: bool = False,
    has_clipboard: bool = False
):
    return ft.PopupMenuButton(
        content=ft.Text(get_text("menu.edit")),
        items=[
            # 撤销
            ft.PopupMenuItem(
                text=get_text("menu.edit.undo"),
                icon=ft.icons.UNDO,
                on_click=lambda _: on_undo and on_undo(),
                disabled=not can_undo
            ),
            
            # 重做
            ft.PopupMenuItem(
                text=get_text("menu.edit.redo"), 
                icon=ft.icons.REDO,
                on_click=lambda _: on_redo and on_redo(),
                disabled=not can_redo
            ),
            
            ft.Divider(),
            
            # 剪切
            ft.PopupMenuItem(
                text=get_text("menu.edit.cut"),
                icon=ft.icons.CONTENT_CUT,
                on_click=lambda _: on_cut and on_cut(),
                disabled=not has_selection
            ),
            
            # 复制
            ft.PopupMenuItem(
                text=get_text("menu.edit.copy"),
                icon=ft.icons.CONTENT_COPY,
                on_click=lambda _: on_copy and on_copy(),
                disabled=not has_selection
            ),
            
            # 粘贴
            ft.PopupMenuItem(
                text=get_text("menu.edit.paste"),
                icon=ft.icons.CONTENT_PASTE,
                on_click=lambda _: on_paste and on_paste(),
                disabled=not has_clipboard
            ),
            
            ft.Divider(),
            
            # 删除
            ft.PopupMenuItem(
                text=get_text("menu.edit.delete"),
                icon=ft.icons.DELETE,
                on_click=lambda _: on_delete and on_delete(),
                disabled=not has_selection
            ),
            
            # 全选
            ft.PopupMenuItem(
                text=get_text("menu.edit.selectAll"),
                icon=ft.icons.SELECT_ALL,
                on_click=lambda _: on_select_all and on_select_all()
            ),
        ],
        menu_position=ft.PopupMenuPosition.UNDER # 设置菜单在按钮下方显示
    )
    
def main(page: ft.Page):   
    page.title = "NNDeploy 文件菜单演示"
    page.add(EditMenu())
    
    
if __name__ == "__main__":
    ft.app(target=main, view=ft.WEB_BROWSER, port=9090)