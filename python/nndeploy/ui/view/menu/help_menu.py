"""
帮助菜单模块

负责:
- 提供帮助和支持菜单
- 管理操作文档和教程
- 处理用户反馈
- 显示关于信息

帮助图标采用问号设计,菜单项简洁明了
"""

from typing import Optional, Callable
import flet as ft
from nndeploy.ui.config.language import get_text

def HelpMenu(
    on_documentation: Optional[Callable[[], None]] = None,
    on_tutorial: Optional[Callable[[], None]] = None,
    on_feedback: Optional[Callable[[], None]] = None,
    on_about: Optional[Callable[[], None]] = None
):
    return ft.PopupMenuButton(
        content=ft.Text(get_text("menu.help")),
        items=[
            # 文档
            ft.PopupMenuItem(
                text=get_text("menu.help.documentation"),
                icon=ft.icons.DESCRIPTION,
                on_click=lambda _: (
                    on_documentation and on_documentation()
                )
            ),
            
            # 教程
            ft.PopupMenuItem(
                text=get_text("menu.help.tutorial"),
                icon=ft.icons.SCHOOL,
                on_click=lambda _: (
                    on_tutorial and on_tutorial()
                )
            ),
            
            ft.Divider(),
            
            # 反馈
            ft.PopupMenuItem(
                text=get_text("menu.help.feedback"),
                icon=ft.icons.FEEDBACK,
                on_click=lambda _: (
                    on_feedback and on_feedback()
                )
            ),
            
            # 关于
            ft.PopupMenuItem(
                text=get_text("menu.help.about"),
                icon=ft.icons.INFO,
                on_click=lambda _: (
                    on_about and on_about()
                )
            ),
        ],
        menu_position=ft.PopupMenuPosition.UNDER # 设置菜单在按钮下方显示
    ) 
    

def main(page: ft.Page):
    page.add(HelpMenu())

if __name__ == "__main__":
    ft.app(target=main, view=ft.WEB_BROWSER, port=9090)