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
from ...config.language import get_text

class HelpMenu(ft.UserControl):
    """帮助菜单"""
    
    def __init__(
        self,
        on_documentation: Optional[Callable[[], None]] = None,
        on_tutorial: Optional[Callable[[], None]] = None,
        on_feedback: Optional[Callable[[], None]] = None,
        on_about: Optional[Callable[[], None]] = None
    ):
        super().__init__()
        self.on_documentation = on_documentation
        self.on_tutorial = on_tutorial
        self.on_feedback = on_feedback
        self.on_about = on_about
        
    def build(self):
        return ft.PopupMenuButton(
            content=ft.Text(get_text("menu.help")),
            items=[
                # 文档
                ft.PopupMenuItem(
                    text=get_text("menu.help.documentation"),
                    icon=ft.icons.DESCRIPTION,
                    on_click=lambda _: (
                        self.on_documentation and self.on_documentation()
                    )
                ),
                
                # 教程
                ft.PopupMenuItem(
                    text=get_text("menu.help.tutorial"),
                    icon=ft.icons.SCHOOL,
                    on_click=lambda _: (
                        self.on_tutorial and self.on_tutorial()
                    )
                ),
                
                ft.PopupMenuDivider(),
                
                # 反馈
                ft.PopupMenuItem(
                    text=get_text("menu.help.feedback"),
                    icon=ft.icons.FEEDBACK,
                    on_click=lambda _: (
                        self.on_feedback and self.on_feedback()
                    )
                ),
                
                # 关于
                ft.PopupMenuItem(
                    text=get_text("menu.help.about"),
                    icon=ft.icons.INFO,
                    on_click=lambda _: (
                        self.on_about and self.on_about()
                    )
                ),
            ]
        ) 