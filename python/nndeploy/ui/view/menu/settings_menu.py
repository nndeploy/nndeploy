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
from nndeploy.ui.config.language import language_config, get_text, Language
from nndeploy.ui.config.settings import settings
from nndeploy.ui.config.language import LanguageConfig
from nndeploy.ui.config.shortcuts import get_shortcut

def SettingsMenu(
    on_language_settings: Optional[Callable[[], None]] = None,
    on_theme_settings: Optional[Callable[[], None]] = None,
    on_auto_save_settings: Optional[Callable[[], None]] = None,
):
    """创建设置菜单组件
    
    使用Flet的SubmenuButton实现的设置菜单，支持多级菜单结构。
    每个设置项都有对应的图标和回调函数。
    """
    def build_auto_save_settings():
        """构建自动保存设置菜单项"""
        return ft.SubmenuButton(
            content=ft.Text("menu.settings.auto_save"),
            leading=ft.Icon(ft.Icons.SAVE),
            # on_click=lambda _: on_auto_save_settings and on_auto_save_settings(),
            controls=[
                ft.MenuItemButton(
                    content=ft.Text("menu.settings.auto_save.enabled"),
                    on_click=lambda e: settings.set(
                            "auto_save", "enabled", e.control.value
                        )
                ),
            ]
        )
        
    def build_language_settings():
        """构建语言设置菜单项"""
        # language_config = LanguageConfig.get_instance()
        current_language = language_config.get_current_language()
        
        def switch_language(lang_code):
            """切换语言并更新UI"""
            if lang_code == "zh":
                language_config.switch_language(Language.CHINESE)
            else:
                language_config.switch_language(Language.ENGLISH)
                
        return ft.SubmenuButton(
            content=ft.Text("menu.settings.language"),
            leading=ft.Icon(ft.Icons.LANGUAGE),
            # on_click=lambda _: on_language_settings and on_language_settings(),
            controls=[
                ft.MenuItemButton(
                    content=ft.Text("menu.settings.language.chinese"),
                    leading=ft.Icon(
                        ft.icons.CHECK if current_language == Language.CHINESE else ft.icons.CIRCLE,
                        color=ft.colors.GREEN if current_language == Language.CHINESE else ft.colors.TRANSPARENT
                    ),
                    on_click=lambda _: switch_language("zh")
                ),
                ft.MenuItemButton(
                    content=ft.Text("menu.settings.language.english"),
                    leading=ft.Icon(
                        ft.icons.CHECK if current_language == Language.ENGLISH else ft.icons.CIRCLE,
                        color=ft.colors.GREEN if current_language == Language.ENGLISH else ft.colors.TRANSPARENT
                    ),
                    on_click=lambda _: switch_language("en")
                ),
            ]
        )
        
    def build_theme_settings():
        """构建主题设置菜单项"""
        return ft.SubmenuButton(
            content=ft.Text("menu.settings.theme"),
            leading=ft.Icon(ft.Icons.PALETTE),
            # on_click=lambda _: on_theme_settings and on_theme_settings(),
            controls=[
                ft.MenuItemButton(
                    content=ft.Text("menu.settings.theme.light"),
                    leading=ft.Icon(ft.Icons.LIGHT_MODE),
                    on_click=lambda _: settings.set("theme", "mode", "light")
                ),
                ft.MenuItemButton(
                    content=ft.Text("menu.settings.theme.dark"),
                    leading=ft.Icon(ft.Icons.DARK_MODE),
                    on_click=lambda _: settings.set("theme", "mode", "dark")
                ),
                ft.MenuItemButton(
                    content=ft.Text("menu.settings.theme.system"),
                    leading=ft.Icon(ft.Icons.SETTINGS_SUGGEST),
                    on_click=lambda _: settings.set("theme", "mode", "system")
                ),
            ]
        )

    # 创建主设置菜单
    return ft.MenuBar(
        controls=[
            ft.SubmenuButton(
                content=ft.Text(get_text("menu.settings")),
                tooltip=get_text("menu.settings"),
                controls=[                    
                    # 自动保存设置
                    build_auto_save_settings(),
                    
                    # 语言设置
                    build_language_settings(),
                    
                    # 主题设置
                    build_theme_settings()
                ]
            )
        ]
    )
    
def main(page: ft.Page):   
    page.title = "NNDeploy 设置菜单演示"
    page.add(SettingsMenu())
    
    
if __name__ == "__main__":
    ft.app(target=main, view=ft.WEB_BROWSER, port=8080)