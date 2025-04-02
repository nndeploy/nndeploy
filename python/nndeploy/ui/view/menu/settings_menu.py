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
from nndeploy.ui.config.language import Language, get_text, switch_language, get_current_language, get_supported_languages, language_manager
from nndeploy.ui.config.theme import ThemeType, switch_theme


def SettingsMenu(
    on_language_settings: Optional[Callable[[], None]] = None, # 语言设置回调函数（通常不用提供）
    on_theme_settings: Optional[Callable[[], None]] = None, # 主题设置回调函数
):
    """创建设置菜单组件
    
    使用Flet的SubmenuButton实现的设置菜单，支持多级菜单结构。
    每个设置项都有对应的图标和回调函数。
    """        
    def build_language_settings():
        """构建语言设置菜单项"""
        current_language = get_current_language()
                
        return ft.SubmenuButton(
            content=ft.Text(get_text("menu.settings.language")),
            leading=ft.Icon(ft.Icons.LANGUAGE),
            controls=[
                ft.MenuItemButton(
                    content=ft.Text(get_text("menu.settings.language.chinese")),
                    leading=ft.Icon(
                        ft.Icons.CHECK if current_language == Language.CHINESE else ft.Icons.CIRCLE,
                        color=ft.Colors.GREEN if current_language == Language.CHINESE else ft.Colors.TRANSPARENT
                    ),
                    on_click=lambda e: handle_language_switch(Language.CHINESE)
                ),
                ft.MenuItemButton(
                    content=ft.Text(get_text("menu.settings.language.english")),
                    leading=ft.Icon(
                        ft.Icons.CHECK if current_language == Language.ENGLISH else ft.Icons.CIRCLE,
                        color=ft.Colors.GREEN if current_language == Language.ENGLISH else ft.Colors.TRANSPARENT
                    ),
                    on_click=lambda e: handle_language_switch(Language.ENGLISH)
                ),
            ]
        )
        
    def handle_language_switch(language: Language):
        """处理语言切换，并通知观察者"""
        switch_language(language)
        if on_language_settings:
            on_language_settings()
            
    # 添加语言变化观察者
    def language_change_observer(language: Language):
        """语言变化观察者回调函数"""
        # 在这里可以处理语言变化后的全局更新
        print(f"语言变化: {language}")
        # 更新UI
        if page:
            page.update()
        pass
        
    # 注册语言变化观察者
    language_manager.add_observer(language_change_observer)
        
    def build_theme_settings():
        """构建主题设置菜单项"""
        return ft.SubmenuButton(
            content=ft.Text(get_text("menu.settings.theme")),
            leading=ft.Icon(ft.Icons.PALETTE),
            controls=[
                ft.MenuItemButton(
                    content=ft.Text(get_text("menu.settings.theme.light")),
                    leading=ft.Icon(ft.Icons.LIGHT_MODE),
                    on_click=lambda e: switch_theme(ThemeType.LIGHT)
                ),
                ft.MenuItemButton(
                    content=ft.Text(get_text("menu.settings.theme.dark")),
                    leading=ft.Icon(ft.Icons.DARK_MODE),
                    on_click=lambda e: switch_theme(ThemeType.DARK)
                ),
                ft.MenuItemButton(
                    content=ft.Text(get_text("menu.settings.theme.system")),
                    leading=ft.Icon(ft.Icons.SETTINGS_SUGGEST),
                    on_click=lambda e: switch_theme(ThemeType.SYSTEM)
                ),
            ]
        )

    # 创建主设置菜单
    return ft.MenuBar(
        controls=[
            ft.SubmenuButton(
                content=ft.Text(get_text("menu.settings")),
                controls=[                                     
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
    page.update()
    
    
if __name__ == "__main__":
    ft.app(target=main, view=ft.WEB_BROWSER, port=8080)