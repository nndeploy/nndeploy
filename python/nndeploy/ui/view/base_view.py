from flet import View, Page, ThemeMode
import flet
import settings
from .theme import ThemeManager 

class BaseView:
    def __init__(self, page: Page):
        self.page = page
        self.page.theme_mode = ThemeMode.LIGHT
        self.theme = ThemeManager.get_theme()
        self.theme_button = None
        
    def _toggle_theme(self, e):
        self.page.theme_mode = (
            ThemeMode.LIGHT if self.page.theme_mode == ThemeMode.DARK else ThemeMode.DARK
        )
        # 更新主题按钮图标
        if self.theme_button:
            current_theme = "dark" if self.page.theme_mode == ThemeMode.DARK else "light"
            self.theme_button.icon = flet.icons.DARK_MODE if current_theme == "light" else flet.icons.LIGHT_MODE
            self.theme_button.icon_color = self.get_theme_color("text")
        
        # 调用主题更新回调
        self.on_theme_changed()
        self.page.update()
    
    def on_theme_changed(self):
        """主题改变时的回调方法，子类可以重写此方法来更新颜色"""
        pass

    def _get_app_bar(self):
        """获取统一的顶部导航栏"""
        from . import common
        current_theme = "dark" if self.page.theme_mode == ThemeMode.DARK else "light"
        # 创建主题切换按钮并保存引用
        self.theme_button = flet.IconButton(
            icon=flet.icons.DARK_MODE if current_theme == "light" else flet.icons.LIGHT_MODE,
            tooltip="切换主题",
            on_click=self._toggle_theme,
            icon_color=self.get_theme_color("text"),
        )
        
        return flet.AppBar(
            leading=common.AnimatedIconTextButton(
                "nndeploy",
                flet.Icons.WB_SUNNY_OUTLINED,
                flet.Icons.WB_SUNNY_OUTLINED,
                on_click=lambda _: self.page.go("/")
            ),
            leading_width=120,
            center_title=True,
            title=flet.Row(
                controls=[
                    flet.TextButton(text="GitHub", on_click=lambda _: self.page.launch_url(settings.github_url)),
                    flet.TextButton(text="Docs", on_click=lambda _: self.page.launch_url(settings.docs_url)),
                    flet.TextButton(text="About US", on_click=lambda _: self.page.go("/about")),
                ],
                alignment=flet.MainAxisAlignment.CENTER,
            ),
            actions=[
                self.theme_button,
            ],
            bgcolor=self.get_theme_color("toolbar"),
            toolbar_opacity=1.0,
            shape=flet.RoundedRectangleBorder(
                radius=flet.BorderRadius(
                    top_left=0,
                    top_right=0,
                    bottom_left=10,
                    bottom_right=10
                )
            ),
        )
    
    def get_current_theme(self):
        """获取当前主题配置"""
        current_theme = "dark" if self.page.theme_mode == ThemeMode.DARK else "light"
        return self.theme[current_theme]
    
    def get_theme_color(self, color_key: str):
        """获取当前主题下指定的颜色
        Args:
            color_key: 颜色键名，如 'text', 'background', 'toolbar' 等
        """
        return self.get_current_theme()[color_key]