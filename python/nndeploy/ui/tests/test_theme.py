import flet as ft
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from nndeploy.ui.config.theme import theme_config, ThemeType, get_color, get_style

class ThemeDemo:
    """主题演示应用"""
    
    def __init__(self):
        self.page = None
        
    def main(self, page: ft.Page):
        """主函数"""
        self.page = page
        page.title = "NNDeploy 主题演示"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.padding = 20
        page.spacing = 20
        page.scroll = ft.ScrollMode.AUTO
        
        # 创建主界面
        page.add(
            ft.Text("NNDeploy 主题演示", size=30, weight=ft.FontWeight.BOLD),
            ft.Text("这个演示展示了主题系统的颜色和样式", size=16),
            ft.Divider(),
            self.create_theme_switcher(),
            self.create_color_preview(),
            self.create_typography_preview(),
            self.create_component_preview()
        )
        
    def create_theme_switcher(self):
        """创建主题切换器"""
        def on_theme_change(e):
            if e.control.value:
                theme_config.switch_theme(ThemeType.DARK)
                self.page.theme_mode = ft.ThemeMode.DARK
            else:
                theme_config.switch_theme(ThemeType.LIGHT)
                self.page.theme_mode = ft.ThemeMode.LIGHT
            self.page.update()
            # 重新加载页面以应用新主题
            self.page.go(self.page.route)
        
        is_dark = theme_config.get_current_theme() == ThemeType.DARK
        
        return ft.Container(
            content=ft.Row([
                ft.Text("切换主题:", size=16),
                ft.Switch(
                    value=is_dark,
                    label="暗色主题" if is_dark else "亮色主题",
                    on_change=on_theme_change
                )
            ]),
            padding=10,
            border_radius=10,
            bgcolor=get_color("surface"),
            border=ft.border.all(1, get_color("border"))
        )
    
    def create_color_preview(self):
        """创建颜色预览"""
        color_groups = {
            "主色": ["primary", "primary_hover", "primary_pressed", "primary_disabled"],
            "次要色": ["secondary", "secondary_hover", "secondary_pressed", "secondary_disabled"],
            "背景色": ["background", "background_hover", "background_pressed", "background_disabled"],
            "表面色": ["surface", "surface_hover", "surface_pressed", "surface_disabled"],
            "边框色": ["border", "border_hover", "border_pressed", "border_disabled"],
            "文本色": ["text", "text_secondary", "text_disabled", "text_placeholder"],
            "功能色": ["error", "success", "warning", "info"]
        }
        
        color_cards = []
        
        for group_name, color_keys in color_groups.items():
            color_swatches = []
            
            for color_key in color_keys:
                color_value = get_color(color_key)
                color_swatches.append(
                    ft.Column([
                        ft.Container(
                            width=80,
                            height=40,
                            bgcolor=color_value,
                            border_radius=5,
                            border=ft.border.all(1, get_color("border"))
                        ),
                        ft.Text(color_key, size=12),
                        ft.Text(color_value, size=10, color=get_color("text_secondary"))
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
                )
            
            color_cards.append(
                ft.Container(
                    content=ft.Column([
                        ft.Text(group_name, size=16, weight=ft.FontWeight.BOLD),
                        ft.Row(color_swatches, wrap=True)
                    ]),
                    padding=10,
                    border_radius=10,
                    bgcolor=get_color("surface"),
                    border=ft.border.all(1, get_color("border")),
                    margin=ft.margin.only(bottom=10)
                )
            )
        
        return ft.Container(
            content=ft.Column([
                ft.Text("颜色系统", size=20, weight=ft.FontWeight.BOLD),
                ft.Column(color_cards)
            ]),
            padding=10,
            border_radius=10,
            bgcolor=get_color("background"),
            border=ft.border.all(1, get_color("border")),
            margin=ft.margin.only(top=10, bottom=10)
        )
    
    def create_typography_preview(self):
        """创建排版预览"""
        font_sizes = ["xs", "sm", "base", "lg", "xl", "2xl", "3xl", "4xl"]
        font_weights = ["thin", "light", "normal", "medium", "semibold", "bold", "black"]
        
        # 字体大小预览
        size_samples = []
        for size_key in font_sizes:
            size_value = get_style("font_sizes", size_key)
            size_samples.append(
                ft.Container(
                    content=ft.Row([
                        ft.Text(f"{size_key} ({size_value}px):", width=120),
                        ft.Text("示例文本", size=size_value)
                    ]),
                    padding=5
                )
            )
        
        # 字体粗细预览
        weight_samples = []
        for weight_key in font_weights:
            weight_value = get_style("font_weights", weight_key)
            weight = ft.FontWeight.NORMAL
            if weight_value <= 300:
                weight = ft.FontWeight.W_300
            elif weight_value <= 400:
                weight = ft.FontWeight.W_400
            elif weight_value <= 500:
                weight = ft.FontWeight.W_500
            elif weight_value <= 600:
                weight = ft.FontWeight.W_600
            elif weight_value <= 700:
                weight = ft.FontWeight.W_700
            elif weight_value <= 800:
                weight = ft.FontWeight.W_800
            else:
                weight = ft.FontWeight.W_900
                
            weight_samples.append(
                ft.Container(
                    content=ft.Row([
                        ft.Text(f"{weight_key} ({weight_value}):", width=120),
                        ft.Text("示例文本", weight=weight)
                    ]),
                    padding=5
                )
            )
        
        return ft.Container(
            content=ft.Column([
                ft.Text("排版系统", size=20, weight=ft.FontWeight.BOLD),
                ft.Container(
                    content=ft.Column([
                        ft.Text("字体大小", size=16, weight=ft.FontWeight.BOLD),
                        ft.Column(size_samples)
                    ]),
                    padding=10,
                    border_radius=10,
                    bgcolor=get_color("surface"),
                    border=ft.border.all(1, get_color("border")),
                    margin=ft.margin.only(bottom=10)
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Text("字体粗细", size=16, weight=ft.FontWeight.BOLD),
                        ft.Column(weight_samples)
                    ]),
                    padding=10,
                    border_radius=10,
                    bgcolor=get_color("surface"),
                    border=ft.border.all(1, get_color("border"))
                )
            ]),
            padding=10,
            border_radius=10,
            bgcolor=get_color("background"),
            border=ft.border.all(1, get_color("border")),
            margin=ft.margin.only(top=10, bottom=10)
        )
    
    def create_component_preview(self):
        """创建组件预览"""
        # 按钮预览
        buttons = ft.Row([
            ft.ElevatedButton("主按钮", bgcolor=get_color("primary"), color=get_color("background")),
            ft.OutlinedButton("次要按钮"),
            ft.TextButton("文本按钮"),
            ft.ElevatedButton("禁用按钮", disabled=True)
        ])
        
        # 输入框预览
        inputs = ft.Row([
            ft.TextField(label="标准输入框", hint_text="请输入内容"),
            ft.TextField(label="禁用输入框", hint_text="禁用状态", disabled=True),
            ft.Dropdown(
                label="下拉选择",
                options=[
                    ft.dropdown.Option("选项1"),
                    ft.dropdown.Option("选项2"),
                    ft.dropdown.Option("选项3"),
                ]
            )
        ])
        
        # 卡片预览
        cards = ft.Row([
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("卡片标题", size=16, weight=ft.FontWeight.BOLD),
                        ft.Text("卡片内容示例，展示卡片组件的样式。")
                    ]),
                    padding=15
                ),
                elevation=3
            ),
            ft.Container(
                content=ft.Column([
                    ft.Text("自定义卡片", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text("使用Container实现的卡片样式。")
                ]),
                padding=15,
                border_radius=10,
                bgcolor=get_color("surface"),
                border=ft.border.all(1, get_color("border")),
                shadow=ft.BoxShadow(
                    spread_radius=0,
                    blur_radius=5,
                    color=ft.colors.with_opacity(0.3, ft.colors.BLACK),
                    offset=ft.Offset(0, 2)
                )
            )
        ])
        
        return ft.Container(
            content=ft.Column([
                ft.Text("组件预览", size=20, weight=ft.FontWeight.BOLD),
                ft.Container(
                    content=ft.Column([
                        ft.Text("按钮", size=16, weight=ft.FontWeight.BOLD),
                        buttons
                    ]),
                    padding=10,
                    border_radius=10,
                    bgcolor=get_color("surface"),
                    border=ft.border.all(1, get_color("border")),
                    margin=ft.margin.only(bottom=10)
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Text("输入控件", size=16, weight=ft.FontWeight.BOLD),
                        inputs
                    ]),
                    padding=10,
                    border_radius=10,
                    bgcolor=get_color("surface"),
                    border=ft.border.all(1, get_color("border")),
                    margin=ft.margin.only(bottom=10)
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Text("卡片", size=16, weight=ft.FontWeight.BOLD),
                        cards
                    ]),
                    padding=10,
                    border_radius=10,
                    bgcolor=get_color("surface"),
                    border=ft.border.all(1, get_color("border"))
                )
            ]),
            padding=10,
            border_radius=10,
            bgcolor=get_color("background"),
            border=ft.border.all(1, get_color("border")),
            margin=ft.margin.only(top=10)
        )

if __name__ == "__main__":
    ft.app(ThemeDemo().main, view=ft.WEB_BROWSER, port=9090)
