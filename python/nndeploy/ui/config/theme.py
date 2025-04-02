"""
主题配置模块

负责:
- 定义应用的颜色方案
- 管理字体和排版样式
- 提供组件样式定义
- 支持主题切换和自定义

采用dify的设计风格,支持明暗两套主题
"""

from enum import Enum
from typing import Dict, Any, List, Callable
import flet as ft
from pathlib import Path
import json

class ThemeType(Enum):
    """主题类型枚举"""
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"
# 主题颜色定义 - 全局字典实现
THEME_COLORS = {
    # 明亮主题
    "light": {
        # 主色
        "primary": "#2970FF",
        "primary_hover": "#1C54DB", 
        "primary_pressed": "#1542AF",
        "primary_disabled": "#A4CAFE",
        
        # 次要色
        "secondary": "#6B7280",
        "secondary_hover": "#4B5563",
        "secondary_pressed": "#374151",
        "secondary_disabled": "#D1D5DB",
        
        # 背景色
        "background": "#FFFFFF",
        "background_hover": "#F9FAFB",
        "background_pressed": "#F3F4F6",
        "background_disabled": "#F9FAFB",
        
        # 表面色
        "surface": "#F3F4F6",
        "surface_hover": "#E5E7EB",
        "surface_pressed": "#D1D5DB",
        "surface_disabled": "#F3F4F6",
        
        # 边框色
        "border": "#E5E7EB",
        "border_hover": "#D1D5DB",
        "border_pressed": "#9CA3AF",
        "border_disabled": "#F3F4F6",
        
        # 文本色
        "text": "#111827",
        "text_secondary": "#6B7280",
        "text_disabled": "#9CA3AF",
        "text_placeholder": "#9CA3AF",
        
        # 功能色
        "error": "#DC2626",
        "error_hover": "#B91C1C",
        "error_pressed": "#991B1B",
        "error_disabled": "#FCA5A5",
        
        "success": "#059669",
        "success_hover": "#047857",
        "success_pressed": "#065F46",
        "success_disabled": "#6EE7B7",
        
        "warning": "#D97706", 
        "warning_hover": "#B45309",
        "warning_pressed": "#92400E",
        "warning_disabled": "#FCD34D",
        
        "info": "#2563EB",
        "info_hover": "#1D4ED8",
        "info_pressed": "#1E40AF",
        "info_disabled": "#93C5FD",
    },
    
    # 暗色主题
    "dark": {
        # 主色
        "primary": "#60A5FA",
        "primary_hover": "#3B82F6",
        "primary_pressed": "#2563EB", 
        "primary_disabled": "#BFDBFE",
        
        # 次要色
        "secondary": "#9CA3AF",
        "secondary_hover": "#6B7280",
        "secondary_pressed": "#4B5563",
        "secondary_disabled": "#E5E7EB",
        
        # 背景色
        "background": "#1F2937",
        "background_hover": "#111827",
        "background_pressed": "#0F172A",
        "background_disabled": "#1F2937",
        
        # 表面色
        "surface": "#374151",
        "surface_hover": "#4B5563",
        "surface_pressed": "#6B7280",
        "surface_disabled": "#374151",
        
        # 边框色
        "border": "#4B5563",
        "border_hover": "#6B7280",
        "border_pressed": "#9CA3AF",
        "border_disabled": "#374151",
        
        # 文本色
        "text": "#F9FAFB",
        "text_secondary": "#D1D5DB",
        "text_disabled": "#9CA3AF",
        "text_placeholder": "#6B7280",
        
        # 功能色
        "error": "#EF4444",
        "error_hover": "#DC2626",
        "error_pressed": "#B91C1C",
        "error_disabled": "#FCA5A5",
        
        "success": "#10B981",
        "success_hover": "#059669",
        "success_pressed": "#047857",
        "success_disabled": "#6EE7B7",
        
        "warning": "#F59E0B",
        "warning_hover": "#D97706",
        "warning_pressed": "#B45309",
        "warning_disabled": "#FCD34D",
        
        "info": "#3B82F6",
        "info_hover": "#2563EB",
        "info_pressed": "#1D4ED8",
        "info_disabled": "#93C5FD",
    }
}

# 主题样式定义 - 全局字典实现
THEME_STYLES = {
    # 字体设置
    "fonts": {
        "default": "Roboto",
        "monospace": "Roboto Mono",
        "heading": "Roboto",
        "body": "Roboto"
    },
    
    # 字体大小
    "font_sizes": {
        "xs": 12,
        "sm": 14,
        "base": 16,
        "lg": 18,
        "xl": 20,
        "2xl": 24,
        "3xl": 30,
        "4xl": 36,
        "5xl": 48,
        "6xl": 60,
    },
    
    # 字重
    "font_weights": {
        "thin": 100,
        "extralight": 200,
        "light": 300,
        "normal": 400,
        "medium": 500,
        "semibold": 600,
        "bold": 700,
        "extrabold": 800,
        "black": 900,
    },
    
    # 行高
    "line_heights": {
        "none": 1,
        "tight": 1.25,
        "snug": 1.375,
        "normal": 1.5,
        "relaxed": 1.625,
        "loose": 2,
    },
    
    # 字间距
    "letter_spacings": {
        "tighter": -0.05,
        "tight": -0.025,
        "normal": 0,
        "wide": 0.025,
        "wider": 0.05,
        "widest": 0.1,
    },
    
    # 圆角大小
    "border_radius": {
        "none": 0,
        "sm": 4,
        "base": 6,
        "md": 8,
        "lg": 12,
        "xl": 16,
        "2xl": 24,
        "3xl": 32,
        "full": 9999
    },
    
    # 阴影样式
    "shadows": {
        "none": "none",
        "sm": "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
        "base": "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1)",
        "md": "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)",
        "lg": "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1)",
        "xl": "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)",
        "2xl": "0 25px 50px -12px rgba(0, 0, 0, 0.25)",
        "inner": "inset 0 2px 4px 0 rgba(0, 0, 0, 0.05)",
    },
    
    # 动画时长
    "animation_duration": {
        "none": 0,
        "fast": 150,
        "normal": 250,
        "slow": 350,
        "slower": 500,
        "slowest": 700,
    },
    
    # 动画曲线
    "animation_curves": {
        "linear": "linear",
        "ease": "ease",
        "ease-in": "ease-in",
        "ease-out": "ease-out",
        "ease-in-out": "ease-in-out",
    },
    
    # 间距
    "spacing": {
        "0": 0,
        "1": 4,
        "2": 8,
        "3": 12,
        "4": 16,
        "5": 20,
        "6": 24,
        "8": 32,
        "10": 40,
        "12": 48,
        "16": 64,
        "20": 80,
        "24": 96,
        "32": 128,
    }
}

class ThemeConfig:
    """主题配置类"""
    
    def __init__(self):
        self._current_theme = ThemeType.LIGHT
        self._observers: List[Callable[[ThemeType], None]] = []
        self._custom_colors: Dict[str, str] = {}
        self._custom_styles: Dict[str, Any] = {}
        self._load_custom_theme()
        
    def _load_custom_theme(self):
        """加载自定义主题配置"""
        config_path = Path(__file__).parent / "../config/custom_theme.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._custom_colors = data.get("colors", {})
                    self._custom_styles = data.get("styles", {})
            except Exception as e:
                print(f"加载自定义主题配置失败: {e}")
        else:
            self._custom_colors = THEME_COLORS.get(self._current_theme.value, {})
            self._custom_styles = THEME_STYLES
                
    def _save_custom_theme(self):
        """保存自定义主题配置"""
        config_path = Path(__file__).parent / "../config/custom_theme.json"
        try:
            data = {
                "colors": self._custom_colors,
                "styles": self._custom_styles
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"保存自定义主题配置失败: {e}")
        
    def get_color(self, color_key: str) -> str:
        """获取主题颜色
        
        Args:
            color_key: 颜色键值
            
        Returns:
            颜色值
        """
        theme_name = self._current_theme.value
        colors = THEME_COLORS.get(theme_name, {})
        return self._custom_colors.get(color_key, colors.get(color_key))
        
    def get_style(self, style_type: str, key: str) -> Any:
        """获取主题样式
        
        Args:
            style_type: 样式类型(fonts/font_sizes/border_radius等)
            key: 样式键值
            
        Returns:
            样式值
        """
        style_map = THEME_STYLES.get(style_type.lower(), {})
        custom_styles = self._custom_styles.get(style_type, {})
        return custom_styles.get(key, style_map.get(key))
        
    def switch_theme(self, theme_type: ThemeType):
        """切换主题
        
        Args:
            theme_type: 目标主题类型
            
        Raises:
            ValueError: 如果指定了不支持的主题类型
        """
        if not isinstance(theme_type, ThemeType):
            raise ValueError(f"不支持的主题类型: {theme_type}")
            
        if theme_type != self._current_theme:
            # print(f"切换主题: {theme_type}")
            self._current_theme = theme_type
            self._custom_colors = THEME_COLORS.get(self._current_theme.value, {})
            for observer in self._observers:
                print(f"通知观察者: {observer}")
                observer(theme_type)
                
    def get_current_theme(self) -> ThemeType:
        """获取当前主题类型"""
        return self._current_theme
        
    def set_custom_color(self, color_key: str, color_value: str):
        """设置自定义颜色
        
        Args:
            color_key: 颜色键值
            color_value: 颜色值(十六进制格式)
        """
        self._custom_colors[color_key] = color_value
        self._save_custom_theme()
        
    def set_custom_style(self, style_type: str, key: str, value: Any):
        """设置自定义样式
        
        Args:
            style_type: 样式类型
            key: 样式键值
            value: 样式值
        """
        if style_type not in self._custom_styles:
            self._custom_styles[style_type] = {}
        self._custom_styles[style_type][key] = value
        self._save_custom_theme()
        
    def add_observer(self, observer: Callable[[ThemeType], None]):
        """添加主题变化观察者
        
        Args:
            observer: 观察者回调函数,接收新的主题类型作为参数
        """
        if observer not in self._observers:
            self._observers.append(observer)
            
    def remove_observer(self, observer: Callable[[ThemeType], None]):
        """移除主题变化观察者
        
        Args:
            observer: 要移除的观察者回调函数
        """
        if observer in self._observers:
            self._observers.remove(observer)
            
    def get_flet_theme(self) -> ft.Theme:
        """获取Flet主题对象"""
        # 创建一个基本的Theme对象
        theme = ft.Theme(
            # 主题色
            color_scheme_seed=self.get_color("primary"),
            
            # 文本主题
            text_theme=ft.TextTheme(
                # 标题
                display_large=ft.TextStyle(
                    font_family=self.get_style("fonts", "heading"),
                    size=self.get_style("font_sizes", "4xl"),
                    weight=ft.FontWeight.BOLD,
                    color=self.get_color("text"),
                ),
                display_medium=ft.TextStyle(
                    font_family=self.get_style("fonts", "heading"),
                    size=self.get_style("font_sizes", "3xl"),
                    weight=ft.FontWeight.BOLD,
                    color=self.get_color("text"),
                ),
                display_small=ft.TextStyle(
                    font_family=self.get_style("fonts", "heading"),
                    size=self.get_style("font_sizes", "2xl"),
                    weight=ft.FontWeight.BOLD,
                    color=self.get_color("text"),
                ),
                
                # 正文
                body_large=ft.TextStyle(
                    font_family=self.get_style("fonts", "body"),
                    size=self.get_style("font_sizes", "lg"),
                    color=self.get_color("text"),
                ),
                body_medium=ft.TextStyle(
                    font_family=self.get_style("fonts", "body"), 
                    size=self.get_style("font_sizes", "base"),
                    color=self.get_color("text"),
                ),
                body_small=ft.TextStyle(
                    font_family=self.get_style("fonts", "body"),
                    size=self.get_style("font_sizes", "sm"),
                    color=self.get_color("text_secondary"),
                ),
            ),
            
            # 视觉密度
            visual_density=ft.VisualDensity.COMFORTABLE,
        )
        
        return theme

# 创建全局主题配置实例
theme_config = ThemeConfig()

# 便捷函数
def get_color(color_key: str) -> str:
    """获取主题颜色的便捷函数
    
    Args:
        color_key: 颜色键值
        
    Returns:
        颜色值
    """
    return theme_config.get_color(color_key)

def get_style(style_type: str, key: str) -> Any:
    """获取主题样式的便捷函数
    
    Args:
        style_type: 样式类型
        key: 样式键值
        
    Returns:
        样式值
    """
    return theme_config.get_style(style_type, key) 

def switch_theme(theme_type: ThemeType):
    """切换主题的便捷函数
    
    Args:
        theme_type: 目标主题类型
    """
    theme_config.switch_theme(theme_type)

def get_current_theme() -> ThemeType:
    """获取当前主题类型的便捷函数
    
    Returns:
        当前主题类型
    """
    return theme_config.get_current_theme()

def set_custom_color(color_key: str, color_value: str):
    """设置自定义颜色的便捷函数
    
    Args:
        color_key: 颜色键值
        color_value: 颜色值(十六进制格式)
    """
    theme_config.set_custom_color(color_key, color_value)

def set_custom_style(style_type: str, key: str, value: Any):
    """设置自定义样式的便捷函数
    
    Args:
        style_type: 样式类型
        key: 样式键值
        value: 样式值
    """
    theme_config.set_custom_style(style_type, key, value)

def get_flet_theme() -> ft.Theme:
    """获取Flet主题对象的便捷函数
    
    Returns:
        Flet主题对象
    """
    return theme_config.get_flet_theme()

### 预览demo ###
if __name__ == "__main__":
    def main(page: ft.Page):
        # 设置页面主题
        page.theme = get_flet_theme()
        page.title = "主题预览"
        
        # 主题切换按钮
        def toggle_theme(e):
            current_theme = get_current_theme()
            if current_theme == ThemeType.LIGHT:
                switch_theme(ThemeType.DARK)
            else:
                switch_theme(ThemeType.LIGHT)
            # 更新页面主题
            page.theme = get_flet_theme()
            # 更新主题切换按钮图标
            theme_switch.icon = ft.Icons.DARK_MODE if get_current_theme() == ThemeType.LIGHT else ft.Icons.LIGHT_MODE
            # 强制刷新所有组件
            page.update()
            # 重新构建页面内容以应用新主题
            build_page_content()
        
        theme_switch = ft.IconButton(
            icon=ft.Icons.DARK_MODE if get_current_theme() == ThemeType.LIGHT else ft.Icons.LIGHT_MODE,
            on_click=toggle_theme,
            tooltip="切换主题"
        )
        
        # 颜色展示卡片
        def create_color_card(name, color_key):
            color_value = get_color(color_key)
            # print(name, color_value)
            return ft.Container(
                content=ft.Column([
                    ft.Text(name, size=get_style("font_sizes", "sm")),
                    ft.Text(get_color(color_key), size=get_style("font_sizes", "xs"), color=get_color("text_secondary")),
                ], spacing=4),
                bgcolor=get_color(color_key),
                border_radius=get_style("border_radius", "md"),
                padding=10,
                width=120,
                height=80,
            )
        
        # 页面内容容器
        content_container = ft.Column(
            spacing=16,
            scroll=ft.ScrollMode.ALWAYS,
            on_scroll_interval=0
        )
        
        # 构建页面内容
        def build_page_content():
            # 清空现有内容
            content_container.controls.clear()
            
            # 主色展示
            primary_colors = ft.Row([
                create_color_card("主色", "primary"),
                create_color_card("悬停", "primary_hover"),
                create_color_card("按下", "primary_pressed"),
                create_color_card("禁用", "primary_disabled"),
            ], wrap=True, spacing=10)
            
            # 功能色展示
            functional_colors = ft.Row([
                create_color_card("成功", "success"),
                create_color_card("错误", "error"),
                create_color_card("警告", "warning"),
                create_color_card("信息", "info"),
            ], wrap=True, spacing=10)
            
            # 背景色展示
            background_colors = ft.Row([
                create_color_card("背景", "background"),
                create_color_card("表面", "surface"),
                create_color_card("边框", "border"),
            ], wrap=True, spacing=10)
            
            # 文本色展示
            text_colors = ft.Row([
                create_color_card("文本", "text"),
                create_color_card("次要文本", "text_secondary"),
                create_color_card("禁用文本", "text_disabled"),
            ], wrap=True, spacing=10)
            
            # 按钮展示
            buttons_section = ft.Column([
                ft.Text("按钮", size=get_style("font_sizes", "xl"), weight=ft.FontWeight.BOLD),
                ft.Row([
                    ft.ElevatedButton("主按钮", bgcolor=get_color("primary"), color=get_color("background")),
                    ft.OutlinedButton("次要按钮"),
                    ft.TextButton("文本按钮"),
                    ft.ElevatedButton("禁用按钮", disabled=True),
                ], spacing=10),
            ])
            
            # 文本展示
            text_section = ft.Column([
                ft.Text("排版", size=get_style("font_sizes", "xl"), weight=ft.FontWeight.BOLD),
                ft.Text("标题 1", size=get_style("font_sizes", "4xl"), weight=ft.FontWeight.BOLD),
                ft.Text("标题 2", size=get_style("font_sizes", "3xl"), weight=ft.FontWeight.BOLD),
                ft.Text("标题 3", size=get_style("font_sizes", "2xl"), weight=ft.FontWeight.BOLD),
                ft.Text("正文大", size=get_style("font_sizes", "lg")),
                ft.Text("正文中", size=get_style("font_sizes", "base")),
                ft.Text("正文小", size=get_style("font_sizes", "sm"), color=get_color("text_secondary")),
            ])
            
            # 表单控件展示
            form_section = ft.Column([
                ft.Text("表单控件", size=get_style("font_sizes", "xl"), weight=ft.FontWeight.BOLD),
                ft.Row([
                    ft.TextField(label="输入框", hint_text="请输入内容"),
                    ft.Dropdown(
                        label="下拉选择",
                        options=[
                            ft.dropdown.Option("选项1"),
                            ft.dropdown.Option("选项2"),
                            ft.dropdown.Option("选项3"),
                        ],
                    ),
                ], spacing=10),
                ft.Row([
                    ft.Checkbox(label="复选框"),
                    ft.Radio(label="单选框", value="option1"),
                    ft.Switch(label="开关"),
                ], spacing=20),
            ])
            
            # 卡片展示
            card_section = ft.Column([
                ft.Text("卡片", size=get_style("font_sizes", "xl"), weight=ft.FontWeight.BOLD),
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("卡片标题", size=get_style("font_sizes", "lg"), weight=ft.FontWeight.BOLD),
                            ft.Text("这是一个卡片示例，展示了卡片的基本样式。"),
                            ft.Row([
                                ft.TextButton("操作1"),
                                ft.TextButton("操作2"),
                            ], alignment=ft.MainAxisAlignment.END),
                        ]),
                        padding=20,
                    ),
                ),
            ])
            
            # 添加所有内容到容器
            content_container.controls.extend([
                ft.Text("颜色系统", size=get_style("font_sizes", "2xl"), weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Text("主色", size=get_style("font_sizes", "lg"), weight=ft.FontWeight.BOLD),
                primary_colors,
                ft.Text("功能色", size=get_style("font_sizes", "lg"), weight=ft.FontWeight.BOLD),
                functional_colors,
                ft.Text("背景色", size=get_style("font_sizes", "lg"), weight=ft.FontWeight.BOLD),
                background_colors,
                ft.Text("文本色", size=get_style("font_sizes", "lg"), weight=ft.FontWeight.BOLD),
                text_colors,
                ft.Divider(height=40),
                buttons_section,
                ft.Divider(height=40),
                text_section,
                ft.Divider(height=40),
                form_section,
                ft.Divider(height=40),
                card_section,
            ])
            
            # 更新页面
            page.update()
        
        # 初始构建页面
        build_page_content()
        
        # 组装页面
        page.add(
            ft.AppBar(
                title=ft.Text("主题预览"),
                center_title=True,
                bgcolor=get_color("primary"),
                actions=[theme_switch],
            ),
            content_container,
        )
    
    ft.app(target=main, view=ft.WEB_BROWSER, port=8080)
