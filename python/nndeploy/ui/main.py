"""
主应用程序入口点

该模块负责:
1. 初始化应用程序
2. 配置全局设置
3. 启动主界面
4. 管理应用生命周期

主要组件:
- App类: 应用程序主类,管理整个应用的生命周期
- main(): 程序入口函数
"""

import flet as ft
from typing import Optional, Any, Dict
import sys
import os
import logging
from pathlib import Path

from config.language import language_config, Language, get_text
from config.theme import theme_config, ThemeType
from config.settings import settings
from config.shortcuts import shortcut_config

# 导入视图组件
from view.menu.file_menu import FileMenu
from view.menu.edit_menu import EditMenu
from view.menu.settings_menu import SettingsMenu
from view.menu.help_menu import HelpMenu

from view.sidebar.node_panel import NodePanel
from view.sidebar.model_panel import ModelPanel
from view.sidebar.material_panel import MaterialPanel

from view.canvas.canvas import Canvas
from view.canvas.zoom import ZoomControl
from view.canvas.minimap import Minimap

from view.node.config_panel import ConfigPanel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class App:
    """应用程序主类"""
    
    def __init__(self):
        """初始化应用程序"""
        self.page: Optional[ft.Page] = None
        self.page.adaptive = True
        
        # 组件引用
        self.canvas: Optional[Canvas] = None
        self.node_panel: Optional[NodePanel] = None
        self.model_panel: Optional[ModelPanel] = None
        self.material_panel: Optional[MaterialPanel] = None
        self.config_panel: Optional[ConfigPanel] = None
        
        # 初始化配置
        self._init_config()
        
    def _init_config(self):
        """初始化配置"""
        try:
            # 加载语言配置
            language_config.add_observer(self._on_language_change)
            
            # 加载主题配置
            theme_config.add_observer(self._on_theme_change)
            
            # 加载设置
            settings.add_observer(self._on_setting_change)
            
        except Exception as e:
            logger.error(f"配置初始化失败: {e}")
            sys.exit(1)
            
    def _init_page(self, page: ft.Page):
        """初始化页面"""
        try:
            self.page = page
            
            # 设置页面属性
            self.page.title = get_text("app.title")
            self.page.theme = theme_config.get_flet_theme()
            self.page.window_width = settings.get("ui", "window_width", 1280)
            self.page.window_height = settings.get("ui", "window_height", 800)
            self.page.window_min_width = 800
            self.page.window_min_height = 600
            self.page.window_center()
            
            # 初始化组件
            self._init_components()
            
            # 初始化布局
            self._init_layout()
            
            # 注册快捷键
            self._register_shortcuts()
            
            # 注册窗口事件
            self._register_window_events()
            
            logger.info("应用程序初始化完成")
            
        except Exception as e:
            logger.error(f"页面初始化失败: {e}")
            sys.exit(1)
            
    def _init_components(self):
        """初始化组件"""
        # 创建画布
        self.canvas = Canvas(
            on_selection_change=self._on_selection_change,
            on_node_double_click=self._on_node_double_click
        )
        
        # 创建面板
        self.node_panel = NodePanel(
            on_node_drag=self._on_node_drag
        )
        
        self.model_panel = ModelPanel(
            on_model_selected=self._on_model_selected
        )
        
        self.material_panel = MaterialPanel(
            on_material_selected=self._on_material_selected
        )
        
        self.config_panel = ConfigPanel(
            on_config_change=self._on_config_change
        )
        
    def _init_layout(self):
        """初始化界面布局"""
        try:
            # 创建菜单栏
            menu_bar = ft.Row(
                [
                    FileMenu(on_action=self._on_file_action),
                    EditMenu(on_action=self._on_edit_action),
                    SettingsMenu(on_action=self._on_settings_action),
                    HelpMenu(on_action=self._on_help_action),
                ],
                spacing=0,
            )
            
            # 创建主布局
            self.page.add(
                ft.Column(
                    [
                        # 菜单栏
                        menu_bar,
                        
                        # 主区域
                        ft.Row(
                            [
                                # 左侧边栏
                                self._create_sidebar(),
                                
                                # 垂直分隔线
                                ft.VerticalDivider(),
                                
                                # 主工作区
                                self._create_main_area(),
                                
                                # 垂直分隔线
                                ft.VerticalDivider(),
                                
                                # 右侧属性面板
                                self._create_property_panel(),
                            ],
                            expand=True,
                        ),
                    ],
                    expand=True,
                )
            )
            
        except Exception as e:
            logger.error(f"布局初始化失败: {e}")
            raise
            
    def _create_sidebar(self) -> ft.Control:
        """创建左侧边栏"""
        return ft.Container(
            content=ft.Tabs(
                selected_index=0,
                tabs=[
                    ft.Tab(
                        text=get_text("sidebar.nodes"),
                        content=self.node_panel
                    ),
                    ft.Tab(
                        text=get_text("sidebar.models"),
                        content=self.model_panel
                    ),
                    ft.Tab(
                        text=get_text("sidebar.materials"),
                        content=self.material_panel
                    ),
                ],
            ),
            width=settings.get("ui", "sidebar_width"),
            bgcolor=theme_config.get_color("surface"),
        )
        
    def _create_main_area(self) -> ft.Control:
        """创建主工作区"""
        return ft.Container(
            content=ft.Column(
                [
                    # 工具栏
                    self._create_toolbar(),
                    
                    # 画布区域
                    ft.Stack(
                        [
                            # 画布组件
                            self.canvas,
                            
                            # 缩放控件
                            ZoomControl(
                                on_zoom_change=self.canvas.set_zoom
                            ),
                            
                            # 小地图
                            Minimap(
                                canvas=self.canvas,
                                width=200,
                                height=150,
                            ),
                        ],
                        expand=True,
                    ),
                    
                    # 状态栏
                    self._create_statusbar(),
                ],
                spacing=0,
            ),
            expand=True,
        )
        
    def _create_toolbar(self) -> ft.Control:
        """创建工具栏"""
        return ft.Container(
            content=ft.Row(
                [
                    ft.IconButton(
                        icon=ft.icons.PLAY_ARROW,
                        tooltip=get_text("toolbar.run"),
                        on_click=self._on_run_click,
                    ),
                    ft.IconButton(
                        icon=ft.icons.STOP,
                        tooltip=get_text("toolbar.stop"),
                        on_click=self._on_stop_click,
                    ),
                    ft.VerticalDivider(),
                    ft.IconButton(
                        icon=ft.icons.UNDO,
                        tooltip=get_text("toolbar.undo"),
                        on_click=self._on_undo_click,
                    ),
                    ft.IconButton(
                        icon=ft.icons.REDO,
                        tooltip=get_text("toolbar.redo"),
                        on_click=self._on_redo_click,
                    ),
                ],
                spacing=10,
            ),
            padding=10,
            bgcolor=theme_config.get_color("surface"),
        )
        
    def _create_statusbar(self) -> ft.Control:
        """创建状态栏"""
        return ft.Container(
            content=ft.Row(
                [
                    ft.Text("Ready"),  # TODO: 状态文本
                    ft.ProgressBar(width=100, visible=False),  # 进度条
                ],
                spacing=10,
            ),
            padding=5,
            bgcolor=theme_config.get_color("surface"),
        )
        
    def _create_property_panel(self) -> ft.Control:
        """创建右侧属性面板"""
        return ft.Container(
            content=ft.Column(
                [
                    # TODO: 实现属性编辑面板
                ],
                spacing=10,
            ),
            width=settings.get("ui", "panel_width"),
            bgcolor=theme_config.get_color("surface"),
        )
        
    def _register_shortcuts(self):
        """注册快捷键"""
        def on_keyboard_event(e: ft.KeyboardEvent):
            modifiers = []
            if e.ctrl: modifiers.append("ctrl")
            if e.shift: modifiers.append("shift")
            if e.alt: modifiers.append("alt")
            if e.meta: modifiers.append("meta")
            
            shortcut_config.handle_keypress(e.key, modifiers)
            
        self.page.on_keyboard_event = on_keyboard_event
        
    def _register_window_events(self):
        """注册窗口事件"""
        def on_window_event(e):
            if e.data == "close":
                # TODO: 处理未保存的更改
                self.page.window_destroy()
                
        self.page.on_window_event = on_window_event
        
    def _on_selection_change(self, selected_nodes):
        """画布选择变化处理"""
        if len(selected_nodes) == 1:
            # 单选节点时更新属性面板
            self.config_panel.set_node(selected_nodes[0])
        else:
            # 多选或无选择时清空属性面板
            self.config_panel.clear()
            
    def _on_node_double_click(self, node):
        """节点双击处理"""
        # TODO: 实现节点双击行为
        pass
        
    def _on_node_drag(self, node_type: str, position: Dict[str, float]):
        """节点拖放处理"""
        self.canvas.create_node(node_type, position)
        
    def _on_model_selected(self, model_id: str):
        """模型选择处理"""
        # TODO: 实现模型选择行为
        pass
        
    def _on_material_selected(self, material_id: str):
        """素材选择处理"""
        # TODO: 实现素材选择行为
        pass
        
    def _on_config_change(self, node_id: str, config: Dict):
        """节点配置变化处理"""
        self.canvas.update_node_config(node_id, config)
        
    def _on_file_action(self, action: str):
        """文件菜单动作处理"""
        # TODO: 实现文件菜单动作
        pass
        
    def _on_edit_action(self, action: str):
        """编辑菜单动作处理"""
        # TODO: 实现编辑菜单动作
        pass
        
    def _on_settings_action(self, action: str):
        """设置菜单动作处理"""
        # TODO: 实现设置菜单动作
        pass
        
    def _on_help_action(self, action: str):
        """帮助菜单动作处理"""
        # TODO: 实现帮助菜单动作
        pass
        
    def _on_run_click(self, e):
        """运行按钮点击处理"""
        # TODO: 实现工作流运行
        pass
        
    def _on_stop_click(self, e):
        """停止按钮点击处理"""
        # TODO: 实现工作流停止
        pass
        
    def _on_undo_click(self, e):
        """撤销按钮点击处理"""
        self.canvas.undo()
        
    def _on_redo_click(self, e):
        """重做按钮点击处理"""
        self.canvas.redo()
        
    def _on_language_change(self, language: Language):
        """语言变化处理"""
        if self.page:
            self.page.title = get_text("app.title")
            self.page.update()
            
    def _on_theme_change(self, theme: ThemeType):
        """主题变化处理"""
        if self.page:
            self.page.theme = theme_config.get_flet_theme()
            self.page.update()
            
    def _on_setting_change(self, section: str, key: str, value: Any):
        """设置变化处理"""
        if self.page:
            if section == "ui":
                if key == "sidebar_width":
                    # TODO: 更新侧边栏宽度
                    pass
                elif key == "panel_width":
                    # TODO: 更新属性面板宽度
                    pass
            self.page.update()
            
    def run(self):
        """启动应用程序"""
        ft.app(target=self._init_page)

def main():
    """程序入口函数"""
    try:
        # 设置工作目录
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # 创建配置目录
        Path("config").mkdir(exist_ok=True)
        
        # 创建并运行应用
        app = App()
        app.run()
        
    except Exception as e:
        logger.error(f"应用程序启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 