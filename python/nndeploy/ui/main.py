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
import flet.version
from typing import Optional, Any, Dict
import sys
import os
import logging
from pathlib import Path
import argparse

from config.language import language_config, Language, get_text
from config.theme import theme_config, ThemeType
from config.settings import settings
from config.shortcuts import shortcut_config

# 导入视图组件
# from view.menu.file_menu import FileMenu
# from view.menu.edit_menu import EditMenu
# from view.menu.settings_menu import SettingsMenu
# from view.menu.help_menu import HelpMenu

# from view.sidebar.node_panel import NodePanel
# from view.sidebar.model_panel import ModelPanel
# from view.sidebar.material_panel import MaterialPanel


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="nndeploy")
    parser.add_argument("--web_view", action="store_true", help="show nndeploy web view")
    parser.add_argument("--port", type=int, default=8080, help="web view port")
    return parser.parse_args()


class App:
    """应用程序主类"""
    
    def __init__(self):
        pass
    
    def init_page(self, page: ft.Page):
        """初始化应用程序"""
        self.page = page
        self.page.title = "nndeploy"
        self.page.adaptive = True
        
        # self.page.fonts = {
        #     "Roboto Mono": "RobotoMono-VariableFont_wght.ttf",
        #     "RobotoSlab": "RobotoSlab[wght].ttf",
        # }
        
        # 初始化配置
        self._init_config()
        
        # appbar
        # log, menu(File, Edit, Setting, Help), github, 知乎, bilibili, Wechat
        # self.memu = self.create_menu()
        # 创建菜单栏        
        self.page.appbar = ft.AppBar(
            leading=ft.Container(
                self.menu_bar,
            ),
            leading_width = 200,
            title=ft.Text("nndeploy"),
            center_title=True,
            bgcolor=ft.Colors.INVERSE_PRIMARY,
            actions=[
                ft.TextButton(text="GitHub", on_click=lambda e: self.page.launch_url("https://github.com/nndeploy/nndeploy")),
                ft.TextButton(text="Zhihu", on_click=lambda e: self.page.launch_url("https://www.zhihu.com/column/c_1690464325314240512")),
                ft.TextButton(text="Bilibili", on_click=lambda e: self.page.launch_url("https://space.bilibili.com/435543077?spm_id_from=333.1007.0.0"))
            ],
        )
        
        # 左边 sidebar
        # Node, Model, Materail, Workflow
        # 支持收起来，支持拖拉
        # self.sidebar = self.create_sidebar()
        # self.sidebar = ft.Container(
        #     content=ft.Tabs(
        #         selected_index=0,
        #         tabs=[
        #             ft.Tab(text="Node"),
        #             ft.Tab(text="Model"),
        #             ft.Tab(text="Material"),
        #             ft.Tab(text="Workflow"),
        #         ],
        #     ),
        #     expand=True,
        # )
        self.sidebar = ft.Container(
            content=ft.Text("Node"),
            expand=True,
        )
        self.page.add(self.sidebar)
        
        # # grid
        # # 画布，缩放，小地图
        # # self.grid = self.create_grid()
        # self.grid = ft.Container(
        #     content=ft.Stack(
        #         [
        #             ft.Canvas(
        #                 on_selection_change=self._on_selection_change,
        #                 on_node_double_click=self._on_node_double_click
        #             )
        #         ]
        #     )
        # )
        
        
        # self.column = ft.Column(
        #     [
        #         self.sidebar,
        #         self.grid
        #     ],
        # )
             
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
                               
    def _create_sidebar(self) -> ft.Control:
        """创建左侧边栏"""
        # return ft.Container(
        #     content=ft.Tabs(
        #         selected_index=0,
        #         tabs=[
        #             ft.Tab(
        #                 text=get_text("sidebar.nodes"),
        #                 content=self.node_panel
        #             ),
        #             ft.Tab(
        #                 text=get_text("sidebar.models"),
        #                 content=self.model_panel
        #             ),
        #             ft.Tab(
        #                 text=get_text("sidebar.materials"),
        #                 content=self.material_panel
        #             ),
        #         ],
        #     ),
        #     width=settings.get("ui", "sidebar_width"),
        #     bgcolor=theme_config.get_color("surface"),
        # )
        pass
        
    def _create_main_area(self) -> ft.Control:
        """创建主工作区"""
        # return ft.Container(
        #     content=ft.Column(
        #         [
        #             # 工具栏
        #             self._create_toolbar(),
                    
        #             # 画布区域
        #             ft.Stack(
        #                 [
        #                     # 画布组件
        #                     self.canvas,
                            
        #                     # 缩放控件
        #                     ZoomControl(
        #                         on_zoom_change=self.canvas.set_zoom
        #                     ),
                            
        #                     # 小地图
        #                     Minimap(
        #                         canvas=self.canvas,
        #                         width=200,
        #                         height=150,
        #                     ),
        #                 ],
        #                 expand=True,
        #             ),
                    
        #             # 状态栏
        #             self._create_statusbar(),
        #         ],
        #         spacing=0,
        #     ),
        #     expand=True,
        # )
        pass        
        
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
            
    def run(self, web_view: bool = True, port: int = 8080):
        """启动应用程序"""
        if web_view:
            ft.app(target=self.init_page, view=ft.WEB_BROWSER, port=port)
        else:
            ft.app(target=self.init_page)


def show(graph, web_view: bool = True, port: int = 8080):
    """显示应用程序"""
    app = App()
    # app.add_graph(graph)
    app.run(web_view=web_view, port=port)


def main():
    """程序入口函数"""
    try:
        # 设置工作目录
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # 创建配置目录
        Path("config").mkdir(exist_ok=True)
        
        # 解析命令行参数
        args = parse_args()
        web_view = args.web_view
        port = args.port
        
        # 创建并运行应用
        app = App()
        print(web_view, port)
        app.run(web_view=web_view, port=port)
        
    except Exception as e:
        logger.error(f"应用程序启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
    
