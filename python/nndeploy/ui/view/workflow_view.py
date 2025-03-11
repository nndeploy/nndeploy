from flet import View, Container, Page, Text, ThemeMode, VerticalDivider
from flet import Icon, IconButton, TextButton, AppBar, Row, Column, Divider
import flet
from .base_view import BaseView

from .theme import ThemeManager 

from view.dag.ui_graph import UiGraph

# from view.workflow.flow_stack import FlowStack

class WorkflowView(BaseView):
    def __init__(self, page: Page):
        super().__init__(page)
        self.row_bar = self._get_workflow_row_bar()
        self.column_bar = self._get_workflow_column_bar()
        # self.divider_bar = self._get_workflow_divider_bar()
        self.workflow_content = self._get_workflow_content_v2()
        # self.page = page

    def build(self) -> View:
        return View(
            route="/workflow",
            # controls=[
            #     self._get_app_bar(),
            #     self._get_workflow_content()
            # ]
            controls=[
                self.row_bar,
                flet.Row(
                    controls=[
                        self.column_bar,
                        # self.divider_bar,
                        self.workflow_content
                    ],
                ),
            ]
        )
        
    def _get_workflow_row_bar(self):
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
                    flet.TextButton(text="文件", on_click=lambda _: self.page.launch_url(settings.github_url)),
                    flet.TextButton(text="编辑", on_click=lambda _: self.page.launch_url(settings.docs_url)),
                    flet.TextButton(text="设置", on_click=lambda _: self.page.go("/about")),
                    flet.TextButton(text="帮助", on_click=lambda _: self.page.go("/about")),
                ],
                # alignment=flet.MainAxisAlignment.CENTER,
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
    
    def _get_workflow_column_bar(self):
        """获取工作流左侧列栏"""
        from . import common
        current_theme = "dark" if self.page.theme_mode == ThemeMode.DARK else "light"
        return flet.Container(
            content=flet.Column(
                controls=[
                    flet.TextButton(
                        text="节点",
                        on_click=lambda _: self.page.launch_url(settings.github_url),
                        width=120
                    ),
                    flet.TextButton(
                        text="素材", 
                        on_click=lambda _: self.page.go("/about"),
                        width=120
                    ),
                    flet.TextButton(
                        text="模型",
                        on_click=lambda _: self.page.go("/about"),
                        width=120
                    ),
                ],
                alignment=flet.MainAxisAlignment.START,
                spacing=10,
            ),
            bgcolor=self.get_theme_color("toolbar"),
            border_radius=flet.BorderRadius(
                top_left=0,
                top_right=0, 
                bottom_left=10,
                bottom_right=10
            ),
            # padding=flet.padding.all(10),
            width=120,
        )

    def _get_workflow_content(self):
        return Container(
            content=flet.ListView(
                controls=[
                    Container(
                        content=flet.Row( 
                            controls=[
                            Text(
                            "Workflow Builder",
                            size=30,
                            weight=flet.FontWeight.BOLD,
                            text_align=flet.TextAlign.CENTER,
                            ),
                            # FlowStack(),
                        ]),
                        margin=flet.margin.only(top=20, bottom=20),
                        alignment=flet.alignment.center,
                    ),
                    # 这里添加工作流编辑器的具体内容
                ],
                spacing=20,
                padding=20,
            ),
            expand=True,
        )
        
    def _get_workflow_content_v2(self):
        return Container(
            content=flet.Container(UiGraph(self.page)),
            expand=True,
        )