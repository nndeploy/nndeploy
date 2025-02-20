from flet import View, Container, Text
import flet
from .base_view import BaseView

class WorkflowView(BaseView):
    def build(self) -> View:
        return View(
            route="/workflow",
            controls=[
                self._get_app_bar(),
                self._get_workflow_content()
            ]
        )
    
    def _get_workflow_content(self):
        return Container(
            content=flet.ListView(
                controls=[
                    Container(
                        content=Text(
                            "Workflow Builder",
                            size=30,
                            weight=flet.FontWeight.BOLD,
                            text_align=flet.TextAlign.CENTER,
                        ),
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