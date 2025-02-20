from flet import View, Container, Text
import flet
from .base_app_view import BaseAppView

class ObjectDetectionView(BaseAppView):
    def __init__(self, page: flet.Page):
        super().__init__(page)
        self.title = "对象检测"
    
    def get_route(self) -> str:
        return "/app/object_detection"
    
    def _get_toolbar(self):
        return Container(
            content=flet.Row(
                controls=[
                    flet.ElevatedButton(
                        "选择图片",
                        icon=flet.icons.UPLOAD_FILE,
                        on_click=lambda _: self._pick_files()
                    ),
                ],
                alignment=flet.MainAxisAlignment.CENTER,
            ),
            margin=10,
        )
    
    def _get_main_content(self):
        return Container(
            content=flet.Column(
                controls=[
                    # 图片预览区域和检测结果显示
                ],
                horizontal_alignment=flet.CrossAxisAlignment.CENTER,
            ),
            margin=10,
        )
    
    def _pick_files(self):
        self.page.pick_files(
            allow_multiple=False,
            allowed_extensions=["png", "jpg", "jpeg"],
            on_result=self._on_file_picked
        )
    
    def _on_file_picked(self, e):
        if e.files:
            file_path = e.files[0].path
            # TODO: 实现对象检测逻辑
            pass