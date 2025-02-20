from flet import View, Container, Markdown, Column, ScrollMode
import flet
from .base_view import BaseView
import os

class AboutUsView(BaseView):
    def build(self) -> View:
        return View(
            route="/about",
            controls=[
                self._get_app_bar(),
                self._get_content()
            ]
        )
    
    def _get_content(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        md_path = os.path.join(root_dir, "README.md")
        
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
                # 替换相对图片路径为绝对路径
                md_content = md_content.replace(
                    "](docs/images/",
                    f"]({root_dir}/docs/images/"
                ).replace(
                    "](images/",
                    f"]({root_dir}/images/"
                )
                
                # 优化表格显示
                md_content = md_content.replace("|--", "| --")
                md_content = md_content.replace("--|", "-- |")
        except FileNotFoundError:
            md_content = "# About Us\n未找到 README.md 文件"
        
        return Container(
            content=Column(
                controls=[
                    Container(
                        content=Markdown(
                            md_content,
                            selectable=True,
                            extension_set="github",
                            code_style=flet.TextStyle(
                                font_family="monospace",
                                size=14,
                            ),
                            code_theme="github-dark",
                        ),
                        expand=True,
                    )
                ],
                scroll=ScrollMode.ALWAYS,
                expand=True,
            ),
            padding=20,
            expand=True,
        )