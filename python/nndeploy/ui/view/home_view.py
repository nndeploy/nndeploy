from flet import View, Page, Container, Column, Text, Row
import flet
from .base_view import BaseView
from . import common
import settings

class HomeView(BaseView):
    def build(self) -> View:
        return View(
            route="/",
            controls=[
                self._get_app_bar(),
                self._get_home_content()
            ]
        )
 
    
    def _get_home_content(self):
        return Container(
            content=flet.ListView(
                controls=[
                    # 三部分容器
                    flet.Container(
                        content=flet.Column(
                            controls=[
                                # 上部分 - 大标题
                                flet.Container(
                                    content=flet.Text(
                                        "nndeploy",
                                        size=40,
                                        weight=flet.FontWeight.BOLD,
                                        text_align=flet.TextAlign.CENTER,
                                        color=flet.colors.WHITE,
                                    ),
                                    margin=flet.margin.only(top=20, bottom=10),
                                    alignment=flet.alignment.center,
                                ),
                                # 中部分 - 文字描述
                                flet.Container(
                                    content=flet.Text(
                                        settings.nndeploy_summary,
                                        size=16,
                                        text_align=flet.TextAlign.CENTER,
                                        color=flet.colors.WHITE,
                                    ),
                                    margin=flet.margin.only(bottom=20),
                                    alignment=flet.alignment.center,
                                ),
                                # 添加按钮
                                flet.Container(
                                    content=flet.ElevatedButton(
                                        text="Start with workflow",
                                        style=flet.ButtonStyle(
                                            color=flet.colors.WHITE,
                                            bgcolor=flet.colors.BLUE,
                                        ),
                                        on_click=lambda _: self.page.go("/workflow")  # 修改这里，使用 self.page
                                    ),
                                    margin=flet.margin.only(bottom=20),
                                    alignment=flet.alignment.center,
                                ),
                                # 下部分 - 图片
                                flet.Container(
                                    content=flet.Image(
                                        src="img/workflow.png",
                                        fit=flet.ImageFit.COVER,
                                        border_radius=10,
                                    ),
                                    height=300,
                                    alignment=flet.alignment.center,
                                ),
                            ],
                            alignment=flet.MainAxisAlignment.CENTER,
                            horizontal_alignment=flet.CrossAxisAlignment.CENTER,
                        ),
                        margin=10,
                        padding=20,
                        border_radius=10,
                        bgcolor=None,
                        image_src="img/main_bg.png",
                        image_fit=flet.ImageFit.COVER,
                        alignment=flet.alignment.center,
                    ),
                    
                    flet.Container(
                        content=flet.Divider(height=1, color="#cccccc"),
                        margin=flet.margin.only(left=self.page.width/3, right=self.page.width/3, top=20, bottom=20),
                    ),
                    
                    flet.Container(
                        content=flet.Text(
                            "Applications",
                            size=30,
                            text_align=flet.TextAlign.CENTER,
                        ),
                        alignment=flet.alignment.center,
                    ),
                    
                    # 下方色块网格
                    flet.Container(
                        content=flet.Column(
                            controls=[
                                flet.Row(
                                    controls=[
                                        common.SubRect(
                                            "https://picsum.photos/200/200", 
                                            "对象检测", 
                                            "使用 Faster R-CNN Inception ResNet V2 640x640 模型检测图像中的对象。", 
                                            width=350, 
                                            height=500, 
                                            bgcolor=flet.colors.WHITE,
                                            border_radius=10,
                                            on_click=lambda _: self.page.go("/app/object_detection")  # 添加点击事件
                                        ),
                                        common.SubRect(
                                            "https://picsum.photos/201/201", 
                                            "风格迁移", 
                                            "使用图像风格迁移模型将一个图像的风格迁移到另一个图像上。", 
                                            width=350, 
                                            height=500, 
                                            bgcolor=flet.colors.WHITE,
                                            border_radius=10
                                        ),
                                        common.SubRect(
                                            "https://picsum.photos/202/202", 
                                            "BERT", 
                                            "查看适用于 NLP 任务的 BERT，包括文本分类和问题解答。", 
                                            width=350, 
                                            height=500, 
                                            bgcolor=flet.colors.WHITE,
                                            border_radius=10
                                        ),
                                        common.SubRect(
                                            "https://picsum.photos/203/203", 
                                            "设备端食品分类器", 
                                            "在移动设备上使用该 TFLite 模型对食品照片进行分类。", 
                                            width=350, 
                                            height=500, 
                                            bgcolor=flet.colors.WHITE,
                                            border_radius=10
                                        ),
                                    ],
                                    alignment=flet.MainAxisAlignment.CENTER,
                                    spacing=20,
                                ),
                                flet.Row(
                                    controls=[
                                        common.SubRect(
                                            "https://picsum.photos/204/204", 
                                            "对象检测", 
                                            "使用 Faster R-CNN Inception ResNet V2 640x640 模型检测图像中的对象。", 
                                            width=350, 
                                            height=500, 
                                            bgcolor=flet.colors.WHITE,
                                            border_radius=10,
                                            on_click=lambda _: self.page.go("/app/object_detection")  # 添加点击事件
                                        ),
                                        common.SubRect(
                                            "https://picsum.photos/205/205", 
                                            "风格迁移", 
                                            "使用图像风格迁移模型将一个图像的风格迁移到另一个图像上。", 
                                            width=350, 
                                            height=500, 
                                            bgcolor=flet.colors.WHITE,
                                            border_radius=10
                                        ),
                                        common.SubRect(
                                            "https://picsum.photos/206/206", 
                                            "BERT", 
                                            "查看适用于 NLP 任务的 BERT，包括文本分类和问题解答。", 
                                            width=350, 
                                            height=500, 
                                            bgcolor=flet.colors.WHITE,
                                            border_radius=10
                                        ),
                                        common.SubRect(
                                            "https://picsum.photos/207/207", 
                                            "设备端食品分类器", 
                                            "在移动设备上使用该 TFLite 模型对食品照片进行分类。", 
                                            width=350, 
                                            height=500, 
                                            bgcolor=flet.colors.WHITE,
                                            border_radius=10
                                        ),
                                    ],
                                    alignment=flet.MainAxisAlignment.CENTER,
                                    spacing=20,
                                )
                            ],
                            alignment=flet.MainAxisAlignment.CENTER,
                        ),
                        alignment=flet.alignment.center
                    )
                ],
                spacing=20,
                padding=20,
            ),
            expand=True,
        )