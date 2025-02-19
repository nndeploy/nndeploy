import flet

from view import common
import settings

def main(page: flet.Page):
    # 设置窗口大小
    page.window_width = 1000
    page.window_height = 800
    # 设置页面滚动
    page.scroll = "auto"
    
    # 创建一个AppBar来固定顶部按钮
    page.appbar = flet.AppBar(
        leading= common.AnimatedIconTextButton("nndeploy",flet.Icons.WB_SUNNY_OUTLINED,flet.Icons.WB_SUNNY_OUTLINED),
        leading_width=120,
        center_title=True,
        title=flet.Row(
            controls=[
                flet.TextButton(text="GitHub", on_click=lambda _: page.launch_url(settings.github_url)),
                flet.TextButton(text="Docs", on_click=lambda _: page.launch_url(settings.docs_url)),
                flet.TextButton(text="About US", on_click=lambda _: None),
            ],
            alignment=flet.MainAxisAlignment.CENTER,
        ),
        actions=[
            flet.IconButton(flet.Icons.WB_SUNNY_OUTLINED),
            flet.IconButton(flet.Icons.FILTER_3),
        ],
        toolbar_opacity = 0.0,
        bgcolor="#cccccc",
        shape=flet.RoundedRectangleBorder(radius=10), # 添加圆角
    )
    # page.floating_action_button = flet.FloatingActionButton(text="WorkFlow")

    # 创建一个两列布局：第一行为图片容器，下方为色块网格
    page.add(
        flet.Column(
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
                                    on_click=lambda _: None
                                ),
                                margin=flet.margin.only(bottom=20),
                                alignment=flet.alignment.center,
                            ),
                            # 下部分 - 图片
                            flet.Container(
                                content=flet.Image(
                                    src="https://picsum.photos/3000/1000",
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
                    expand=True,
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
                    margin=flet.margin.only(left=page.width/3, right=page.width/3, top=20, bottom=20),
                ),

                flet.Container(
                    content=flet.Text(
                        "Applications",
                        size=30,
                        text_align=flet.TextAlign.CENTER,
                        color="#1d1c1c",
                    ),
                    # margin=flet.margin.only(),
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
                                        border_radius=10
                                    ),
                                    common.SubRect(
                                        "https://picsum.photos/200/200", 
                                        "风格迁移", 
                                        "使用图像风格迁移模型将一个图像的风格迁移到另一个图像上。", 
                                        width=350, 
                                        height=500, 
                                        bgcolor=flet.colors.WHITE,
                                        border_radius=10
                                    ),
                                    common.SubRect(
                                        "https://picsum.photos/200/200", 
                                        "BERT", 
                                        "查看适用于 NLP 任务的 BERT，包括文本分类和问题解答。", 
                                        width=350, 
                                        height=500, 
                                        bgcolor=flet.colors.WHITE,
                                        border_radius=10
                                    ),
                                    common.SubRect(
                                        "https://picsum.photos/200/200", 
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
                                    flet.Container(width=300, height=400, bgcolor=flet.colors.YELLOW),
                                    flet.Container(width=300, height=400, bgcolor=flet.colors.PURPLE),
                                    flet.Container(width=300, height=400, bgcolor=flet.colors.ORANGE),
                                ],
                                alignment=flet.MainAxisAlignment.CENTER,
                            ),
                            flet.Row(
                                controls=[
                                    flet.Container(width=300, height=400, bgcolor=flet.colors.PINK),
                                    flet.Container(width=300, height=400, bgcolor=flet.colors.BROWN),
                                    flet.Container(width=300, height=400, bgcolor=flet.colors.CYAN),
                                ],
                                alignment=flet.MainAxisAlignment.CENTER,
                            ),
                            flet.Row(
                                controls=[
                                    flet.Container(width=300, height=400, bgcolor=flet.colors.TEAL),
                                    flet.Container(width=300, height=400, bgcolor=flet.colors.INDIGO),
                                    flet.Container(width=300, height=400, bgcolor=flet.colors.LIME),
                                ],
                                alignment=flet.MainAxisAlignment.CENTER,
                            ),
                            flet.Row(
                                controls=[
                                    flet.Container(width=300, height=400, bgcolor=flet.colors.AMBER),
                                    flet.Container(width=300, height=400, bgcolor=flet.colors.DEEP_PURPLE),
                                    flet.Container(width=300, height=400, bgcolor=flet.colors.LIGHT_BLUE),
                                ],
                                alignment=flet.MainAxisAlignment.CENTER,
                            ),
                            flet.Row(
                                controls=[
                                    flet.Container(width=300, height=400, bgcolor=flet.colors.DEEP_ORANGE),
                                    flet.Container(width=300, height=400, bgcolor=flet.colors.LIGHT_GREEN),
                                    flet.Container(width=300, height=400, bgcolor=flet.colors.BLUE_GREY),
                                ],
                                alignment=flet.MainAxisAlignment.CENTER,
                            ),
                        ],
                        alignment=flet.MainAxisAlignment.CENTER,
                    ),
                    alignment=flet.alignment.center
                ),
            ],
            expand=True,  # 让Column占满整个页面
            spacing=20,
        )
    )

flet.app(target= main, assets_dir=settings.res)
# , view=flet.WEB_BROWSER