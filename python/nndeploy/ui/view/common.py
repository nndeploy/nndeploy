import flet

class AnimatedIconTextButton(flet.TextButton):
    def __init__(
        self,
        text: str,
        icon: str,
        animated_icon: str,
        on_click=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.text = text
        self._static_icon = icon
        self._animated_icon = animated_icon
        self.on_click = on_click
        self.expand = False
        
        # 创建图标和文本布局
        self.content = flet.Row(
            controls=[
                flet.Icon(self._static_icon),
                flet.Text(text)
            ],
            spacing=5,
            alignment=flet.MainAxisAlignment.CENTER
        )
        
        # 添加鼠标悬停事件处理
        self.on_hover = self._handle_hover
        
    def _handle_hover(self, e):
        if e.data == "true":  # 鼠标悬停
            self.content.controls[0].name = self._animated_icon
        else:  # 鼠标离开
            self.content.controls[0].name = self._static_icon
        self.content.controls[0].update()


class SubRect(flet.Container):
    def __init__(self,
                 icon: str,
                 name: str,
                 tip: str,
                 on_click=None,  # 添加点击事件参数
                 **kwargs):
        super().__init__(on_hover=self.hover_handler, **kwargs)
        self.case_icon = icon
        self.case_name = name
        self.case_tip = tip
        self.on_click = on_click  # 保存点击事件
        
        self.init_rect()

    def hover_handler(e):
        e.control.shadow = flet.BoxShadow(
        blur_radius=15,
        spread_radius=4,
        color=flet.colors.GREY,
        offset=flet.Offset(5, 5),
        ) if e.data == "true" else None
        e.control.update()

    def init_rect(self):
        self.border = flet.border.all(1, "#cccccc")
        self.content = flet.Column(
            controls=[
                flet.Container(
                    content=flet.Image(
                        src=self.case_icon,
                        width=self.width,
                        height=self.height*0.6,
                        fit=flet.ImageFit.COVER,
                        border_radius=flet.border_radius.only(top_left = 9, top_right= 9),
                    ),
                    height=self.height*0.6,
                    alignment=flet.alignment.top_center,
                ),
                flet.Container(
                    content=flet.Text(
                        self.case_name,
                        size=24,
                        text_align=flet.TextAlign.LEFT,
                        color="#3b3c3d",
                    ),
                    margin=flet.margin.only(left=20, top=10, right=20, bottom=10),
                    alignment=flet.alignment.top_left,
                ),
                flet.Container(
                    content=flet.Text(
                        self.case_tip,
                        size=16,
                        text_align=flet.TextAlign.LEFT,
                        color="#3b3c3d",
                    ),
                    margin=flet.margin.only(left=20, top=10, right=20),
                    alignment=flet.alignment.top_left,
                ),
                flet.Container(
                    content=flet.TextButton(
                        content=flet.Row(
                            [flet.Text("Go use"), flet.Icon(flet.icons.SUBDIRECTORY_ARROW_LEFT)],
                            alignment=flet.MainAxisAlignment.SPACE_BETWEEN,
                        ),
                        width=100,
                        on_click=self.on_click,  # 添加点击事件处理
                    ),
                    margin=flet.margin.only(left=20, bottom=20),
                    alignment=flet.alignment.bottom_left,
                ),
            ],
        )
