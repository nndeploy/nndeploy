from flet import View, Container, Text
import flet
from .base_app_view import BaseAppView
import os

class ObjectDetectionView(BaseAppView):
    def __init__(self, page: flet.Page):
        super().__init__(page)
        self.title = "对象检测"
        # 初始化显示组件
        self.source_view = None
        self.result_view = None
        self.current_file = None
        self.video_player = None
    
    def get_route(self) -> str:
        return "/app/object_detection"
    
    def _get_toolbar(self):
        # 顶部工具栏
        toolbar = Container(
            content=flet.Row(
                controls=[
                    flet.ElevatedButton(
                        "选择图片",
                        icon=flet.icons.IMAGE,
                        on_click=lambda _: self._pick_files("image")
                    ),
                    flet.ElevatedButton(
                        "选择视频",
                        icon=flet.icons.VIDEO_FILE,
                        on_click=lambda _: self._pick_files("video")
                    ),
                    flet.Text(
                        "当前文件：未选择",
                        ref=self.current_file,
                        color=flet.colors.GREY_700,
                    ),
                ],
                alignment=flet.MainAxisAlignment.START,
                spacing=20,
            ),
            padding=10,
            bgcolor=flet.colors.BLUE_GREY_50,
        )

        # 底部内容区域
        content = Container(
            content=flet.Row(
                controls=[
                    # 左侧源文件显示
                    Container(
                        content=Container(
                            content=flet.Text("请选择文件", color=flet.colors.GREY_400),
                            ref=self.source_view,
                            bgcolor=flet.colors.GREY_100,
                            border_radius=10,
                            alignment=flet.alignment.center,
                            expand=True,
                        ),
                        expand=True,
                        padding=10,
                        border=flet.border.all(1, flet.colors.GREY_400),
                        border_radius=10,
                    ),
                    # 右侧结果显示
                    Container(
                        content=Container(
                            content=flet.Text("等待处理", color=flet.colors.GREY_400),
                            ref=self.result_view,
                            bgcolor=flet.colors.GREY_100,
                            border_radius=10,
                            alignment=flet.alignment.center,
                            expand=True,
                        ),
                        expand=True,
                        padding=10,
                        border=flet.border.all(1, flet.colors.GREY_400),
                        border_radius=10,
                    ),
                ],
                spacing=20,
                expand=True,
            ),
            expand=True,
            padding=10,
        )

        return Container(
            content=flet.Column(
                controls=[toolbar, content],
                spacing=0,
                expand=True,
            ),
            expand=True,
        )

    def _pick_files(self, file_type: str):
        extensions = ["png", "jpg", "jpeg"] if file_type == "image" else ["mp4", "avi", "mov"]
        self.page.pick_files(
            allow_multiple=False,
            allowed_extensions=extensions,
            on_result=lambda e: self._on_file_picked(e, file_type)
        )
    
    def _on_file_picked(self, e, file_type: str):
        if not e.files:
            return
            
        file_path = e.files[0].path
        file_name = e.files[0].name
        
        # 更新当前文件显示
        if self.current_file:
            self.current_file.value = f"当前文件：{file_name}"
            self.current_file.update()
        
        # 处理图片文件
        if file_type == "image":
            self._display_image(file_path)
        # 处理视频文件
        else:
            self._display_video(file_path)
    
    def _display_image(self, file_path: str):
        """显示图片文件"""
        if not os.path.exists(file_path):
            return
            
        # 创建图片显示
        image = flet.Image(
            src=file_path,
            fit=flet.ImageFit.CONTAIN,
            expand=True,
        )
        
        # 更新源文件显示
        if self.source_view:
            self.source_view.content = image
            self.source_view.update()
            
        # 清除之前的视频播放器（如果有）
        if self.video_player:
            self.video_player = None
    
    def _display_video(self, file_path: str):
        """显示视频文件"""
        if not os.path.exists(file_path):
            return
            
        # 创建视频播放器
        self.video_player = flet.Video(
            src=file_path,
            show_controls=True,
            expand=True,
        )
        
        # 创建视频控制按钮
        controls = flet.Row(
            controls=[
                flet.IconButton(
                    icon=flet.icons.PLAY_ARROW,
                    on_click=lambda _: self.video_player.play(),
                ),
                flet.IconButton(
                    icon=flet.icons.PAUSE,
                    on_click=lambda _: self.video_player.pause(),
                ),
                flet.IconButton(
                    icon=flet.icons.STOP,
                    on_click=lambda _: self.video_player.stop(),
                ),
            ],
            alignment=flet.MainAxisAlignment.CENTER,
        )
        
        # 更新源文件显示
        if self.source_view:
            self.source_view.content = flet.Column(
                controls=[self.video_player, controls],
                expand=True,
            )
            self.source_view.update()