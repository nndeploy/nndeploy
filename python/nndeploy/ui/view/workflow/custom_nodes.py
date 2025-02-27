from node import WorkflowNode
import flet as ft

class ImageNode(WorkflowNode):
    """图像处理节点"""
    def __init__(self, title: str, position: tuple = (0, 0), page: ft.Page = None, canvas_manager=None):
        # 先调用父类初始化
        super().__init__(title, position=position, page=page, canvas_manager=canvas_manager)
    
    def init_properties(self):
        """初始化节点的属性"""
        # 设置输入输出端口名称
        self.inputs = ["image", "text"]
        self.outputs = ["processed"]
        # 设置属性
        self.properties = {
            "brightness": 0,
            "contrast": 1,
        }
        # 确保在父类创建端口之前设置好输入输出
        super().init_properties()
    
    def _create_content_area(self) -> ft.Container:
        """自定义内容区域"""
        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Slider(
                        label="亮度",
                        min=-1,
                        max=1,
                        value=self.properties["brightness"],
                        on_change=self._on_brightness_change,
                    ),
                    ft.Slider(
                        label="对比度",
                        min=0,
                        max=2,
                        value=self.properties["contrast"],
                        on_change=self._on_contrast_change,
                    ),
                ],
                spacing=10,
            ),
            padding=10,
        )
    
    def _on_brightness_change(self, e):
        self.properties["brightness"] = e.value
        self.update()
    
    def _on_contrast_change(self, e):
        self.properties["contrast"] = e.value
        self.update()

class TextNode(WorkflowNode):
    """文本处理节点"""
    def __init__(self, title: str, position: tuple = (0, 0), page: ft.Page = None, canvas_manager=None):
        super().__init__(title, position=position, page=page, canvas_manager=canvas_manager)
    
    def init_properties(self):
        """初始化节点的属性"""
        # 设置输入输出端口名称
        self.inputs = ["text"]
        self.outputs = ["processed"]
        # 设置属性
        self.properties = {
            "text": "",
        }
        # 确保在父类创建端口之前设置好输入输出
        super().init_properties()
    
    def _create_content_area(self) -> ft.Container:
        return ft.Container(
            content=ft.TextField(
                value=self.properties["text"],
                multiline=True,
                min_lines=3,
                on_change=self._on_text_change,
            ),
            padding=10,
        )
    
    def _on_text_change(self, e):
        self.properties["text"] = e.value
        self.update()

class VideoNode(WorkflowNode):
    """视频播放节点"""
    def __init__(self, title: str, position: tuple = (0, 0), page: ft.Page = None, canvas_manager=None):
        self._is_playing = False
        self.video_media = None  # 添加 video_media 属性
        super().__init__(title, position=position, page=page, canvas_manager=canvas_manager)
    
    def init_properties(self):
        self.inputs = ["video"]
        self.outputs = ["frame"]
        self.properties = {
            "url": "",
            "volume": 0.5,
            "playback_rate": 1.0,
            "media": None,  # 添加 media 属性
        }
        super().init_properties()
    
    def _create_content_area(self) -> ft.Container:
        """创建视频播放器界面"""
        # 创建视频播放器
        self.video_player = ft.Video(
            width=200,
            height=120,
            volume=self.properties["volume"],
            playback_rate=self.properties["playback_rate"],
            autoplay=False,
            show_controls=True,
        )
        
        self.url_input = ft.TextField(
            value=self.properties["url"],
            label="视频地址",
            on_change=self._on_url_change,
        )
        
        self.volume_slider = ft.Slider(
            label="音量",
            min=0,
            max=1,
            value=self.properties["volume"],
            on_change=self._on_volume_change,
        )
        
        self.speed_dropdown = ft.Dropdown(
            label="播放速度",
            value=str(self.properties["playback_rate"]),
            options=[
                ft.dropdown.Option("0.5", "0.5x"),
                ft.dropdown.Option("1.0", "1.0x"),
                ft.dropdown.Option("1.5", "1.5x"),
                ft.dropdown.Option("2.0", "2.0x"),
            ],
            on_change=self._on_speed_change,
        )
        
        self.play_button = ft.IconButton(
            icon=ft.icons.PLAY_ARROW if not self._is_playing else ft.icons.PAUSE,
            on_click=self._toggle_play,
        )
        
        return ft.Container(
            content=ft.Column(
                controls=[
                    self.video_player,
                    self.url_input,
                    ft.Row(
                        controls=[
                            self.play_button,
                            self.volume_slider,
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    self.speed_dropdown,
                ],
                spacing=10,
            ),
            padding=10,
        )
    
    def _on_url_change(self, e):
        self.properties["url"] = e.value
        if e.value:
            # 创建新的 VideoMedia
            self.video_media = ft.VideoMedia(
                source_url=e.value,
                type="video/mp4",  # 假设是 MP4 格式
            )
            # 设置视频源
            self.video_player.source = self.video_media
            self.properties["media"] = self.video_media
        self.update()
    
    def _on_volume_change(self, e):
        self.properties["volume"] = e.value
        if self.video_player:
            self.video_player.volume = e.value
        self.update()
    
    def _on_speed_change(self, e):
        self.properties["playback_rate"] = float(e.value)
        if self.video_player:
            self.video_player.playback_rate = float(e.value)
        self.update()
    
    def _toggle_play(self, e):
        self._is_playing = not self._is_playing
        if self.video_player and self.video_media:
            if self._is_playing:
                self.video_player.play()
            else:
                self.video_player.pause()
        self.play_button.icon = ft.icons.PAUSE if self._is_playing else ft.icons.PLAY_ARROW
        self.play_button.update()
        # 这里可以添加实际的视频播放/暂停逻辑