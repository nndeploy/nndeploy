from flet import View, Container, Text
import flet
from .base_app_view import BaseAppView
import os
import base64

try:
    import nndeploy._nndeploy_internal as _C
    import nndeploy.base
    import nndeploy.device
    import nndeploy.dag
    import nndeploy.detect
except ImportError:
    nndeploy = None  # 如果导入失败，设置为 None 或其他默认值
    print("optional_module 未安装，部分功能不可用")

class ObjectDetectionView(BaseAppView):
    def __init__(self, page: flet.Page):
        super().__init__(page)
        self.title = "对象检测"
        # 初始化显示组件
        self.source_view = None
        self.result_view = None
        self.current_file_text = flet.Text(  # 修改变量名并直接创建Text实例
            "当前文件：未选择",
            color=flet.colors.GREY_700,
        )
        self.video_player = None
        self.current_image = None

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
                    self.current_file_text,  # 直接使用Text实例
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
                            bgcolor=flet.colors.GREY_100,
                            border_radius=10,
                            alignment=flet.alignment.center,
                            expand=True,
                            height=600,
                        ),
                        expand=True,
                        padding=10,
                        border=flet.border.all(1, flet.colors.GREY_400),
                        border_radius=10,
                    ),
                    # 中间控制区
                    Container(
                        content=flet.Column(
                            controls=[
                                flet.IconButton(
                                    icon=flet.icons.PLAY_ARROW_ROUNDED,  # 修改图标
                                    tooltip="运行检测",
                                    on_click=lambda _: self._run_detection(),  # 修复回调函数
                                    icon_color=self.get_theme_color("text"),
                                    icon_size=32,  # 增加图标大小
                                ),
                            ],
                            alignment=flet.MainAxisAlignment.CENTER,
                            spacing=10,
                        ),
                        padding=10,  # 增加内边距
                        bgcolor=flet.colors.BLUE_GREY_50,  # 添加背景色
                        border_radius=10,  # 添加圆角
                    ),
                    # 右侧结果显示
                    Container(
                        content=Container(
                            content=flet.Text("等待处理", color=flet.colors.GREY_400),
                            bgcolor=flet.colors.GREY_100,
                            border_radius=10,
                            alignment=flet.alignment.center,
                            expand=True,
                            height=600,
                        ),
                        expand=True,
                        padding=10,
                        border=flet.border.all(1, flet.colors.GREY_400),
                        border_radius=10,
                    ),
                ],
                spacing=20,
                expand=True,
                height=620,
            ),
            expand=True,
            padding=10,
        )

        # 保存对左侧显示区域的引用
        self.source_view = content.content.controls[0].content
        self.result_view = content.content.controls[2].content

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
        
        # 直接使用本地文件路径
        def handle_result(e):
            if e.files:
                print(f"Selected file path: {e.files[0].path}")
                print(f"Selected file name: {e.files[0].name}")
                self._on_file_picked(e, file_type)
        
        self.page.file_picker.on_result = handle_result
        self.page.file_picker.pick_files(
            allow_multiple=False,
            allowed_extensions=extensions
        )
    
    def _on_file_picked(self, e, file_type: str):
        print(f"\n=== 文件选择事件开始 ===")
        if not e.files:
            print("没有选择文件")
            return
            
        file_path = e.files[0].path
        file_name = e.files[0].name
        print(f"文件类型: {file_type}")
        print(f"文件路径: {file_path}")
        print(f"文件名称: {file_name}")
        
        # 更新当前文件显示
        self.current_file_text.value = f"当前文件：{file_path}"
        self.current_file_text.update()
        
        # 处理图片文件
        if file_type == "image":
            print("开始处理图片文件")
            self._display_image(file_path)
        # 处理视频文件
        else:
            print("开始处理视频文件")
            self._display_video(file_path)
        print("=== 文件选择事件结束 ===\n")
    
    def _display_image(self, file_path: str):
        """显示图片文件"""
        print(f"\n=== 开始显示图片 ===")
        print(f"检查文件路径: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"错误：文件不存在: {file_path}")
            return
        else:
            print(f"文件存在，大小: {os.path.getsize(file_path)} 字节")
            
        try:
            print("创建图片控件...")
            # 将图片转换为 base64
            with open(file_path, "rb") as f:
                image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode()
            
            # 创建图片显示
            image = flet.Image(
                src_base64=image_base64,
                fit=flet.ImageFit.CONTAIN,
                expand=True,
                height=600,
            )
            print(f"图片控件创建成功: {image}")
            
            # 更新源文件显示
            if self.source_view:
                print(f"更新显示区域...")
                print(f"source_view 类型: {type(self.source_view)}")
                self.source_view.content = image
                self.source_view.update()
                print("显示区域更新完成")
            else:
                print("错误：source_view 为空")
                
            # 清除之前的视频播放器（如果有）
            if self.video_player:
                print("清除旧的视频播放器")
                self.video_player = None
                
        except Exception as e:
            print(f"错误：显示图片时出错:")
            print(f"错误类型: {type(e)}")
            print(f"错误信息: {str(e)}")
            
        print("=== 图片显示处理结束 ===\n")
        self.current_image = file_path  # 保存当前图片路径
    
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

    def _run_detection(self):
        """运行目标检测"""
        # def run_detection_click(_):  # 添加点击处理函数
        if not self.current_image:
            print("请先选择图片")
            return

        if not nndeploy:
            print("nndeploy 未安装")
            return
            
        try:
            # 创建detect_graph
            outputs = _C.dag.Edge("outputs")
            detect_graph = nndeploy.detect.DetectGraph("detect_graph", outputs)
            detect_graph.init()
            
            # 设置输入路径
            detect_graph.set_input_path(self.current_image)
            # 获取输入文件名和扩展名
            input_filename = os.path.basename(self.current_image)
            filename, ext = os.path.splitext(input_filename)
            # 设置输出路径为当前运行目录下的 filename_detect.ext
            output_path = os.path.join(os.getcwd(), f"{filename}_detect{ext}")
            # 设置输出路径
            detect_graph.set_output_path(output_path)
            detect_graph.run() 
            detect_graph.deinit()

            # 示例：假设检测后的图片保存在同一位置，文件名加上 _detected 后缀
            file_name, file_ext = os.path.splitext(self.current_image)
            
            # 显示检测结果
            with open(output_path, "rb") as f:
                image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode()
            
            result_image = flet.Image(
                src_base64=image_base64,
                fit=flet.ImageFit.CONTAIN,
                expand=True,
                height=600,
            )
            
            if self.result_view:
                self.result_view.content = result_image
                self.result_view.update()
            
        except Exception as e:
            print(f"检测过程出错: {str(e)}")
        # return run_detection_click  # 返回处理函数