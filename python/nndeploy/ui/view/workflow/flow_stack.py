import flet
from flet.canvas import Canvas, Path
from flet.core.painting import Paint, PaintingStyle
from flet.core.types import StrokeCap
import math
from typing import List
from flet import (
    Container,
    Stack,
    DragTarget,
    Draggable,
    Colors,
    Page,
    GestureDetector,
)

from custom_nodes import ImageNode, TextNode, VideoNode
from connection import Connection
from slot import Slot
from canvas_manager import CanvasManager

class FlowStack:
    def __init__(self, page: Page):
        self.page = page
        
        # 创建画布管理器，使用页面尺寸
        self.canvas_manager = CanvasManager(page.window_width, page.window_height)
        
        self.container = Container(
            content=self.canvas_manager.gesture_detector,
            bgcolor=Colors.BLUE_GREY_50,
            border_radius=10,
            padding=20,
            expand=True,  # 允许容器扩展填充可用空间
        )
        
        # 添加窗口大小变化事件处理
        def on_resize(e):
            # 更新画布管理器尺寸
            self.canvas_manager.width = page.window_width
            self.canvas_manager.height = page.window_height
            
            # 更新容器尺寸
            self.container.width = page.window_width
            self.container.height = page.window_height
            
            # 更新画布和手势检测器的尺寸
            self.canvas_manager.canvas.width = page.window_width
            self.canvas_manager.canvas.height = page.window_height
            self.canvas_manager.gesture_detector.width = page.window_width
            self.canvas_manager.gesture_detector.height = page.window_height
            
            # 重新计算所有连接线
            self.canvas_manager.update_connections()
            
            # 刷新页面
            self.page.update()
            
        page.on_resize = on_resize
        
        # 先将容器添加到页面
        page.add(self.container)
        
        # 创建示例节点，添加视频节点
        video_node = VideoNode("视频播放", position=(100, 100), page=page, canvas_manager=self.canvas_manager)
        image_node = ImageNode("图像处理", position=(400, 100), page=page, canvas_manager=self.canvas_manager)
        text_node = TextNode("文本处理", position=(700, 100), page=page, canvas_manager=self.canvas_manager)
        
        # 设置节点位置更新回调
        def on_node_position_changed():
            self.canvas_manager.update_connections()
            self.page.update()
        
        video_node.on_position_changed = on_node_position_changed
        image_node.on_position_changed = on_node_position_changed
        text_node.on_position_changed = on_node_position_changed
        
        # 使用 canvas_manager 添加节点
        self.canvas_manager.add_node(video_node)
        self.canvas_manager.add_node(image_node)
        self.canvas_manager.add_node(text_node)
        
        # 添加连接
        if "frame" in video_node.output_slots and "image" in image_node.input_slots:
            self.connect_slots(
                video_node.output_slots["frame"],
                image_node.input_slots["image"]
            )
        
        if "processed" in image_node.output_slots and "text" in text_node.input_slots:
            self.connect_slots(
                image_node.output_slots["processed"],
                text_node.input_slots["text"]
            )
    
    def connect_slots(self, from_slot: Slot, to_slot: Slot):
        """连接两个插槽"""
        self.canvas_manager.add_connection(from_slot, to_slot)
        self.page.update()
    
    def update_connections(self):
        """更新所有连接线"""
        self.canvas_manager.update_connections()
        self.page.update()
    
    def remove_connection(self, connection: Connection):
        """移除连接"""
        # 使用 canvas_manager 的 connections 列表
        self.canvas_manager.remove_connection(connection)
        self.page.update()

def main(page: Page):
    page.title = "节点编辑器"
    page.window_width = 1200
    page.window_height = 800
    page.padding = 20
    
    color_blocks = FlowStack(page)
    page.update()

if __name__ == "__main__":
    #view=flet.WEB_BROWSER
    flet.app(target=main)