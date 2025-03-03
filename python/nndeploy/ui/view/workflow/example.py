import cv2
import flet as ft
import base64
import numpy as np
import threading
import time

def update_video(page: ft.Page, image_control: ft.Image):
    # 打开视频文件
    cap = cv2.VideoCapture("/Users/tguang/229373720-14d69157-1a56-4a78-a2f4-d7a134d7c3e9.mp4")  # 替换为你的视频文件路径
    
    if not cap.isOpened():
        print("无法打开视频文件")
        return
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 调整视频帧大小
        max_width = 800
        scale = max_width / frame.shape[1]
        new_width = int(frame.shape[1] * scale)
        new_height = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, (new_width, new_height))
        
        # 转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 编码为base64
        _, buffer = cv2.imencode(".jpg", frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_base64 = base64.b64encode(buffer).decode()
        
        # 更新图像控件
        image_control.src_base64 = image_base64
        page.update()
        
        # 控制帧率
        time.sleep(1/30)  # 约30fps
        
    cap.release()

def main(page: ft.Page):
    page.title = "Video Player"
    
    # 创建图像控件
    image_control = ft.Image(
        width=800,
        height=450,
        fit=ft.ImageFit.CONTAIN,
    )
    
    # 创建容器
    container = ft.Container(
        content=image_control,
        alignment=ft.alignment.center,
        border=ft.border.all(1, ft.Colors.GREY_400),
        border_radius=10,
        padding=10,
    )
    
    # 添加到页面
    page.add(container)
    
    # 在新线程中运行视频更新
    threading.Thread(target=update_video, args=(page, image_control), daemon=True).start()

# 启动应用
ft.app(target=main)