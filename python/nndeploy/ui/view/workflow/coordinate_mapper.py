from typing import Tuple, Optional, Any
import flet as ft
from flet.canvas import Canvas

class CoordinateMapper:
    """坐标映射工具，用于在不同控件之间转换坐标"""
    
    @staticmethod
    def map_to_parent(control: ft.Control, point: Tuple[float, float]) -> Tuple[float, float]:
        """将控件的本地坐标映射到父控件的坐标系"""
        x, y = point
        
        # 获取控件的尺寸
        width = float(control.width) if hasattr(control, 'width') and control.width is not None else 0
        height = float(control.height) if hasattr(control, 'height') and control.height is not None else 0
        
        # 添加控件自身的位置偏移
        if hasattr(control, 'left') and control.left is not None:
            x += float(control.left)
        if hasattr(control, 'top') and control.top is not None:
            y += float(control.top)
        
        # 处理对齐方式
        if hasattr(control, 'alignment') and control.alignment is not None:
            if hasattr(control.alignment, 'x'):
                x += float(control.alignment.x) * width
            if hasattr(control.alignment, 'y'):
                y += float(control.alignment.y) * height
        
        # 处理外边距
        if hasattr(control, 'margin') and control.margin is not None:
            margin = control.margin
            if isinstance(margin, (int, float)):
                x += float(margin)
                y += float(margin)
            else:
                if hasattr(margin, 'left') and margin.left is not None:
                    x += float(margin.left)
                if hasattr(margin, 'top') and margin.top is not None:
                    y += float(margin.top)
        
        return (x, y)
    
    @staticmethod
    def map_to_canvas(control: ft.Control, point: Tuple[float, float]) -> Tuple[float, float]:
        """将控件的本地坐标映射到画布坐标系"""
        x, y = point
        current = control
        total_x = float(x)
        total_y = float(y)
        
        # 向上遍历所有父容器
        while current and not isinstance(current, Canvas):
            # 映射到父控件
            parent_x, parent_y = CoordinateMapper.map_to_parent(current, (0, 0))
            total_x += parent_x
            total_y += parent_y
            
            # 处理容器布局
            if isinstance(current, ft.Row):
                for child in current.controls:
                    if child == control:
                        break
                    if hasattr(child, 'width') and child.width is not None:
                        total_x += float(child.width)
                    if hasattr(current, 'spacing') and current.spacing is not None:
                        total_x += float(current.spacing)
            
            elif isinstance(current, ft.Column):
                for child in current.controls:
                    if child == control:
                        break
                    if hasattr(child, 'height') and child.height is not None:
                        total_y += float(child.height)
                    if hasattr(current, 'spacing') and current.spacing is not None:
                        total_y += float(current.spacing)
            
            current = current.parent
        
        return (total_x, total_y)
    
    @staticmethod
    def map_from_canvas(canvas: Canvas, control: ft.Control, point: Tuple[float, float]) -> Tuple[float, float]:
        """将画布坐标映射到指定控件的本地坐标系"""
        x, y = point
        path = []
        current = control
        
        # 构建从控件到画布的路径
        while current and not isinstance(current, Canvas):
            path.append(current)
            current = current.parent
        
        # 从画布反向映射到控件
        for ctrl in reversed(path):
            # 减去父控件的内边距
            parent = ctrl.parent
            if parent and hasattr(parent, 'padding') and parent.padding is not None:
                padding = parent.padding
                if hasattr(padding, 'left') and padding.left is not None:
                    x -= padding.left
                if hasattr(padding, 'top') and padding.top is not None:
                    y -= padding.top
            
            # 减去控件的位置偏移和外边距
            if hasattr(ctrl, 'left') and ctrl.left is not None:
                x -= ctrl.left
            if hasattr(ctrl, 'top') and ctrl.top is not None:
                y -= ctrl.top
            
            if hasattr(ctrl, 'margin') and ctrl.margin is not None:
                margin = ctrl.margin
                if hasattr(margin, 'left') and margin.left is not None:
                    x -= margin.left
                if hasattr(margin, 'top') and margin.top is not None:
                    y -= margin.top
        
        return (x, y)