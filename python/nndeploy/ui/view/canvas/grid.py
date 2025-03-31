"""
画布网格组件

负责:
- 绘制画布背景网格
- 支持无限画布功能，允许用户向任何方向拖拉扩展画布
"""

# 导入类型提示相关模块
from typing import Optional, Tuple, List  # 导入类型提示工具
import flet as ft  # 导入Flet UI框架
import flet.canvas  # 导入Flet画布模块
from nndeploy.ui.config import get_color, get_style, settings  # 导入配置相关函数和设置
from zoom import ZoomControl

class Grid:
    """画布网格组件
    
    - 绘制画布背景网格
    - 支持无限画布功能，允许用户向任何方向拖拉扩展画布
    """
    
    def __init__(
        self,
        page: ft.Page,  # Flet页面对象
        width: float,  # 画布宽度
        height: float,  # 画布高度
        cell_size: int = 20,  # 网格单元格大小，默认20像素
        color: Optional[str] = None,  # 网格线颜色，可选参数
        opacity: float = 0.3,  # 网格线透明度，默认0.3
    ):
        """初始化网格组件
        
        Args:
            page: Flet页面对象，用于更新UI
            width: 画布宽度，决定网格的水平范围
            height: 画布高度，决定网格的垂直范围
            cell_size: 网格单元格大小(像素)，控制网格的密度
            color: 网格线颜色，默认使用主题的secondary颜色
            opacity: 网格线透明度(0-1)，控制网格线的可见度
        """
        # 存储网格基本属性
        self.page = page  # 保存页面引用
        self.width = width  # 设置画布宽度
        self.height = height  # 设置画布高度
        self.cell_size = cell_size  # 设置网格单元格大小
        self.base_color = color or get_color("secondary")  # 设置基础颜色，如果未提供则使用主题secondary颜色
        self.opacity = opacity  # 设置透明度
        
        # 创建带透明度的颜色
        # 将透明度值(0-1)转换为十六进制(00-FF)
        alpha_hex = hex(int(opacity * 255))[2:].zfill(2).upper()  # 将透明度转换为十六进制
        if self.base_color.startswith('#') and len(self.base_color) == 7:
            # 如果是#RRGGBB格式，转换为#AARRGGBB格式
            self.color = f"#{alpha_hex}{self.base_color[1:]}"  # 添加透明度到颜色值
        else:
            # 如果是其他格式，保持原样并警告opacity可能不生效
            self.color = self.base_color  # 保持原始颜色
            print(f"Warning: opacity may not work with color format: {self.base_color}")  # 打印警告信息
        
        # 无限画布相关属性
        self.offset_x = 0  # 画布X轴偏移量，初始为0
        self.offset_y = 0  # 画布Y轴偏移量，初始为0
        self.is_dragging = False  # 是否正在拖动画布，初始为False
        self.drag_start_x = 0  # 拖动起始X坐标，初始为0
        self.drag_start_y = 0  # 拖动起始Y坐标，初始为0
        
        # 存储内容项
        self.contents = []  # 初始化空列表存储画布上的内容
        
        # 创建控件
        self.canvas = ft.canvas.Canvas(  # 创建Flet画布对象
            width=self.width,  # 设置画布宽度
            height=self.height,  # 设置画布高度
            shapes=[]  # 初始化空形状列表
        )
        
        # 绘制初始网格
        self._draw_grid()  # 调用方法绘制初始网格线
        
        # 创建可拖动的容器
        self.container = ft.GestureDetector(  # 创建手势检测器
            mouse_cursor=ft.MouseCursor.MOVE,  # 设置鼠标光标为移动样式
            on_pan_start=self._on_pan_start,  # 设置拖动开始事件处理函数
            on_pan_update=self._on_pan_update,  # 设置拖动更新事件处理函数
            on_pan_end=self._on_pan_end,  # 设置拖动结束事件处理函数
            content=ft.Stack([  # 创建堆叠布局
                # 背景层 - 纯色背景
                ft.Container(  # 创建背景容器
                    width=self.width,  # 设置背景宽度
                    height=self.height,  # 设置背景高度
                    bgcolor=get_color("background")  # 设置背景颜色
                ),
                # 网格线层
                self.canvas,  # 添加画布作为网格线层
            ])
        )
        
    def _draw_grid(self):
        """绘制网格线
        
        根据当前的宽度、高度、单元格大小和偏移量，绘制所有水平和垂直网格线。
        考虑画布偏移量，实现无限画布效果。
        """
        # 清除现有网格线
        self.canvas.shapes.clear()  # 清空画布上的所有形状
        
        # 计算可见区域的起始和结束坐标（考虑偏移量）
        start_x = -self.offset_x % self.cell_size  # 计算水平方向第一条线的位置
        start_y = -self.offset_y % self.cell_size  # 计算垂直方向第一条线的位置
        
        # 绘制水平线
        y = start_y  # 初始化y坐标
        while y < self.height:  # 当y坐标小于画布高度时循环
            self.canvas.shapes.append(  # 向画布添加形状
                ft.canvas.Line(  # 创建线条
                    x1=0, y1=y, x2=self.width, y2=y,  # 设置线条起点和终点坐标
                    paint=ft.Paint(  # 设置绘制属性
                        color=self.color,  # 设置线条颜色
                        stroke_width=1  # 设置线条宽度为1像素
                    )
                )
            )
            y += self.cell_size  # y坐标增加一个单元格大小
        
        # 绘制垂直线
        x = start_x  # 初始化x坐标
        while x < self.width:  # 当x坐标小于画布宽度时循环
            self.canvas.shapes.append(  # 向画布添加形状
                ft.canvas.Line(  # 创建线条
                    x1=x, y1=0, x2=x, y2=self.height,  # 设置线条起点和终点坐标
                    paint=ft.Paint(  # 设置绘制属性
                        color=self.color,  # 设置线条颜色
                        stroke_width=1  # 设置线条宽度为1像素
                    )
                )
            )
            x += self.cell_size  # x坐标增加一个单元格大小
    
    def _update_content_position(self):
        """更新内容层中所有元素的位置
        
        根据画布偏移量，更新所有内容项的位置，使其与画布拖动同步
        """
        # 使用stack获取内容层
        if isinstance(self.container.content, ft.Stack):  # 检查容器内容是否为Stack类型
            stack = self.container.content  # 获取Stack对象
            
            # 更新所有内容项的位置
            for content_item in self.contents:  # 遍历所有内容项
                control = content_item["control"]  # 获取控件引用
                canvas_x = content_item["canvas_x"]  # 获取控件在画布上的X坐标
                canvas_y = content_item["canvas_y"]  # 获取控件在画布上的Y坐标
                
                # 更新位置
                control.left = canvas_x + self.offset_x  # 计算并设置控件的左侧位置
                control.top = canvas_y + self.offset_y  # 计算并设置控件的顶部位置
    
    def _on_pan_start(self, e):
        """开始拖动画布
        
        记录拖动起始位置
        
        Args:
            e: 拖动事件对象
        """
        self.is_dragging = True  # 设置拖动状态为True
        self.drag_start_x = e.local_x  # 记录拖动起始X坐标
        self.drag_start_y = e.local_y  # 记录拖动起始Y坐标
    
    def _on_pan_update(self, e):
        """更新画布拖动
        
        根据拖动距离更新画布偏移量
        
        Args:
            e: 拖动事件对象
        """
        if not self.is_dragging:  # 如果不是拖动状态
            return  # 直接返回，不执行后续代码
            
        # 计算拖动距离
        delta_x = e.local_x - self.drag_start_x  # 计算X方向拖动距离
        delta_y = e.local_y - self.drag_start_y  # 计算Y方向拖动距离
        
        # 更新偏移量
        self.offset_x += delta_x  # 更新X轴偏移量
        self.offset_y += delta_y  # 更新Y轴偏移量
        
        # 更新拖动起始位置
        self.drag_start_x = e.local_x  # 更新拖动起始X坐标
        self.drag_start_y = e.local_y  # 更新拖动起始Y坐标
        
        # 重新绘制网格线
        self._draw_grid()  # 调用方法重新绘制网格
        
        # 更新内容层中的元素位置
        self._update_content_position()  # 调用方法更新内容位置
        
        # 更新显示
        if self.container.page:  # 如果容器已添加到页面
            self.container.update()  # 更新容器显示
    
    def _on_pan_end(self, e):
        """结束画布拖动
        
        重置拖动状态
        
        Args:
            e: 拖动事件对象
        """
        self.is_dragging = False  # 设置拖动状态为False
    
    def add_content(self, control, canvas_x=0, canvas_y=0):
        """添加内容到画布
        
        将控件添加到画布上的指定位置
        
        Args:
            control: 要添加的控件
            canvas_x: 控件在画布上的X坐标
            canvas_y: 控件在画布上的Y坐标
        """
        # 计算控件在屏幕上的实际位置
        control.left = canvas_x + self.offset_x  # 设置控件的左侧位置
        control.top = canvas_y + self.offset_y  # 设置控件的顶部位置
        
        # 存储内容项信息
        self.contents.append({  # 向内容列表添加新项
            "control": control,  # 存储控件引用
            "canvas_x": canvas_x,  # 存储画布X坐标
            "canvas_y": canvas_y  # 存储画布Y坐标
        })
        
        # 将控件添加到Stack中
        if isinstance(self.container.content, ft.Stack):  # 检查容器内容是否为Stack类型
            stack = self.container.content  # 获取Stack对象
            stack.controls.append(control)  # 将控件添加到Stack的控件列表
            
            # 只有当container已添加到页面时才调用update
            if self.container.page:  # 如果容器已添加到页面
                self.container.update()  # 更新容器显示
            
    def resize(self, width: float, height: float):
        """调整网格大小
        
        当画布大小变化时调用此方法，重新计算网格线并更新显示。
        
        Args:
            width: 新宽度，单位为像素
            height: 新高度，单位为像素
        """
        # 更新尺寸属性
        self.width = width  # 更新画布宽度
        self.height = height  # 更新画布高度
        
        # 更新画布尺寸
        self.canvas.width = width  # 更新Canvas控件宽度
        self.canvas.height = height  # 更新Canvas控件高度
        
        # 更新背景层尺寸
        if isinstance(self.container.content, ft.Stack) and len(self.container.content.controls) > 0:  # 检查容器内容是否为Stack且有子控件
            background = self.container.content.controls[0]  # 获取第一个子控件作为背景
            if isinstance(background, ft.Container):  # 检查背景是否为Container类型
                background.width = width  # 更新背景宽度
                background.height = height  # 更新背景高度
        
        # 重新绘制网格线
        self._draw_grid()  # 调用方法重新绘制网格
        
        # 更新显示
        if self.container.page:  # 如果容器已添加到页面
            self.container.update()  # 更新容器显示

if __name__ == "__main__":  # 如果直接运行此文件
    def main(page: ft.Page):  # 定义主函数
        # 设置页面属性
        page.title = "无限画布测试"  # 设置页面标题
        page.padding = 0  # 设置页面内边距为0
        page.bgcolor = get_color("background")  # 设置页面背景色
        
        # 获取窗口尺寸
        # 获取屏幕尺寸并设置窗口大小与屏幕一致
        # page.width = page.window_client_width  # 设置窗口宽度与屏幕宽度一致
        # page.height = page.window_client_height  # 设置窗口高度与屏幕高度一致
        page.window_resizable = True  # 允许调整窗口大小
        
        # 创建满屏网格
        # grid = Grid(page, width=page.width, height=page.height)  # 创建网格对象
        grid = Grid(page, width=page.width, height=page.height)  # 创建网格对象
        
        # 创建缩放控制器
        def on_scale_change(scale):
            # 更新所有内容的缩放
            for content_item in grid.contents:
                control = content_item["control"]
                control.scale = scale
                # 调整位置以保持中心点不变
                canvas_x = content_item["canvas_x"]
                canvas_y = content_item["canvas_y"]
                control.left = canvas_x * scale + grid.offset_x
                control.top = canvas_y * scale + grid.offset_y
                control.update()
        
        zoom_control = ZoomControl(
            min_scale=0.5,
            max_scale=2.0,
            step=0.1,
            on_scale_change=on_scale_change
        )
        
        # 创建布局并添加到页面
        layout = ft.Stack([
            grid.container,
            ft.Container(
                content=zoom_control.control,
                alignment=ft.alignment.bottom_left,
                padding=ft.padding.only(left=10, bottom=10)
            )
        ])  # 创建堆叠布局并添加网格容器和缩放控制器
        
        page.add(layout)  # 将布局添加到页面
        
        # 添加一些测试节点 - 模拟Dify图片中的节点
        # 创建节点拖动处理函数
        def on_node_drag_start(e, node):  # 节点拖动开始
            node.start_x = node.left
            node.start_y = node.top
            node.drag_start_x = e.local_x
            node.drag_start_y = e.local_y
            
        def on_node_drag_update(e, node):  # 节点拖动更新
            delta_x = e.local_x - node.drag_start_x
            delta_y = e.local_y - node.drag_start_y
            node.left = node.start_x + delta_x
            node.top = node.start_y + delta_y
            # 更新节点在画布中的位置记录
            for content_item in grid.contents:
                if content_item["control"] == node:
                    content_item["canvas_x"] = node.left - grid.offset_x
                    content_item["canvas_y"] = node.top - grid.offset_y
                    break
            node.update()
            
        def on_node_drag_end(e, node):  # 节点拖动结束
            pass  # 可以添加拖动结束后的逻辑
            
        start_node = ft.GestureDetector(  # 创建可拖动的开始节点
            mouse_cursor=ft.MouseCursor.MOVE,  # 设置鼠标光标为移动样式
            drag_interval=10,  # 设置拖动间隔
            on_pan_start=lambda e: on_node_drag_start(e, start_node),  # 设置拖动开始事件
            on_pan_update=lambda e: on_node_drag_update(e, start_node),  # 设置拖动更新事件
            on_pan_end=lambda e: on_node_drag_end(e, start_node),  # 设置拖动结束事件
            content=ft.Container(  # 创建开始节点容器
                width=100,  # 设置宽度
                height=40,  # 设置高度
                border_radius=20,  # 设置边框圆角
                bgcolor="#2563EB",  # 设置背景色
                content=ft.Row([  # 创建行布局作为内容
                    ft.Icon(name=ft.Icons.HOME, color=ft.Colors.WHITE),  # 添加家图标
                    ft.Text("开始", color=ft.Colors.WHITE)  # 添加文本
                ], alignment=ft.MainAxisAlignment.CENTER),  # 设置行布局居中对齐
            )
        )
        
        llm_node = ft.GestureDetector(  # 创建可拖动的LLM节点
            mouse_cursor=ft.MouseCursor.MOVE,  # 设置鼠标光标为移动样式
            drag_interval=10,  # 设置拖动间隔
            on_pan_start=lambda e: on_node_drag_start(e, llm_node),  # 设置拖动开始事件
            on_pan_update=lambda e: on_node_drag_update(e, llm_node),  # 设置拖动更新事件
            on_pan_end=lambda e: on_node_drag_end(e, llm_node),  # 设置拖动结束事件
            content=ft.Container(  # 创建LLM节点
                width=160,  # 设置宽度
                height=80,  # 设置高度
                border_radius=10,  # 设置边框圆角
                bgcolor="#FFFFFF",  # 设置背景色
                border=ft.border.all(2, "#2563EB"),  # 设置边框
                content=ft.Column([  # 创建列布局作为内容
                    ft.Row([  # 创建第一行
                        ft.Icon(name=ft.Icons.SMART_TOY, color="#2563EB"),  # 添加机器人图标
                        ft.Text("LLM", color="#2563EB"),  # 添加文本
                        ft.Container(width=60),  # 添加占位容器
                        ft.Icon(name=ft.Icons.ADD_CIRCLE_OUTLINE, color="#2563EB"),  # 添加添加图标
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),  # 设置行布局两端对齐
                    ft.Container(  # 创建子容器
                        content=ft.Text("gpt-3.5-turbo", size=12),  # 添加文本
                        bgcolor="#F3F4F6",  # 设置背景色
                        padding=5,  # 设置内边距
                        border_radius=5,  # 设置边框圆角
                    )
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=5)  # 设置列布局居中对齐，间距为5
            )
        )
        
        search_node = ft.GestureDetector(  # 创建可拖动的搜索节点
            mouse_cursor=ft.MouseCursor.MOVE,  # 设置鼠标光标为移动样式
            drag_interval=10,  # 设置拖动间隔
            on_pan_start=lambda e: on_node_drag_start(e, search_node),  # 设置拖动开始事件
            on_pan_update=lambda e: on_node_drag_update(e, search_node),  # 设置拖动更新事件
            on_pan_end=lambda e: on_node_drag_end(e, search_node),  # 设置拖动结束事件
            content=ft.Container(  # 创建搜索节点
                width=100,  # 设置宽度
                height=40,  # 设置高度
                border_radius=20,  # 设置边框圆角
                bgcolor="#10B981",  # 设置背景色
                content=ft.Row([  # 创建行布局作为内容
                    ft.Icon(name=ft.Icons.SEARCH, color=ft.Colors.WHITE),  # 添加搜索图标
                    ft.Text("知识检索", color=ft.Colors.WHITE, size=12)  # 添加文本
                ], alignment=ft.MainAxisAlignment.CENTER),  # 设置行布局居中对齐
            )
        )
        
        end_node = ft.GestureDetector(  # 创建可拖动的结束节点
            mouse_cursor=ft.MouseCursor.MOVE,  # 设置鼠标光标为移动样式
            drag_interval=10,  # 设置拖动间隔
            on_pan_start=lambda e: on_node_drag_start(e, end_node),  # 设置拖动开始事件
            on_pan_update=lambda e: on_node_drag_update(e, end_node),  # 设置拖动更新事件
            on_pan_end=lambda e: on_node_drag_end(e, end_node),  # 设置拖动结束事件
            content=ft.Container(  # 创建结束节点
                width=100,  # 设置宽度
                height=40,  # 设置高度
                border_radius=20,  # 设置边框圆角
                bgcolor="#F59E0B",  # 设置背景色
                content=ft.Row([  # 创建行布局作为内容
                    ft.Icon(name=ft.Icons.CHECK_CIRCLE, color=ft.Colors.WHITE),  # 添加勾选图标
                    ft.Text("结束", color=ft.Colors.WHITE)  # 添加文本
                ], alignment=ft.MainAxisAlignment.CENTER),  # 设置行布局居中对齐
            )
        )
        
        # 添加节点到画布 - 中心位置为原点
        center_x = page.width / 2  # 计算中心X坐标
        center_y = page.height / 2  # 计算中心Y坐标
        
        grid.add_content(start_node, center_x - 300, center_y)  # 添加开始节点到左侧
        grid.add_content(llm_node, center_x - 50, center_y + 100)  # 添加LLM节点到下方
        grid.add_content(search_node, center_x, center_y - 100)  # 添加搜索节点到上方
        grid.add_content(end_node, center_x + 300, center_y)  # 添加结束节点到右侧
        
        # 添加参考点标记
        center_mark = ft.Container(  # 创建中心标记
            width=10,  # 设置宽度
            height=10,  # 设置高度
            bgcolor=ft.Colors.RED,  # 设置背景色为红色
            border_radius=5,  # 设置边框圆角
            left=(page.width / 2) - 5,  # 设置左侧位置
            top=(page.height / 2) - 5,  # 设置顶部位置
        )
        # layout.controls.append(center_mark)  # 将中心标记添加到布局
        grid.add_content(center_mark, page.width / 2 - 5, page.height / 2 - 5)  # 添加中心标记到画布
        
        # 添加坐标信息文本
        coords_text = ft.Text(  # 创建文本控件
            f"偏移量: (0, 0) 缩放: 100%",  # 设置初始文本
            color=ft.Colors.WHITE,  # 设置文本颜色
            bgcolor=ft.Colors.BLACK54,  # 设置背景色
            size=14,  # 设置文本大小
            left=10,  # 设置左侧位置
            top=10,  # 设置顶部位置
        )
        # layout.controls.append(coords_text)  # 将坐标文本添加到布局
        grid.add_content(coords_text, 10, 10)  # 添加坐标文本到画布
        
        # 更新坐标信息
        def update_coords():  # 定义更新坐标的函数
            scale = zoom_control._scale
            coords_text.value = f"偏移量: ({grid.offset_x:.1f}, {grid.offset_y:.1f}) 缩放: {int(scale * 100)}%"  # 更新文本内容
            coords_text.update()  # 更新文本控件
            
        # 监听拖动事件
        original_on_pan_update = grid._on_pan_update  # 保存原始拖动更新方法
        def on_pan_update_with_coords(e):  # 定义新的拖动更新方法
            original_on_pan_update(e)  # 调用原始方法
            update_coords()  # 更新坐标信息
        grid._on_pan_update = on_pan_update_with_coords  # 替换拖动更新方法
        
        # 更新原始缩放回调以更新坐标信息
        original_on_scale_change = on_scale_change
        def on_scale_change_with_coords(scale):
            original_on_scale_change(scale)
            update_coords()
        zoom_control.on_scale_change = on_scale_change_with_coords
        
        # 处理鼠标滚轮事件
        def on_canvas_wheel(e):
            if zoom_control.on_wheel(e):
                e.prevent_default = True
        
        # 将滚轮事件处理器添加到网格容器
        grid.container.on_scroll = on_canvas_wheel
        
        # 监听窗口大小变化
        def on_resize(e):  # 定义窗口大小变化处理函数
            grid.resize(page.width, page.height)  # 调整网格大小
            center_mark.left = (page.width / 2) - 5  # 更新中心标记左侧位置
            center_mark.top = (page.height / 2) - 5  # 更新中心标记顶部位置
            center_mark.update()  # 更新中心标记
        
        page.on_resize = on_resize  # 设置窗口大小变化事件处理函数
    
    ft.app(target=main, view=ft.WEB_BROWSER, port=8080)  # 启动Flet应用，使用Web浏览器视图，端口9090