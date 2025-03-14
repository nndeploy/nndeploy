import flet as ft
import sys
import os
from pathlib import Path
import flet.canvas

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from nndeploy.ui.config.settings import settings, get_setting, set_setting
from nndeploy.ui.config.theme import get_color, get_style

class SettingsDemo:
    """设置演示应用"""
    
    def __init__(self):
        self.page = None
        
    def main(self, page: ft.Page):
        """主函数"""
        self.page = page
        page.title = "NNDeploy 设置演示"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.padding = 20
        page.spacing = 20
        page.scroll = ft.ScrollMode.AUTO
        
        self.preview_settings = self.create_settings_ui()
        
        # 创建主界面
        page.add(
            ft.Text("NNDeploy 设置演示", size=30, weight=ft.FontWeight.BOLD),
            ft.Text("这个演示展示了如何使用设置系统并提供了可视化预览", size=16),
            ft.Divider(),
            self.preview_settings
        )
        
    def create_settings_ui(self):
        """创建设置界面"""
        # 创建设置标签页
        tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                ft.Tab(
                    text="画布设置",
                    icon=ft.Icons.GRID_ON,
                    content=self.create_setting_card("画布设置", settings.get_section("canvas"))
                ),
                ft.Tab(
                    text="节点设置",
                    icon=ft.Icons.WIDGETS,
                    content=self.create_setting_card("节点设置", settings.get_section("node"))
                ),
                ft.Tab(
                    text="连线设置",
                    icon=ft.Icons.TIMELINE,
                    content=self.create_setting_card("连线设置", settings.get_section("edge"))
                ),
                ft.Tab(
                    text="自动保存",
                    icon=ft.Icons.SAVE,
                    content=self.create_setting_card("自动保存设置", settings.get_section("auto_save"))
                ),
                ft.Tab(
                    text="性能设置",
                    icon=ft.Icons.SPEED,
                    content=self.create_setting_card("性能设置", settings.get_section("performance"))
                ),
                ft.Tab(
                    text="界面设置",
                    icon=ft.Icons.DASHBOARD_CUSTOMIZE,
                    content=self.create_setting_card("界面设置", settings.get_section("ui"))
                ),
            ],
            expand=1,
        )
        
        # 创建预览区域
        preview = self.create_preview_area()
        # print(preview)
        
        # 创建重置按钮
        reset_button = ft.ElevatedButton(
            "重置所有设置",
            icon=ft.Icons.RESTORE,
            on_click=self.reset_all_settings
        )
        
        # 创建导入/导出按钮
        import_export = ft.Row([
            ft.FilledButton(
                "导出设置",
                icon=ft.Icons.UPLOAD,
                on_click=self.export_settings
            ),
            ft.FilledButton(
                "导入设置",
                icon=ft.Icons.DOWNLOAD,
                on_click=self.import_settings
            ),
        ], alignment=ft.MainAxisAlignment.CENTER)
        
        print(tabs)
        print(preview)
        print(reset_button)
        print(import_export)
        
        # 确保返回完整的UI结构
        return ft.Column([
            ft.Container(
                content=ft.Text("设置"),
                expand=True,
            ),
            # ft.Row([
            #     ft.Container(
            #         content=tabs,
            #         expand=2,
            #         padding=10,
            #         border_radius=10,
            #         border=ft.border.all(1, get_color("border")),
            #     ),
            #     ft.Container(
            #         content=preview,
            #         expand=1,
            #         padding=10,
            #         border_radius=10,
            #         border=ft.border.all(1, get_color("border")),
            #     ),
            # ], expand=True),
            # ft.Container(
            #         content=tabs,
            #         expand=2,
            #         padding=10,
            #         border_radius=10,
            #         border=ft.border.all(1, get_color("border")),
            #     ),
                ft.Container(
                    content=preview,
                    expand=1,
                    padding=10,
                    border_radius=10,
                    border=ft.border.all(1, get_color("border")),
                ),
            ft.Row([
                reset_button,
                ft.Container(expand=True),
                import_export,
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ])
    
    def create_setting_card(self, title, section_settings):
        """创建设置卡片"""
        controls = []
        
        # 添加标题
        controls.append(ft.Text(title, size=20, weight=ft.FontWeight.BOLD))
        controls.append(ft.Divider())
        
        # 为每个设置创建控件
        for key, value in section_settings.items():
            # 根据值类型创建不同的控件
            if isinstance(value, bool):
                # 布尔值使用开关
                switch = ft.Switch(
                    value=value,
                    label=key.replace("_", " ").title(),
                )
                
                # 设置回调
                section = title.split("设置")[0].strip().lower()
                if section == "自动保存":
                    section = "auto_save"
                elif section == "性能":
                    section = "performance"
                elif section == "界面":
                    section = "ui"
                
                switch.on_change = lambda e, s=section, k=key: self.on_setting_changed(e, s, k)
                
                controls.append(ft.Row([
                    ft.Text(key.replace("_", " ").title(), expand=1),
                    switch,
                ]))
            elif isinstance(value, int) or isinstance(value, float):
                # 数值使用滑块或输入框
                if key.endswith("_size") or key.endswith("_width") or key.endswith("_height"):
                    # 尺寸类使用滑块
                    slider = ft.Slider(
                        min=0,
                        max=value * 2,
                        value=value,
                        divisions=100,
                        # 在flet 0.27.0中，label参数不再支持格式化字符串
                        # label="{value}",
                    )
                    
                    # 设置回调
                    section = title.split("设置")[0].strip().lower()
                    if section == "自动保存":
                        section = "auto_save"
                    elif section == "性能":
                        section = "performance"
                    elif section == "界面":
                        section = "ui"
                    
                    slider.on_change = lambda e, s=section, k=key: self.on_setting_changed(e, s, k)
                    
                    controls.append(ft.Column([
                        ft.Text(key.replace("_", " ").title()),
                        slider,
                    ]))
                else:
                    # 其他数值使用输入框
                    text_field = ft.TextField(
                        value=str(value),
                        label=key.replace("_", " ").title(),
                    )
                    
                    # 设置回调
                    section = title.split("设置")[0].strip().lower()
                    if section == "自动保存":
                        section = "auto_save"
                    elif section == "性能":
                        section = "performance"
                    elif section == "界面":
                        section = "ui"
                    
                    text_field.on_change = lambda e, s=section, k=key: self.on_setting_changed(e, s, k)
                    
                    controls.append(text_field)
            elif isinstance(value, str):
                # 字符串使用输入框或下拉菜单
                if key.endswith("_color"):
                    # 颜色使用颜色选择器
                    color_picker = ft.TextField(
                        value=value,
                        label=key.replace("_", " ").title(),
                        prefix_icon=ft.Icons.COLOR_LENS,
                    )
                    
                    # 设置回调
                    section = title.split("设置")[0].strip().lower()
                    if section == "自动保存":
                        section = "auto_save"
                    elif section == "性能":
                        section = "performance"
                    elif section == "界面":
                        section = "ui"
                    
                    color_picker.on_change = lambda e, s=section, k=key: self.on_setting_changed(e, s, k)
                    
                    controls.append(color_picker)
                elif key == "theme" or key == "toolbar_position" or key == "render_quality":
                    # 特定选项使用下拉菜单
                    options = []
                    if key == "theme":
                        options = ["light", "dark"]
                    elif key == "toolbar_position":
                        options = ["top", "bottom", "left", "right"]
                    elif key == "render_quality":
                        options = ["low", "medium", "high"]
                    
                    dropdown = ft.Dropdown(
                        options=[ft.dropdown.Option(option) for option in options],
                        value=value,
                        label=key.replace("_", " ").title(),
                    )
                    
                    # 设置回调
                    section = title.split("设置")[0].strip().lower()
                    if section == "自动保存":
                        section = "auto_save"
                    elif section == "性能":
                        section = "performance"
                    elif section == "界面":
                        section = "ui"
                    
                    dropdown.on_change = lambda e, s=section, k=key: self.on_setting_changed(e, s, k)
                    
                    controls.append(dropdown)
                else:
                    # 其他字符串使用输入框
                    text_field = ft.TextField(
                        value=value,
                        label=key.replace("_", " ").title(),
                    )
                    
                    # 设置回调
                    section = title.split("设置")[0].strip().lower()
                    if section == "自动保存":
                        section = "auto_save"
                    elif section == "性能":
                        section = "performance"
                    elif section == "界面":
                        section = "ui"
                    
                    text_field.on_change = lambda e, s=section, k=key: self.on_setting_changed(e, s, k)
                    
                    controls.append(text_field)
        
        # 创建重置按钮
        section = title.split("设置")[0].strip().lower()
        if section == "自动保存":
            section = "auto_save"
        elif section == "性能":
            section = "performance"
        elif section == "界面":
            section = "ui"
        
        reset_button = ft.OutlinedButton(
            "重置此分类",
            icon=ft.Icons.RESTORE,
            on_click=lambda e, s=section: self.reset_section(e, s)
        )
        
        controls.append(ft.Container(
            content=reset_button,
            alignment=ft.alignment.center_right,
            margin=ft.margin.only(top=20)
        ))
        
        return ft.Container(
            content=ft.Column(controls, scroll=ft.ScrollMode.AUTO),
            padding=10,
        )
    
    def on_setting_changed(self, e, section, key):
        """设置变更回调"""
        value = e.control.value
        
        # 根据控件类型转换值
        if isinstance(e.control, ft.Slider):
            if isinstance(get_setting(section, key), int):
                value = int(value)
            else:
                value = float(value)
        elif isinstance(e.control, ft.TextField) and key != "grid_color" and key != "line_color":
            try:
                # 尝试转换为数值
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # 保持为字符串
                pass
        
        # 更新设置
        set_setting(section, key, value)
        
        # 更新预览
        self.page.update()
    
    def reset_section(self, e, section):
        """重置分类设置"""
        settings.reset_section(section)
        # 刷新页面
        self.page.go(self.page.route)
    
    def reset_all_settings(self, e):
        """重置所有设置"""
        settings.reset_all()
        # 刷新页面
        self.page.go(self.page.route)
    
    def export_settings(self, e):
        """导出设置"""
        def on_dialog_result(e):
            if e.data == "确认":
                # 导出设置
                path = Path.home() / "nndeploy_settings.json"
                if settings.export_settings(path):
                    self.page.snack_bar = ft.SnackBar(
                        content=ft.Text(f"设置已导出到: {path}"),
                        action="确定"
                    )
                    self.page.snack_bar.open = True
                    self.page.update()
                else:
                    self.page.snack_bar = ft.SnackBar(
                        content=ft.Text("导出设置失败"),
                        action="确定"
                    )
                    self.page.snack_bar.open = True
                    self.page.update()
        
        # 显示确认对话框
        self.page.dialog = ft.AlertDialog(
            title=ft.Text("导出设置"),
            content=ft.Text("设置将导出到您的主目录。确认继续吗？"),
            actions=[
                ft.TextButton("取消", on_click=lambda e: self._close_dialog(e)),
                ft.TextButton("确认", on_click=lambda e: on_dialog_result(e)),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.dialog.open = True
        self.page.update()
        
    def _close_dialog(self, e):
        """关闭对话框"""
        self.page.dialog.open = False
        self.page.update()
    
    def import_settings(self, e):
        """导入设置"""
        def on_dialog_result(e):
            if e.data == "确认":
                # 导入设置
                path = Path.home() / "nndeploy_settings.json"
                if path.exists():
                    if settings.import_settings(path):
                        self.page.snack_bar = ft.SnackBar(
                            content=ft.Text("设置导入成功"),
                            action="确定"
                        )
                        self.page.snack_bar.open = True
                        # 刷新页面
                        self.page.go(self.page.route)
                    else:
                        self.page.snack_bar = ft.SnackBar(
                            content=ft.Text("导入设置失败"),
                            action="确定"
                        )
                        self.page.snack_bar.open = True
                        self.page.update()
                else:
                    self.page.snack_bar = ft.SnackBar(
                        content=ft.Text(f"找不到设置文件: {path}"),
                        action="确定"
                    )
                    self.page.snack_bar.open = True
                    self.page.update()
        
        # 显示确认对话框
        self.page.dialog = ft.AlertDialog(
            title=ft.Text("导入设置"),
            content=ft.Text("将从您的主目录导入设置。这将覆盖当前设置。确认继续吗？"),
            actions=[
                ft.TextButton("取消", on_click=lambda e: self._close_dialog(e)),
                ft.TextButton("确认", on_click=lambda e: on_dialog_result(e)),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.dialog.open = True
        self.page.update()
    
    def create_preview_area(self):
        """创建预览区域"""
        # 获取设置
        node_settings = settings.get_section("node")
        edge_settings = settings.get_section("edge")
        canvas_settings = settings.get_section("canvas")
        
        # 创建节点预览
        node_preview = ft.Container(
            width=node_settings["default_width"],
            height=node_settings["default_height"],
            border_radius=node_settings["border_radius"],
            bgcolor=get_color("surface"),
            border=ft.border.all(1, get_color("border")),
            padding=node_settings["padding"],
            content=ft.Column([
                ft.Container(
                    content=ft.Text(
                        "节点标题",
                        size=node_settings["font_size"],
                        weight=ft.FontWeight.BOLD,
                    ),
                    bgcolor=get_color("primary"),
                    padding=5,
                    border_radius=ft.border_radius.only(
                        top_left=node_settings["border_radius"],
                        top_right=node_settings["border_radius"],
                    ),
                ),
                ft.Container(
                    content=ft.Text(
                        "节点内容示例\n这是第二行",
                        size=node_settings["font_size"],
                    ),
                    expand=True,
                    padding=5,
                ),
            ]),
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=5,
                color=ft.Colors.with_opacity(0.3, ft.Colors.BLACK),
            ),
        )
        
        # 创建连线预览
        # 使用新版Canvas API创建连线
        canvas = ft.canvas.Canvas(
            width=300,
            height=100,
        )
        
        # 手动绘制连线
        self.draw_edge_on_canvas(canvas, edge_settings)
        
        edge_preview = ft.Container(
            width=300,
            height=100,
            content=ft.Stack([
                # 绘制连线
                ft.Container(
                    content=canvas,
                ),
                # 起点和终点标记
                ft.Container(
                    left=20,
                    top=50,
                    width=10,
                    height=10,
                    bgcolor=ft.Colors.RED,
                    border_radius=5,
                ),
                ft.Container(
                    left=270,
                    top=50,
                    width=10,
                    height=10,
                    bgcolor=ft.Colors.GREEN,
                    border_radius=5,
                ),
            ]),
        )
        
        return ft.Column([
            ft.Text("节点样式预览", weight=ft.FontWeight.BOLD),
            node_preview,
            ft.Divider(),
            ft.Text("连线样式预览", weight=ft.FontWeight.BOLD),
            edge_preview,
        ])
    
    def draw_edge_on_canvas(self, canvas, edge_settings):
        """在Canvas上绘制连线"""
        # 设置线宽
        line_width = edge_settings["line_width"]
        # 设置线颜色
        line_color = edge_settings["line_color"]
        # # 设置箭头大小
        # arrow_size = edge_settings["arrow_size"]
        # 设置曲线因子
        curve_factor = edge_settings["curve_factor"]
        
        # 起点和终点
        start_x, start_y = 25, 50
        end_x, end_y = 275, 50
        
        # 控制点
        ctrl_x1 = start_x + (end_x - start_x) * curve_factor
        ctrl_y1 = start_y
        ctrl_x2 = end_x - (end_x - start_x) * curve_factor
        ctrl_y2 = end_y
        
        # 绘制贝塞尔曲线
        canvas.shapes.append(
            ft.canvas.Path(
                [
                    ft.canvas.Path.MoveTo(start_x, start_y),
                    ft.canvas.Path.CubicTo(ctrl_x1, ctrl_y1, ctrl_x2, ctrl_y2, end_x, end_y)
                ],
                paint=ft.Paint(
                    stroke_width=line_width,
                    style=ft.PaintingStyle.STROKE,
                    color=line_color,
                )
            )
        )
        
        # 绘制箭头
        canvas.shapes.append(
            ft.canvas.Path(
                [
                    ft.canvas.Path.MoveTo(end_x, end_y),
                    ft.canvas.Path.LineTo(end_x, end_y / 2),
                    ft.canvas.Path.LineTo(end_x, end_y / 2),
                    ft.canvas.Path.Close()
                ],
                paint=ft.Paint(
                    style=ft.PaintingStyle.FILL,
                    color=line_color,
                )
            )
        )

# # 创建设置分类标签页
# tabs = ft.Tabs(
#     selected_index=0,
#     animation_duration=300,
#     tabs=[
#         ft.Tab(
#             text="画布设置",
#             icon=ft.Icons.GRID_ON,
#             content=create_setting_card("画布设置", settings.get_section("canvas"))
#         ),
#         ft.Tab(
#             text="节点设置",
#             icon=ft.Icons.WIDGETS,
#             content=create_setting_card("节点设置", settings.get_section("node"))
#         ),
#         ft.Tab(
#             text="连线设置",
#             icon=ft.Icons.TIMELINE,
#             content=create_setting_card("连线设置", settings.get_section("edge"))
#         ),
#         ft.Tab(
#             text="自动保存",
#             icon=ft.Icons.SAVE,
#             content=create_setting_card("自动保存设置", settings.get_section("auto_save"))
#         ),
#         ft.Tab(
#             text="性能设置",
#             icon=ft.Icons.SPEED,
#             content=create_setting_card("性能设置", settings.get_section("performance"))
#         ),
#         ft.Tab(
#             text="界面设置",
#             icon=ft.Icons.DASHBOARD_CUSTOMIZE,
#             content=create_setting_card("界面设置", settings.get_section("ui"))
#         ),
#     ],
#     expand=1,
# )

if __name__ == "__main__":
    app = SettingsDemo()
    ft.app(target=app.main, view=ft.WEB_BROWSER, port=9090)
