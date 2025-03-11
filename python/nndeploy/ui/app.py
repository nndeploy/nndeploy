import flet
from flet import View, Page, Theme, ThemeMode, colors
import os

# 修改导入路径
from view.home_view import HomeView
from view.apps.object_detection_view import ObjectDetectionView
from view.workflow_view import WorkflowView
from view.about_us_view import AboutUsView
import settings

def app(page: Page):
    # 设置主题
    page.theme_mode = ThemeMode.LIGHT
    page.theme = Theme(color_scheme_seed=flet.Colors.BLUE)
    
    # 设置窗口大小
    # page.window_width = 1200
    # page.window_height = 800
    # page.padding = 0
    page.scroll = None
    page.bgcolor = flet.Colors.WHITE
    
    # 初始化 FilePicker 并添加到页面
    # 创建一个FilePicker实例并将其作为page对象的属性
    # FilePicker是Flet中用于选择文件的组件，可以让用户从本地文件系统中选择文件
    # page.file_picker = flet.FilePicker()  
    
    # 将FilePicker添加到页面的overlay层
    # overlay是Flet中的一个特殊层，用于显示悬浮在页面主内容之上的UI元素
    # FilePicker必须添加到overlay中才能正常工作，因为它需要在用户交互时显示文件选择对话框
    # page.overlay.append(page.file_picker)  
    
    # 视图映射
    views = {
        "/": HomeView(page),
        "/app/object_detection": ObjectDetectionView(page),
        "/workflow": WorkflowView(page),
        "/about": AboutUsView(page),
    }
    
    def route_change(route):
        print(f"Current route: {page.route}")  # 添加调试信息
        page.views.clear()
        view = views.get(page.route)
        if view:
            built_view = view.build()
            print(f"View controls: {built_view.controls}")  # 添加调试信息
            page.views.append(built_view)
            page.update()
        else:
            page.go("/")

    def view_pop(view):
        page.views.pop()
        top_view = page.views[-1]
        page.go(top_view.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop
    
    # 初始化路由
    page.go("/")

if __name__ == "__main__":
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    #view=flet.WEB_BROWSER
    flet.app(target=app, assets_dir=assets_dir, view=flet.WEB_BROWSER, port=9090)
    # flet.app(target=main, assets_dir=assets_dir)