import flet
from flet import View, Page, Theme, ThemeMode, colors
import os

# 修改导入路径
from view.home_view import HomeView
from view.apps.object_detection_view import ObjectDetectionView
from view.workflow_view import WorkflowView
from view.about_us_view import AboutUsView
import settings

def main(page: Page):
    # 设置主题
    page.theme_mode = ThemeMode.LIGHT
    page.theme = Theme(color_scheme_seed=colors.BLUE)
    
    # 设置窗口大小
    page.window_width = 1200
    page.window_height = 800
    page.padding = 0
    page.scroll = None
    page.bgcolor = flet.colors.WHITE
    
    # 视图映射
    views = {
        "/": HomeView(page),
        "/app/object_detection": ObjectDetectionView(page),
        "/workflow": WorkflowView(page),
        "/about": AboutUsView(page),  # 添加关于页面
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
    flet.app(target=main, assets_dir=assets_dir)