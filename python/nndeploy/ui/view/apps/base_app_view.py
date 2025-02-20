from flet import View, Container, Text, Column
import flet
from ..base_view import BaseView

class BaseAppView(BaseView):
    """应用页面的基类，提供通用的页面布局和功能"""
    
    def __init__(self, page: flet.Page):
        super().__init__(page)
        self.title = "应用"  # 子类需要覆盖此属性
        
    def build(self) -> View:
        return View(
            route=self.get_route(),
            controls=[
                self._get_app_bar(),
                self._get_base_content()
            ]
        )
    
    def get_route(self) -> str:
        """获取路由地址，子类需要覆盖此方法"""
        return "/app/base"
    
    def _get_base_content(self):
        """基础内容布局"""
        return Container(
            content=Column(
                controls=[
                    Container(
                        content=Text(self.title, size=30, weight=flet.FontWeight.BOLD),
                        margin=10,
                    ),
                    self._get_toolbar(),  # 工具栏
                    self._get_main_content(),  # 主要内容区域
                ],
                horizontal_alignment=flet.CrossAxisAlignment.CENTER,
            ),
            padding=20,
        )
    
    def _get_toolbar(self):
        """工具栏区域，子类可以覆盖此方法"""
        return Container()
    
    def _get_main_content(self):
        """主要内容区域，子类需要覆盖此方法"""
        return Container()