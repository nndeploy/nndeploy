from flet import View, Page
from .base_view import BaseView

class AppView(BaseView):
    def __init__(self, page: Page, title: str, description: str):
        super().__init__(page)
        self.title = title
        self.description = description
        
    def build(self) -> View:
        return View(
            route=f"/app/{self.title}",
            controls=[
                self._get_app_bar(),
                self._get_content()
            ]
        )
    
    def _get_app_bar(self):
        # 复用HomeView的AppBar代码
        pass
        
    def _get_content(self):
        # 子类实现具体内容
        raise NotImplementedError("Subclass must implement _get_content()")