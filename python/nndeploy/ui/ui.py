
import flet
from flet import View, Page, Theme, ThemeMode, colors
import os

import settings

from view.home_view import HomeView
from view.apps.object_detection_view import ObjectDetectionView
from view.workflow_view import WorkflowView
from view.about_us_view import AboutUsView
import settings
import app

class UI: 
    def __init__(self, graph=None):
        self.assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        self.ui_graph = None
        if graph is not None:
            self.ui_graph = UIGraph(graph)
            self.ui_graph.init()
            
    def show(self):
        def show_ui(page: flet.Page):
            if self.ui_graph is not None:
                page.add(self.ui_graph)
                page.update()
            else:
                app.app(page)
        
        # 启动应用
        flet.app(target=show_ui, assets_dir=self.assets_dir, view=flet.WEB_BROWSER, port=9090)
        
        
def main():
    ui = UI()
    ui.show()

if __name__ == "__main__":
    main()
