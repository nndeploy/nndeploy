"""
选择工具模块

负责:
- 实现节点和连线的选择功能
- 处理单选和多选操作
- 管理选择状态
- 提供选择框绘制

选中元素周围显示高亮边框,多选时显示选择框
"""

from typing import Set, Optional, Tuple, Callable
import flet as ft

class SelectionTool:
    """选择工具"""
    
    def __init__(
        self,
        on_selection_change: Optional[Callable[[Set[str]], None]] = None
    ):
        self.on_selection_change = on_selection_change
        self._selected_ids: Set[str] = set()
        self._selecting = False
        self._selection_start: Optional[Tuple[float, float]] = None
        self._selection_rect: Optional[Tuple[float, float, float, float]] = None
        
    @property
    def selected_ids(self) -> Set[str]:
        """已选择的元素ID集合"""
        return self._selected_ids.copy()
        
    def clear_selection(self):
        """清除选择"""
        if self._selected_ids:
            self._selected_ids.clear()
            if self.on_selection_change:
                self.on_selection_change(self._selected_ids)
                
    def select_elements(self, ids: Set[str], append: bool = False):
        """选择元素
        
        Args:
            ids: 要选择的元素ID集合
            append: 是否追加到已选择的元素
        """
        if not append:
            self._selected_ids.clear()
        self._selected_ids.update(ids)
        
        if self.on_selection_change:
            self.on_selection_change(self._selected_ids)
            
    def deselect_elements(self, ids: Set[str]):
        """取消选择元素
        
        Args:
            ids: 要取消选择的元素ID集合
        """
        self._selected_ids.difference_update(ids)
        
        if self.on_selection_change:
            self.on_selection_change(self._selected_ids)
            
    def on_pointer_down(self, e: ft.PointerEvent) -> bool:
        """处理指针按下事件
        
        Returns:
            是否处理了事件
        """
        # 开始框选
        if e.buttons == ft.MouseButton.LEFT:
            self._selecting = True
            self._selection_start = (e.local_x, e.local_y)
            self._selection_rect = None
            
            # 点击空白处清除选择
            if not e.ctrl:
                self.clear_selection()
                
            return True
            
        return False
        
    def on_pointer_move(self, e: ft.PointerEvent) -> bool:
        """处理指针移动事件"""
        if not self._selecting or not self._selection_start:
            return False
            
        # 更新选择框
        x = min(self._selection_start[0], e.local_x)
        y = min(self._selection_start[1], e.local_y)
        w = abs(e.local_x - self._selection_start[0])
        h = abs(e.local_y - self._selection_start[1])
        
        self._selection_rect = (x, y, w, h)
        return True
        
    def on_pointer_up(self, e: ft.PointerEvent) -> bool:
        """处理指针抬起事件"""
        if not self._selecting:
            return False
            
        self._selecting = False
        self._selection_start = None
        self._selection_rect = None
        return True
        
    def get_selection_rect(self) -> Optional[Tuple[float, float, float, float]]:
        """获取选择框矩形"""
        return self._selection_rect
        
    def is_selected(self, element_id: str) -> bool:
        """检查元素是否被选中"""
        return element_id in self._selected_ids 