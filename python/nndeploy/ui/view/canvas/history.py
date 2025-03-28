"""
操作历史记录模块

负责:
- 记录用户操作历史
- 支持撤销和重做功能
- 管理操作状态
- 提供历史记录查看

以时间线形式展示历史记录,当前状态明显标识
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

@dataclass
class HistoryAction:
    """历史操作记录"""
    type: str                  # 操作类型
    data: Dict[str, Any]      # 操作数据
    timestamp: datetime       # 操作时间
    description: str          # 操作描述

class CanvasHistory:
    """画布历史记录管理器"""
    
    def __init__(
        self,
        max_history: int = 50,
        on_state_change: Optional[Callable[[], None]] = None
    ):
        self._history: List[HistoryAction] = []
        self._current: int = -1
        self._max_history = max_history
        self.on_state_change = on_state_change
        
    @property
    def can_undo(self) -> bool:
        """是否可以撤销"""
        return self._current >= 0
        
    @property
    def can_redo(self) -> bool:
        """是否可以重做"""
        return self._current < len(self._history) - 1
        
    def push(
        self,
        action_type: str,
        data: Dict[str, Any],
        description: str
    ):
        """添加历史记录
        
        Args:
            action_type: 操作类型
            data: 操作数据
            description: 操作描述
        """
        # 删除当前位置之后的记录
        if self._current < len(self._history) - 1:
            self._history = self._history[:self._current + 1]
            
        # 添加新记录
        action = HistoryAction(
            type=action_type,
            data=data,
            timestamp=datetime.now(),
            description=description
        )
        self._history.append(action)
        
        # 限制历史记录数量
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
            
        self._current = len(self._history) - 1
        
        # 通知状态变化
        if self.on_state_change:
            self.on_state_change()
            
    def undo(self) -> Optional[HistoryAction]:
        """撤销操作
        
        Returns:
            撤销的操作记录
        """
        if not self.can_undo:
            return None
            
        action = self._history[self._current]
        self._current -= 1
        
        # 通知状态变化
        if self.on_state_change:
            self.on_state_change()
            
        return action
        
    def redo(self) -> Optional[HistoryAction]:
        """重做操作
        
        Returns:
            重做的操作记录
        """
        if not self.can_redo:
            return None
            
        self._current += 1
        action = self._history[self._current]
        
        # 通知状态变化
        if self.on_state_change:
            self.on_state_change()
            
        return action
        
    def clear(self):
        """清空历史记录"""
        self._history.clear()
        self._current = -1
        
        # 通知状态变化
        if self.on_state_change:
            self.on_state_change()
            
    def get_history(self) -> List[HistoryAction]:
        """获取所有历史记录"""
        return self._history.copy()
        
    def get_current_index(self) -> int:
        """获取当前位置"""
        return self._current 