"""
用户反馈服务模块

负责:
- 收集用户反馈信息
- 处理问题报告提交
- 管理功能建议收集
- 提供满意度评价

反馈表单简洁清晰,提交过程流畅
"""

from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path
import uuid

from ..utils.logger import logger

class FeedbackType:
    """反馈类型"""
    BUG = "bug"           # 错误报告
    FEATURE = "feature"   # 功能建议
    QUESTION = "question" # 使用问题
    OTHER = "other"       # 其他

class FeedbackPriority:
    """反馈优先级"""
    LOW = "low"       # 低
    MEDIUM = "medium" # 中
    HIGH = "high"     # 高
    URGENT = "urgent" # 紧急

class FeedbackStatus:
    """反馈状态"""
    NEW = "new"           # 新建
    PROCESSING = "processing" # 处理中
    RESOLVED = "resolved"    # 已解决
    CLOSED = "closed"        # 已关闭

class Feedback:
    """反馈信息类"""
    
    def __init__(
        self,
        title: str,
        content: str,
        type: str = FeedbackType.OTHER,
        priority: str = FeedbackPriority.MEDIUM,
        contact: str = None,
        attachments: List[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.title = title
        self.content = content
        self.type = type
        self.priority = priority
        self.status = FeedbackStatus.NEW
        self.contact = contact
        self.attachments = attachments or []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.comments: List[Dict] = []
        
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "type": self.type,
            "priority": self.priority,
            "status": self.status,
            "contact": self.contact,
            "attachments": self.attachments,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "comments": self.comments
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'Feedback':
        """从字典创建实例"""
        feedback = cls(
            title=data["title"],
            content=data["content"],
            type=data["type"],
            priority=data["priority"],
            contact=data.get("contact"),
            attachments=data.get("attachments", [])
        )
        feedback.id = data["id"]
        feedback.status = data["status"]
        feedback.created_at = data["created_at"]
        feedback.updated_at = data["updated_at"]
        feedback.comments = data.get("comments", [])
        return feedback

class FeedbackService:
    """反馈服务类"""
    
    def __init__(self):
        self._feedback_dir = Path("feedback")
        self._feedback_dir.mkdir(exist_ok=True)
        self._feedback_file = self._feedback_dir / "feedback.json"
        self._feedback: Dict[str, Feedback] = {}
        self._load_feedback()
        
    def _load_feedback(self):
        """加载反馈数据"""
        if self._feedback_file.exists():
            try:
                with open(self._feedback_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        feedback = Feedback.from_dict(item)
                        self._feedback[feedback.id] = feedback
            except Exception as e:
                logger.error(f"加载反馈数据失败: {e}")
                
    def _save_feedback(self):
        """保存反馈数据"""
        try:
            data = [
                feedback.to_dict()
                for feedback in self._feedback.values()
            ]
            with open(self._feedback_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存反馈数据失败: {e}")
            
    def submit_feedback(
        self,
        title: str,
        content: str,
        type: str = FeedbackType.OTHER,
        priority: str = FeedbackPriority.MEDIUM,
        contact: str = None,
        attachments: List[str] = None
    ) -> Feedback:
        """提交反馈
        
        Args:
            title: 标题
            content: 内容
            type: 反馈类型
            priority: 优先级
            contact: 联系方式
            attachments: 附件列表
            
        Returns:
            反馈对象
        """
        feedback = Feedback(
            title=title,
            content=content,
            type=type,
            priority=priority,
            contact=contact,
            attachments=attachments
        )
        
        self._feedback[feedback.id] = feedback
        self._save_feedback()
        
        return feedback
        
    def get_feedback(self, feedback_id: str) -> Optional[Feedback]:
        """获取反馈"""
        return self._feedback.get(feedback_id)
        
    def update_feedback(
        self,
        feedback_id: str,
        status: str = None,
        priority: str = None
    ) -> Optional[Feedback]:
        """更新反馈状态
        
        Args:
            feedback_id: 反馈ID
            status: 新状态
            priority: 新优先级
            
        Returns:
            更新后的反馈对象
        """
        feedback = self._feedback.get(feedback_id)
        if feedback:
            if status:
                feedback.status = status
            if priority:
                feedback.priority = priority
            feedback.updated_at = datetime.now().isoformat()
            self._save_feedback()
        return feedback
        
    def add_comment(
        self,
        feedback_id: str,
        content: str,
        author: str
    ) -> Optional[Feedback]:
        """添加评论
        
        Args:
            feedback_id: 反馈ID
            content: 评论内容
            author: 评论作者
            
        Returns:
            更新后的反馈对象
        """
        feedback = self._feedback.get(feedback_id)
        if feedback:
            feedback.comments.append({
                "content": content,
                "author": author,
                "created_at": datetime.now().isoformat()
            })
            feedback.updated_at = datetime.now().isoformat()
            self._save_feedback()
        return feedback
        
    def get_feedback_by_type(self, type: str) -> List[Feedback]:
        """获取指定类型的反馈"""
        return [
            feedback for feedback in self._feedback.values()
            if feedback.type == type
        ]
        
    def get_feedback_by_status(self, status: str) -> List[Feedback]:
        """获取指定状态的反馈"""
        return [
            feedback for feedback in self._feedback.values()
            if feedback.status == status
        ]
        
    def get_feedback_by_priority(self, priority: str) -> List[Feedback]:
        """获取指定优先级的反馈"""
        return [
            feedback for feedback in self._feedback.values()
            if feedback.priority == priority
        ]

# 创建全局反馈服务实例
feedback_service = FeedbackService() 