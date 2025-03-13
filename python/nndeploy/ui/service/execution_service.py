"""
执行服务模块

负责:
- 处理工作流的执行调度
- 管理节点的执行顺序
- 处理并行执行任务
- 提供执行状态监控

执行过程可视化,关键节点状态醒目显示
"""

import asyncio
from typing import Dict, List, Optional, Callable
from datetime import datetime
import threading
import queue
from enum import Enum

from ..utils.logger import logger
from ..entity.workflow_repository import WorkflowRepository, Workflow
from ..entity.node_repository import NodeRepository

class ExecutionStatus(Enum):
    """执行状态"""
    READY = "ready"          # 就绪
    RUNNING = "running"      # 运行中
    PAUSED = "paused"       # 已暂停
    COMPLETED = "completed" # 已完成
    FAILED = "failed"       # 失败
    CANCELED = "canceled"   # 已取消

class NodeExecutionStatus:
    """节点执行状态"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.status = ExecutionStatus.READY
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error: Optional[Exception] = None
        self.result: Optional[Dict] = None
        
    @property
    def duration(self) -> Optional[float]:
        """执行时长(秒)"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

class ExecutionContext:
    """执行上下文"""
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.status = ExecutionStatus.READY
        self.node_status: Dict[str, NodeExecutionStatus] = {}
        self.variables: Dict[str, Any] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self._observers = []
        
    def add_observer(self, observer: Callable):
        """添加状态观察者"""
        if observer not in self._observers:
            self._observers.append(observer)
            
    def notify_observers(self):
        """通知所有观察者"""
        for observer in self._observers:
            observer(self)
            
    @property
    def duration(self) -> Optional[float]:
        """执行时长(秒)"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

class ExecutionService:
    """执行服务类"""
    
    def __init__(self):
        self.workflow_repo = WorkflowRepository()
        self.node_repo = NodeRepository()
        self._contexts: Dict[str, ExecutionContext] = {}
        self._running_workflows = set()
        self._executor_thread: Optional[threading.Thread] = None
        self._task_queue = queue.Queue()
        self._stop_event = threading.Event()
        
    def start(self):
        """启动执行服务"""
        if not self._executor_thread:
            self._stop_event.clear()
            self._executor_thread = threading.Thread(
                target=self._executor_loop,
                daemon=True
            )
            self._executor_thread.start()
            
    def stop(self):
        """停止执行服务"""
        if self._executor_thread:
            self._stop_event.set()
            self._executor_thread.join()
            self._executor_thread = None
            
    def _executor_loop(self):
        """执行器循环"""
        while not self._stop_event.is_set():
            try:
                # 获取任务
                task = self._task_queue.get(timeout=1)
                if task:
                    workflow_id, action = task
                    if action == "start":
                        self._execute_workflow(workflow_id)
                    elif action == "stop":
                        self._stop_workflow(workflow_id)
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"执行器错误: {e}")
                
    def execute_workflow(self, workflow_id: str):
        """执行工作流
        
        Args:
            workflow_id: 工作流ID
        """
        if workflow_id in self._running_workflows:
            raise RuntimeError(f"工作流 {workflow_id} 已在运行")
            
        # 创建执行上下文
        context = ExecutionContext(workflow_id)
        self._contexts[workflow_id] = context
        
        # 添加到任务队列
        self._task_queue.put((workflow_id, "start"))
        
    def stop_workflow(self, workflow_id: str):
        """停止工作流执行
        
        Args:
            workflow_id: 工作流ID
        """
        if workflow_id not in self._running_workflows:
            return
            
        # 添加到任务队列
        self._task_queue.put((workflow_id, "stop"))
        
    def get_execution_status(self, workflow_id: str) -> Optional[ExecutionContext]:
        """获取工作流执行状态"""
        return self._contexts.get(workflow_id)
        
    async def _execute_workflow(self, workflow_id: str):
        """执行工作流
        
        Args:
            workflow_id: 工作流ID
        """
        context = self._contexts[workflow_id]
        workflow = self.workflow_repo.get_workflow(workflow_id)
        
        if not workflow:
            context.status = ExecutionStatus.FAILED
            context.notify_observers()
            return
            
        try:
            # 开始执行
            self._running_workflows.add(workflow_id)
            context.status = ExecutionStatus.RUNNING
            context.start_time = datetime.now()
            context.notify_observers()
            
            # 获取工作流数据
            workflow_data = workflow.get_current_version().data
            
            # 构建执行图
            execution_graph = self._build_execution_graph(workflow_data)
            
            # 执行节点
            for node_id in execution_graph:
                if workflow_id not in self._running_workflows:
                    break
                    
                await self._execute_node(node_id, context)
                
            # 完成执行
            context.status = ExecutionStatus.COMPLETED
            context.end_time = datetime.now()
            
        except Exception as e:
            logger.exception(f"工作流执行失败: {e}")
            context.status = ExecutionStatus.FAILED
            context.end_time = datetime.now()
            
        finally:
            self._running_workflows.discard(workflow_id)
            context.notify_observers()
            
    def _stop_workflow(self, workflow_id: str):
        """停止工作流执行"""
        if workflow_id in self._running_workflows:
            self._running_workflows.discard(workflow_id)
            context = self._contexts[workflow_id]
            context.status = ExecutionStatus.CANCELED
            context.end_time = datetime.now()
            context.notify_observers()
            
    async def _execute_node(self, node_id: str, context: ExecutionContext):
        """执行节点
        
        Args:
            node_id: 节点ID
            context: 执行上下文
        """
        # 创建节点状态
        node_status = NodeExecutionStatus(node_id)
        context.node_status[node_id] = node_status
        
        try:
            # 开始执行
            node_status.status = ExecutionStatus.RUNNING
            node_status.start_time = datetime.now()
            context.notify_observers()
            
            # 获取节点定义
            node_type = self.node_repo.get_node(node_id)
            if not node_type:
                raise ValueError(f"节点 {node_id} 不存在")
                
            # TODO: 执行节点逻辑
            # 这里需要根据节点类型调用相应的处理函数
            
            # 完成执行
            node_status.status = ExecutionStatus.COMPLETED
            node_status.end_time = datetime.now()
            
        except Exception as e:
            logger.exception(f"节点执行失败: {e}")
            node_status.status = ExecutionStatus.FAILED
            node_status.error = e
            node_status.end_time = datetime.now()
            raise
            
        finally:
            context.notify_observers()
            
    def _build_execution_graph(self, workflow_data: Dict) -> List[str]:
        """构建执行图
        
        Args:
            workflow_data: 工作流数据
            
        Returns:
            节点执行顺序列表
        """
        # TODO: 实现拓扑排序,生成节点执行顺序
        # 这里需要分析节点间的依赖关系
        return []

# 创建全局执行服务实例
execution_service = ExecutionService() 