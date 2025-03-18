"""
文件操作服务模块

负责:
- 处理工作流文件的读写
- 管理文件的自动保存
- 提供文件格式转换
- 处理文件的备份恢复

文件操作过程显示进度,结果有清晰提示
"""

import os
import shutil
import json
from typing import Dict, List, Optional, BinaryIO
from datetime import datetime
import asyncio
from pathlib import Path
import tempfile

from ..utils.logger import logger
from ..utils.file_utils import FileUtils

class FileService:
    """文件服务类"""
    
    def __init__(self):
        self._data_dir = Path("data")
        self._backup_dir = self._data_dir / "backups"
        self._temp_dir = self._data_dir / "temp"
        
        # 创建目录
        self._data_dir.mkdir(exist_ok=True)
        self._backup_dir.mkdir(exist_ok=True)
        self._temp_dir.mkdir(exist_ok=True)
        
        # 自动保存配置
        self._auto_save_enabled = True
        self._auto_save_interval = 300  # 5分钟
        self._max_backups = 5
        
        # 启动自动保存任务
        # if self._auto_save_enabled:
        #     asyncio.create_task(self._auto_save_loop())
            
    async def _auto_save_loop(self):
        """自动保存循环"""
        while True:
            try:
                await asyncio.sleep(self._auto_save_interval)
                # TODO: 执行自动保存
            except Exception as e:
                logger.error(f"自动保存失败: {e}")
                
    async def save_workflow(
        self,
        workflow_id: str,
        data: Dict,
        create_backup: bool = True
    ):
        """保存工作流
        
        Args:
            workflow_id: 工作流ID
            data: 工作流数据
            create_backup: 是否创建备份
        """
        file_path = self._data_dir / f"{workflow_id}.json"
        
        # 创建备份
        if create_backup and file_path.exists():
            await self._create_backup(workflow_id)
            
        # 保存文件
        try:
            temp_path = self._temp_dir / f"{workflow_id}.json.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            temp_path.replace(file_path)
        except Exception as e:
            logger.error(f"保存工作流失败: {e}")
            raise
            
    async def load_workflow(self, workflow_id: str) -> Optional[Dict]:
        """加载工作流
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            工作流数据
        """
        file_path = self._data_dir / f"{workflow_id}.json"
        
        if not file_path.exists():
            return None
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载工作流失败: {e}")
            raise
            
    async def _create_backup(self, workflow_id: str):
        """创建工作流备份"""
        source_path = self._data_dir / f"{workflow_id}.json"
        if not source_path.exists():
            return
            
        # 生成备份文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self._backup_dir / f"{workflow_id}_{timestamp}.json"
        
        # 复制文件
        try:
            shutil.copy2(source_path, backup_path)
            
            # 清理旧备份
            self._cleanup_backups(workflow_id)
        except Exception as e:
            logger.error(f"创建备份失败: {e}")
            
    def _cleanup_backups(self, workflow_id: str):
        """清理旧备份"""
        # 获取所有备份
        backups = sorted(
            self._backup_dir.glob(f"{workflow_id}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # 删除多余的备份
        for backup in backups[self._max_backups:]:
            try:
                backup.unlink()
            except Exception as e:
                logger.error(f"删除备份失败: {e}")
                
    def get_backups(self, workflow_id: str) -> List[Dict]:
        """获取工作流的所有备份
        
        Returns:
            备份信息列表
        """
        backups = []
        for path in self._backup_dir.glob(f"{workflow_id}_*.json"):
            try:
                stat = path.stat()
                backups.append({
                    "path": str(path),
                    "size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_mtime)
                })
            except Exception as e:
                logger.error(f"获取备份信息失败: {e}")
                
        return sorted(
            backups,
            key=lambda b: b["created_at"],
            reverse=True
        )
        
    async def restore_backup(self, backup_path: str) -> Optional[Dict]:
        """恢复备份
        
        Args:
            backup_path: 备份文件路径
            
        Returns:
            恢复的工作流数据
        """
        if not os.path.exists(backup_path):
            return None
            
        try:
            with open(backup_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"恢复备份失败: {e}")
            raise
            
    def set_auto_save(self, enabled: bool, interval: int = None):
        """设置自动保存
        
        Args:
            enabled: 是否启用
            interval: 保存间隔(秒)
        """
        self._auto_save_enabled = enabled
        if interval:
            self._auto_save_interval = interval
            
    def set_max_backups(self, count: int):
        """设置最大备份数"""
        self._max_backups = count
        
    def cleanup(self):
        """清理临时文件"""
        try:
            shutil.rmtree(self._temp_dir)
            self._temp_dir.mkdir()
        except Exception as e:
            logger.error(f"清理临时文件失败: {e}")

# 创建全局文件服务实例
file_service = FileService() 