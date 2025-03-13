"""
文件工具模块

负责:
- 提供文件操作接口
- 处理文件读写异步
- 管理文件操作进度
- 处理文件操作异常

文件操作显示进度,大文件有详细提示
"""

import os
import shutil
import asyncio
from pathlib import Path
from typing import Optional, Callable, BinaryIO
import aiofiles
import hashlib
import mimetypes
from datetime import datetime

class FileUtils:
    """文件工具类"""
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """获取文件大小(字节)"""
        return os.path.getsize(file_path)
        
    @staticmethod
    def get_file_type(file_path: str) -> str:
        """获取文件MIME类型"""
        return mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        
    @staticmethod
    def get_file_hash(file_path: str, algorithm: str = "md5") -> str:
        """计算文件哈希值"""
        hash_obj = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
        
    @staticmethod
    def get_file_info(file_path: str) -> dict:
        """获取文件信息"""
        stat = os.stat(file_path)
        return {
            "name": os.path.basename(file_path),
            "path": file_path,
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "type": FileUtils.get_file_type(file_path)
        }
        
    @staticmethod
    async def read_file_async(
        file_path: str,
        chunk_size: int = 8192,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bytes:
        """异步读取文件
        
        Args:
            file_path: 文件路径
            chunk_size: 分块大小
            progress_callback: 进度回调函数(当前大小,总大小)
            
        Returns:
            文件内容
        """
        file_size = os.path.getsize(file_path)
        content = bytearray()
        
        async with aiofiles.open(file_path, "rb") as f:
            bytes_read = 0
            while chunk := await f.read(chunk_size):
                content.extend(chunk)
                bytes_read += len(chunk)
                if progress_callback:
                    progress_callback(bytes_read, file_size)
                    
        return bytes(content)
        
    @staticmethod
    async def write_file_async(
        file_path: str,
        content: bytes,
        chunk_size: int = 8192,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """异步写入文件
        
        Args:
            file_path: 文件路径
            content: 文件内容
            chunk_size: 分块大小
            progress_callback: 进度回调函数(当前大小,总大小)
        """
        total_size = len(content)
        bytes_written = 0
        
        async with aiofiles.open(file_path, "wb") as f:
            for i in range(0, total_size, chunk_size):
                chunk = content[i:i + chunk_size]
                await f.write(chunk)
                bytes_written += len(chunk)
                if progress_callback:
                    progress_callback(bytes_written, total_size)
                    
    @staticmethod
    async def copy_file_async(
        src_path: str,
        dst_path: str,
        chunk_size: int = 8192,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """异步复制文件
        
        Args:
            src_path: 源文件路径
            dst_path: 目标文件路径
            chunk_size: 分块大小
            progress_callback: 进度回调函数(当前大小,总大小)
        """
        file_size = os.path.getsize(src_path)
        bytes_copied = 0
        
        async with aiofiles.open(src_path, "rb") as src, \
                  aiofiles.open(dst_path, "wb") as dst:
            while chunk := await src.read(chunk_size):
                await dst.write(chunk)
                bytes_copied += len(chunk)
                if progress_callback:
                    progress_callback(bytes_copied, file_size)
                    
    @staticmethod
    def ensure_dir(dir_path: str):
        """确保目录存在"""
        os.makedirs(dir_path, exist_ok=True)
        
    @staticmethod
    def clean_dir(dir_path: str):
        """清空目录"""
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                
    @staticmethod
    def get_temp_file(suffix: str = None) -> str:
        """获取临时文件路径"""
        import tempfile
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        return path
        
    @staticmethod
    def format_size(size: int) -> str:
        """格式化文件大小"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"

# 创建全局文件工具实例
file_utils = FileUtils() 