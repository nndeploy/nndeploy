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
from typing import Optional, Callable, BinaryIO, List, Union
import aiofiles
import hashlib
import mimetypes
from datetime import datetime
import tempfile
import zipfile
import glob

class FileUtils:
    """文件工具类"""
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """获取文件大小(字节)"""
        try:
            return os.path.getsize(file_path)
        except (FileNotFoundError, PermissionError) as e:
            print(f"Failed to get file size: {e}")
            raise
        
    @staticmethod
    def get_file_type(file_path: str) -> str:
        """获取文件MIME类型"""
        return mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        
    @staticmethod
    def get_file_hash(file_path: str, algorithm: str = "md5") -> str:
        """计算文件哈希值"""
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except (FileNotFoundError, PermissionError) as e:
            print(f"Failed to calculate file hash: {e}")
            raise
        
    @staticmethod
    def get_file_info(file_path: str) -> dict:
        """获取文件信息"""
        try:
            stat = os.stat(file_path)
            return {
                "name": os.path.basename(file_path),
                "path": file_path,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "type": FileUtils.get_file_type(file_path),
                "hash_md5": FileUtils.get_file_hash(file_path, "md5")
            }
        except (FileNotFoundError, PermissionError) as e:
            print(f"Failed to get file info: {e}")
            raise
        
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
        try:
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
        except Exception as e:
            print(f"Failed to read file asynchronously: {e}")
            raise
        
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
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            total_size = len(content)
            bytes_written = 0
            
            async with aiofiles.open(file_path, "wb") as f:
                for i in range(0, total_size, chunk_size):
                    chunk = content[i:i + chunk_size]
                    await f.write(chunk)
                    bytes_written += len(chunk)
                    if progress_callback:
                        progress_callback(bytes_written, total_size)
        except Exception as e:
            print(f"Failed to write file asynchronously: {e}")
            raise
                    
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
        try:
            # 确保目标目录存在
            os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
            
            file_size = os.path.getsize(src_path)
            bytes_copied = 0
            
            async with aiofiles.open(src_path, "rb") as src, \
                      aiofiles.open(dst_path, "wb") as dst:
                while chunk := await src.read(chunk_size):
                    await dst.write(chunk)
                    bytes_copied += len(chunk)
                    if progress_callback:
                        progress_callback(bytes_copied, file_size)
        except Exception as e:
            print(f"Failed to copy file asynchronously: {e}")
            raise
                    
    @staticmethod
    def ensure_dir(dir_path: str) -> str:
        """确保目录存在，并返回创建的目录路径"""
        try:
            os.makedirs(dir_path, exist_ok=True)
            return dir_path
        except Exception as e:
            print(f"Failed to create directory: {e}")
            raise
        
    @staticmethod
    def clean_dir(dir_path: str):
        """清空目录"""
        try:
            if not os.path.exists(dir_path):
                return
                
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        except Exception as e:
            print(f"Failed to clean directory: {e}")
            raise
                
    @staticmethod
    def get_temp_file(suffix: str = None) -> str:
        """获取临时文件路径"""
        try:
            fd, path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            return path
        except Exception as e:
            print(f"Failed to get temporary file: {e}")
            raise
        
    @staticmethod
    def get_temp_dir() -> str:
        """获取临时目录路径"""
        try:
            return tempfile.mkdtemp()
        except Exception as e:
            print(f"Failed to get temporary directory: {e}")
            raise
        
    @staticmethod
    def format_size(size: int) -> str:
        """格式化文件大小"""
        try:
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if size < 1024:
                    return f"{size:.2f} {unit}"
                size /= 1024
            return f"{size:.2f} PB"
        except Exception as e:
            print(f"Failed to format file size: {e}")
            return "Unknown size"
    
    @staticmethod
    def list_files(
        directory: str, 
        pattern: str = "*", 
        recursive: bool = False
    ) -> List[str]:
        """列出目录中的文件
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            recursive: 是否递归查找
            
        Returns:
            文件路径列表
        """
        try:
            if recursive:
                return glob.glob(os.path.join(directory, "**", pattern), recursive=True)
            else:
                return glob.glob(os.path.join(directory, pattern))
        except Exception as e:
            print(f"Failed to list directory files: {e}")
            return []
    
    @staticmethod
    def zip_files(
        files: List[str], 
        output_path: str, 
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """压缩文件
        
        Args:
            files: 要压缩的文件列表
            output_path: 输出zip文件路径
            progress_callback: 进度回调函数(当前索引,总数量)
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            total_files = len(files)
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for i, file in enumerate(files):
                    if os.path.isfile(file):
                        zipf.write(file, os.path.basename(file))
                    if progress_callback:
                        progress_callback(i + 1, total_files)
        except Exception as e:
            print(f"Failed to zip files: {e}")
            raise
    
    @staticmethod
    def unzip_file(
        zip_path: str, 
        extract_dir: str, 
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """解压缩文件
        
        Args:
            zip_path: zip文件路径
            extract_dir: 解压目录
            progress_callback: 进度回调函数(当前索引,总数量)
        """
        try:
            # 确保解压目录存在
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                file_list = zipf.namelist()
                total_files = len(file_list)
                
                for i, file in enumerate(file_list):
                    zipf.extract(file, extract_dir)
                    if progress_callback:
                        progress_callback(i + 1, total_files)
        except Exception as e:
            print(f"Failed to unzip file: {e}")
            raise
    
    @staticmethod
    def move_file(src_path: str, dst_path: str):
        """移动文件
        
        Args:
            src_path: 源文件路径
            dst_path: 目标文件路径
        """
        try:
            # 确保目标目录存在
            os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
            shutil.move(src_path, dst_path)
        except Exception as e:
            print(f"Failed to move file: {e}")
            raise
    
    @staticmethod
    def remove_file(file_path: str):
        """删除文件
        
        Args:
            file_path: 文件路径
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to remove file: {e}")
            raise
    
    @staticmethod
    def remove_dir(dir_path: str):
        """删除目录
        
        Args:
            dir_path: 目录路径
        """
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        except Exception as e:
            print(f"Failed to remove directory: {e}")
            raise

# 创建全局文件工具实例
file_utils = FileUtils() 

# 测试代码
if __name__ == "__main__":
    import asyncio
    
    async def test_async_functions():
        # 测试临时文件和目录
        temp_file = FileUtils.get_temp_file(suffix=".txt")
        temp_dir = FileUtils.get_temp_dir()
        print(f"临时文件: {temp_file}")
        print(f"临时目录: {temp_dir}")
        
        # 测试异步写入
        test_content = b"Hello, NNDeploy!"
        await FileUtils.write_file_async(
            temp_file, 
            test_content,
            progress_callback=lambda current, total: print(f"写入进度: {current}/{total}")
        )
        
        # 测试文件信息
        file_info = FileUtils.get_file_info(temp_file)
        print(f"文件信息: {file_info}")
        print(f"文件大小: {FileUtils.format_size(file_info['size'])}")
        
        # 测试异步读取
        content = await FileUtils.read_file_async(
            temp_file,
            progress_callback=lambda current, total: print(f"读取进度: {current}/{total}")
        )
        print(f"读取内容: {content.decode()}")
        
        # 测试异步复制
        copy_file = os.path.join(temp_dir, "copy.txt")
        await FileUtils.copy_file_async(
            temp_file,
            copy_file,
            progress_callback=lambda current, total: print(f"复制进度: {current}/{total}")
        )
        
        # 测试文件哈希
        original_hash = FileUtils.get_file_hash(temp_file)
        copy_hash = FileUtils.get_file_hash(copy_file)
        print(f"原始文件哈希: {original_hash}")
        print(f"复制文件哈希: {copy_hash}")
        print(f"哈希匹配: {original_hash == copy_hash}")
        
        # 测试压缩和解压
        zip_file = os.path.join(temp_dir, "test.zip")
        FileUtils.zip_files(
            [temp_file, copy_file],
            zip_file,
            progress_callback=lambda current, total: print(f"压缩进度: {current}/{total}")
        )
        
        extract_dir = os.path.join(temp_dir, "extracted")
        FileUtils.unzip_file(
            zip_file,
            extract_dir,
            progress_callback=lambda current, total: print(f"解压进度: {current}/{total}")
        )
        
        # 列出解压文件
        extracted_files = FileUtils.list_files(extract_dir)
        print(f"解压文件: {extracted_files}")
        
        # 清理
        FileUtils.remove_file(temp_file)
        FileUtils.remove_dir(temp_dir)
        print("测试完成，临时文件已清理")
    
    # 运行测试
    asyncio.run(test_async_functions())