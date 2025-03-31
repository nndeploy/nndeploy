"""
日志工具模块

负责:
- 提供多级别日志记录
- 管理日志输出格式
- 支持日志文件轮转
- 处理日志过滤分类

日志在UI中以不同颜色区分级别
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
import sys
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    COLORS = {
        'DEBUG': '\033[94m',    # 蓝色
        'INFO': '\033[92m',     # 绿色
        'WARNING': '\033[93m',  # 黄色
        'ERROR': '\033[91m',    # 红色
        'CRITICAL': '\033[95m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        # 如果是控制台输出且支持颜色
        if sys.stderr.isatty():
            record.levelname = (
                f"{self.COLORS.get(record.levelname, '')}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)

class Logger:
    """日志管理类"""
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        log_dir: str = "logs",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 创建日志目录
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 日志格式
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / f"{name}.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 错误日志处理器
        error_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / f"{name}_error.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        self.logger.addHandler(error_handler)
        
    def set_level(self, level: str):
        """设置日志级别"""
        self.logger.setLevel(level)
        
    def debug(self, msg: str, *args, **kwargs):
        """记录调试日志"""
        self.logger.debug(msg, *args, **kwargs)
        
    def info(self, msg: str, *args, **kwargs):
        """记录信息日志"""
        self.logger.info(msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs):
        """记录警告日志"""
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs):
        """记录错误日志"""
        self.logger.error(msg, *args, **kwargs)
        
    def critical(self, msg: str, *args, **kwargs):
        """记录严重错误日志"""
        self.logger.critical(msg, *args, **kwargs)
        
    def exception(self, msg: str, *args, **kwargs):
        """记录异常日志"""
        self.logger.exception(msg, *args, **kwargs)

# 创建全局日志实例
logger = Logger("nndeploy")

# 便捷函数
def debug(msg: str, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)

def info(msg: str, *args, **kwargs):
    logger.info(msg, *args, **kwargs)

def warning(msg: str, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)

def error(msg: str, *args, **kwargs):
    logger.error(msg, *args, **kwargs)

def critical(msg: str, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)

def exception(msg: str, *args, **kwargs):
    logger.exception(msg, *args, **kwargs) 

# 测试代码
if __name__ == "__main__":
    # 创建测试日志器
    test_logger = Logger(
        name="test_logger",
        level="DEBUG",
        log_dir="test_logs"
    )
    
    # 测试各级别日志
    test_logger.debug("这是一条调试日志")
    test_logger.info("这是一条信息日志")
    test_logger.warning("这是一条警告日志")
    test_logger.error("这是一条错误日志")
    test_logger.critical("这是一条严重错误日志")
    
    # 测试异常日志
    try:
        1 / 0
    except Exception as e:
        test_logger.exception(f"捕获到异常: {e}")
    
    # 测试日志级别切换
    print("\n切换日志级别为WARNING")
    test_logger.set_level("WARNING")
    test_logger.debug("这条调试日志不会显示")
    test_logger.info("这条信息日志不会显示")
    test_logger.warning("这条警告日志会显示")
    
    # 测试全局日志实例
    print("\n测试全局日志实例")
    debug("全局调试日志")
    info("全局信息日志")
    warning("全局警告日志")
    error("全局错误日志")
    
    # 清理测试日志文件
    import shutil
    try:
        shutil.rmtree("test_logs")
        print("\n测试完成，日志文件已清理")
    except:
        print("\n测试完成，但日志文件清理失败")