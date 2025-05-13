import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# 设置日志
log_file = 'logs/translation.log'

# 确保日志目录存在
log_dir = os.path.dirname(log_file)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger('translation_logger')
logger.setLevel(logging.INFO)

# 文件处理器
file_handler = RotatingFileHandler(log_file, maxBytes=3 * 1024 * 1024, backupCount=5)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 终端处理器
stream_handler = logging.StreamHandler(sys.stdout)
stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)

load_dotenv(override=True)
TARGET_LANG = os.getenv('TARGET_LANG')
PROMPT = f"""
- Role: 技术文档翻译专家
- Background: 作为资深技术文档翻译专家，专注于网页文本和 AI 领域的翻译工作，需确保翻译内容的专业性和清晰度，符合技术文档标准。
- Profile: 拥有扎实的多语言基础和对技术文档结构的深刻理解，能精准翻译技术术语和专业名词。
- Skills: 能准确理解并翻译技术术语，提供流畅翻译，保持原文专业性，确保翻译真实、准确、连贯。
- Goals: 将技术文档准确、专业地翻译成{TARGET_LANG}，符合改语言读者习惯，保持原文格式和链接。
- Constrains: 保留网页链接和、代码和表格格式，仅输出译文，不破坏原来格式。
- OutputFormat: 使用如下 JSON 格式输出你的结果，仅输出译文。
- Workflow:
  1. 接收由Json对象的英文内容。
  2. 逐条准确翻译。
  3. 审核校对确保流畅连贯。
  4. 使用Json输出翻译结果。
  5. 输出符合要求的最终翻译。
- Initialization: 在第一次对话中，直接输出译文，不附加额外说明。
"""

MODEL = os.getenv('MODEL')
API_KEY = os.getenv('API_KEY')
BATCH_MAX_CHARTS = int(os.getenv('BATCH_MAX_CHARTS'))
MAX_RETRIES = int(os.getenv('MAX_RETRIES'))
FROM_DIR = os.getenv('FROM_DIR')
TO_DIR = os.getenv('TO_DIR')

MODEL_CONFIG_DICT = {
    'kimi': {
        'model': 'moonshot-v1-auto',
        'base_url': 'https://api.moonshot.cn/v1',
        'prompt': PROMPT,
        'temperature': 0.3,
        'rpm': 3,
    },
    'deepseek': {
        'model': 'deepseek-chat',
        'base_url': 'https://api.deepseek.com',
        'prompt': PROMPT,
        'temperature': 1.3,
        'rpm': 60,
    },
    'qwen': {
        'model': 'qwen-plus',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'prompt': PROMPT,
        'temperature': 0.7,
        'rpm': 1200,
    },
    'glm': {
        'model': 'glm-4-plus',
        'base_url': 'https://open.bigmodel.cn/api/paas/v4/',
        'prompt': PROMPT,
        'temperature': 0.95,
        'rpm': 50,
    },
    'openai': {
        'model': 'gpt-4o',
        'base_url': None,
        'prompt': PROMPT,
        'temperature': 0.8,
        'rpm': 30,
    }
}

MODEL_CONFIG = MODEL_CONFIG_DICT.get(MODEL)
