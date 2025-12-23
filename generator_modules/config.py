"""
Config Module - Configuration
从 data_generator.py 提取
"""

from openai import OpenAI
import sys
import os

# 导入配置
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import adult_config as config

# OpenAI Client
client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_API_BASE,
    timeout=config.REQUEST_TIMEOUT
)

MODEL_NAME = config.FIXED_MODEL_NAME

# 判别式模型可用性
DISCRIMINATIVE_AVAILABLE = False
