import os
from pathlib import Path
import matplotlib.pyplot as plt

IMAGE_DIR = Path("images")

def create_image_dir():
    """创建图片存储目录"""
    os.makedirs(IMAGE_DIR, exist_ok=True)

def configure_plt_settings():
    """配置Matplotlib全局设置"""
    plt.rcParams.update({'font.size': 12})
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("⚠️ 中文显示配置失败，将使用默认字体")