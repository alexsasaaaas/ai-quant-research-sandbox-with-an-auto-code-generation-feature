"""
dependency_mgr.py
Utility for the Agent to safely install missing Python packages.
"""

import sys
import subprocess
import logging
import importlib.util

logger = logging.getLogger(__name__)

def install_package(package_name: str) -> bool:
    """
     versucht, ein Paket über pip zu installieren.
    """
    logger.info(f"Attempting to install missing package: {package_name}")
    try:
        # 由於是在 Streamlit 執行環境中，我們使用目前的 Python 解釋器
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logger.info(f"Successfully installed {package_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to install {package_name}: {e}")
        return False

def check_package(package_name: str) -> bool:
    """檢查包是否已安裝"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None
