"""让控制台测试不依赖 catkin 是否已经生成 devel Python 路径。"""

from pathlib import Path
import sys


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
