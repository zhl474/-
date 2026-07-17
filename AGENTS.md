# Project instructions

本项目必须使用指定的 Python venv 虚拟环境运行代码。

虚拟环境路径：

/home/zhl/fr3env/fr3env/bin

Python 解释器路径：

/home/zhl/fr3env/fr3env/bin/python

## 运行规则

运行 Python 代码时用：/home/zhl/fr3env/fr3env/bin/python 文件名.py

## 安装依赖

安装 Python 包时，必须使用：

/home/zhl/fr3env/fr3env/bin/python -m pip install 包名

## 运行测试

如果需要运行 pytest，使用：

/home/zhl/fr3env/fr3env/bin/python -m pytest

## 检查当前解释器

如果需要确认当前环境，使用：

/home/zhl/fr3env/fr3env/bin/python -c "import sys; print(sys.executable)"

正确输出应包含：

/home/zhl/fr3env/fr3env/bin/python

## 主要工作目录

代码问题只需要看src一个目录。.git跟踪也在src目录下
项目中有大量你不需要看的旧版备份和前期测试例如- runs/
- logs/
- datasets/
- dataset/
- weights/
- build/
- devel/
- install/
- .catkin_tools/
- __pycache__/
- .pytest_cache/
- .git/
- *.bag
- *.log
- *.mp4
- *.avi
- *.png
- *.jpg
- *.jpeg。不要检查src以外的内容除非特别要求

## 代码编写
如果写的是测试类或者标定类的代码，我希望可以python直接运行，传参放在代码开头，通过改代码传参，命令行传参太麻烦