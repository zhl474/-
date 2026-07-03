from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['image_process_lib'],  # 你的代码文件夹名字
    # package_dir={'': 'src'}      # 告诉 setuptools，包代码在 src 目录下
)
setup(**d)