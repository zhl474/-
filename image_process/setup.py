from setuptools import find_packages, setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=find_packages(),
)
setup(**d)
