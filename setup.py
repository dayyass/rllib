from setuptools import setup

from rllib import __version__

with open("README.md", mode="r", encoding="utf-8") as fp:
    long_description = fp.read()


setup(
    name="pytorch-rllib",
    version=__version__,
    description="Reinforcement Learning Library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Dani El-Ayyass",
    author_email="dayyass@yandex.ru",
    license_files=["LICENSE"],
    url="https://github.com/dayyass/rllib",
    packages=["rllib"],
    install_requires=[
        "gym==0.24.1",
        "numpy==1.21.6",
        "tqdm==4.64.0",
    ],
)
