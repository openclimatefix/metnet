import os
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
install_requires = (this_directory / "requirements.txt").read_text().splitlines()
long_description = (this_directory / "README.md").read_text()

setup(
    name="metnet",
    version="4.1.6",
    packages=find_packages(),
    url="https://github.com/openclimatefix/metnet",
    license="MIT License",
    company="Open Climate Fix Ltd",
    author="Jacob Bieker",
    author_email="jacob@openclimatefix.org",
    description="PyTorch MetNet Implementation",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "transformer",
        "attention mechanism",
        "metnet",
        "forecasting",
        "remote-sensing",
    ],
    install_requires=install_requires,
    long_description=long_description,
    extras_require={"train": ["ocf_datapipes", "pytorch-lightning"]},
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
