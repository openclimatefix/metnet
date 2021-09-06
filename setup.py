from distutils.core import setup
with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="metnet",
    version="0.0.2",
    packages=["metnet"],
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
        "remote-sensing"
    ],
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
