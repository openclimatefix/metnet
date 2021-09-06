from setuptools import setup, find_packages

setup(
    name="metnet",
    version="0.0.7",
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
        "remote-sensing"
    ],
    install_requires=["einops~=0.3.0",
                      "numpy~=1.19.5",
                      "torchvision~=0.10.0",
                      "antialiased_cnns",
                      "axial_attention",
                      "pytorch_msssim",
                      "torch"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
