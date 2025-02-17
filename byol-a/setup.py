from setuptools import setup, find_packages

setup(
    name="byol_a",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "torchaudio>=0.7.0",
        "numpy>=1.19.2",
    ],
)
