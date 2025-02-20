from setuptools import setup, find_packages

name = "linalgx"
version = "0.1.0"
description = "A high-performance library for fast linear algebra computations on 2D and 3D geometric shapes."
author = "Your Name"

def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name=name,
    version=version,
    description=description,
    long_description=open("README.md").read(),
    long_description_content_type="",
    author=author,
    author_email="uditsangule@gmail.com",
    url="https://github.com/uditsangule/linalgx.git",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    python_requires=">=3.10",
)