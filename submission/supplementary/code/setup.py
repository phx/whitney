"""Setup configuration for fractal field theory package."""

from setuptools import setup, find_packages
import os

# Read version from version.py
version = {}
with open(os.path.join("core", "version.py")) as f:
    exec(f.read(), version)

# Read long description from README
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fractal-field-theory",
    version=version["__version__"],
    description="A framework for fractal field theory computations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="The Fractal Field Theory Collaboration",
    author_email="contact@fractalfield.org",
    url="https://github.com/fractalfield/fractal-field-theory",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "sympy>=1.8",
        "mpmath>=1.2.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "hypothesis>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "nbsphinx>=0.8",
        ],
        "ml": [
            "scikit-learn>=0.24",
            "tensorflow>=2.6",
            "torch>=1.9",
        ],
        "parallel": [
            "dask>=2021.8.0",
            "ray>=1.5",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
    ],
) 