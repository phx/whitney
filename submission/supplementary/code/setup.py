"""Setup configuration for fractal field theory package."""

from setuptools import setup, find_packages

setup(
    name="fractal-field-theory",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "sympy>=1.5.0",
        "pytest>=6.0.0",
        "pytest-benchmark>=3.2.0",
    ],
    extras_require={
        "dev": [
            "pytest-cov>=2.10.0",
            "psutil>=5.7.0",
        ]
    }
) 