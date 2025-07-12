from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fast-causal-shap",
    version="0.1.0",
    author="woonyee28",
    author_email="ngnwy289@gmail.com",
    description="A Python package for efficient causal SHAP computations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/woonyee28/CausalSHAP",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.0.0",
        "networkx>=2.0",
        "numpy>=1.18.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    keywords="causal inference, machine learning, explainable ai, shap, causal discovery",
)