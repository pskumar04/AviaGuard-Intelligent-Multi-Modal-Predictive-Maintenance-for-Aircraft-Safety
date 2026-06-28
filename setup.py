"""
Setup script for Aircraft Predictive Maintenance System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

with open("website/requirements-website.txt", "r", encoding="utf-8") as fh:
    website_requirements = fh.read().splitlines()

setup(
    name="aircraft-predictive-maintenance",
    version="1.0.0",
    author="Aircraft PM Team",
    author_email="contact@aircraft-pm.com",
    description="Multi-modal AI system for aircraft predictive maintenance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aircraft-predictive-maintenance",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Aviation Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements + website_requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aircraft-pm-website=website.app:main",
            "aircraft-pm-train=scripts.train_model:main",
            "aircraft-pm-predict=scripts.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "data/*.csv",
            "data/*.json",
            "models/*.h5",
            "models/*.pkl",
            "aircraft_configs/*.json",
            "website/templates/*.html",
            "website/static/css/*.css",
            "website/static/js/*.js",
            "website/static/images/*.jpg",
            "website/static/images/*.png",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/aircraft-predictive-maintenance/issues",
        "Source": "https://github.com/yourusername/aircraft-predictive-maintenance",
        "Documentation": "https://aircraft-pm.readthedocs.io/",
    },
)