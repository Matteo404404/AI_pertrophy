"""
Setup configuration for Scientific Hypertrophy Trainer

Installs the package with all dependencies and command-line tools.

Installation:
    pip install -e .

Usage after install:
    hypertrophy-trainer --demo
    validate-training-data data.csv

Author: Scientific Hypertrophy Trainer Team
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Evidence-based hypertrophy training predictions and recommendations"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'python-dateutil>=2.8.0',
    ]

setup(
    name="scientific-hypertrophy-trainer",
    version="1.0.0",
    description="Evidence-based hypertrophy training predictions and recommendations",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Scientific Hypertrophy Trainer Team",
    author_email="contact@hypertrophytrainer.com",
    url="https://github.com/your-username/scientific-hypertrophy-trainer",
    
    # Package configuration
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    
    # Dependencies
    install_requires=read_requirements(),
    python_requires='>=3.7',
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.800',
        ],
        'viz': [
            'matplotlib>=3.3.0',
            'seaborn>=0.11.0',
            'plotly>=4.14.0',
        ],
        'web': [
            'streamlit>=1.0.0',
            'flask>=2.0.0',
        ],
    },
    
    # Command-line tools
    entry_points={
        'console_scripts': [
            'hypertrophy-trainer=trainer:main',
            'validate-training-data=validate_data:main',
        ],
    },
    
    # Metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    
    keywords="hypertrophy training machine-learning fitness prediction",
    
    # Package data
    package_data={
        'ml': ['data/*.yaml', 'models/*.yaml'],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/your-username/scientific-hypertrophy-trainer/issues",
        "Source": "https://github.com/your-username/scientific-hypertrophy-trainer",
        "Documentation": "https://scientific-hypertrophy-trainer.readthedocs.io/",
    },
)
