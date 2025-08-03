"""
Setup script for Stock Analysis AI Agent.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stock-analysis-ai-agent",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An intelligent AI agent for stock analysis using MCP servers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stock-analysis-ai-agent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "isort>=5.10.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stock-analysis-agent=stock_analysis_agent.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "stock_analysis_agent": [
            "templates/*.md",
            "templates/*.html",
            "config/*.yaml",
            "config/*.json",
        ],
    },
    keywords="stock analysis, ai, mcp, financial analysis, trading, investment",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/stock-analysis-ai-agent/issues",
        "Source": "https://github.com/yourusername/stock-analysis-ai-agent",
        "Documentation": "https://stock-analysis-ai-agent.readthedocs.io/",
    },
) 