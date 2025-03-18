#!/usr/bin/env python3
"""
PDF to Markdown Converter - Setup Script

This script installs the PDF to Markdown converter and its dependencies.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pdf2md",
    version="1.1.0",
    author="PDF to Markdown Team",
    author_email="JGoldfed@gmail.com",
    description="Convert PDF documents to Markdown with high accuracy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jgoldfed/pdf-to-markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # PDF processing
        "PyMuPDF>=1.21.0",  # Updated version requirement for better compatibility
        "PyPDF2>=2.0.0",
        "pdfminer.six>=20200517",

        # Image processing
        "pdf2image>=1.16.0",
        "Pillow>=9.0.0",

        # OCR support
        "pytesseract>=0.3.8",

        # AI enhancement
        "google-generativeai>=0.3.0",

        # Utilities
        "tqdm>=4.64.0",
        "requests>=2.28.0",
        "pathlib>=1.0.1",
        "argparse>=1.4.0",
        "re>=2.2.1",  # For regex pattern matching in cleanups
    ],
    entry_points={
        "console_scripts": [
            "pdf2md=pdf2md:main",
        ],
    },
    # Include non-Python files
    package_data={
        "pdf2md": ["*.md"],
    },
    # Add project URLs for more info
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/pdf-to-markdown/issues",
        "Documentation": "https://github.com/yourusername/pdf-to-markdown",
        "Source Code": "https://github.com/yourusername/pdf-to-markdown",
    },
    # Keywords for PyPI
    keywords="pdf, markdown, conversion, ocr, ai",
)
