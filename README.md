# PDF to Markdown Converter

A Python program that converts PDF documents to Markdown format with high accuracy, supporting image extraction, OCR, parallel processing, and AI enhancement.

## Features

- **Accurate Text Extraction**: Uses pdfminer.six for high-quality text extraction with layout preservation
- **Image Handling**: Extracts images from PDFs and saves them to a separate folder with proper references
- **OCR Support**: Integrates Tesseract OCR for handling scanned documents or image-based PDFs
- **Parallel Processing**: Processes multiple pages simultaneously for efficient handling of large documents
- **AI Enhancement**: Uses Google's Gemini 2 API to improve formatting accuracy and structure
- **Flexible Configuration**: Supports command-line options and configuration files

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR (for OCR capabilities)
- Poppler (for PDF processing)

### Install Dependencies

```bash
# Install Tesseract OCR and Poppler
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils

# Install Python dependencies
pip install pytesseract pdf2image pdfminer.six PyPDF2 Pillow google-generativeai tqdm PyMuPDF
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/pdf-to-markdown.git
cd pdf-to-markdown
```

## Usage

### Basic Usage

```bash
python pdf2md.py input.pdf --output output_directory
```

### Command-Line Options

```
usage: pdf2md.py [-h] [--output OUTPUT] [--config CONFIG] [--images-dir IMAGES_DIR] [--ocr] [--no-ai] [--api-key API_KEY] [--parallel PARALLEL] [--verbose]
                 pdf_file

Convert PDF to Markdown with high accuracy

positional arguments:
  pdf_file              Path to the PDF file to convert

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Output directory (default: current directory)
  --config CONFIG, -c CONFIG
                        Path to configuration file
  --images-dir IMAGES_DIR
                        Images directory name (default: images)
  --ocr                 Force OCR for all pages
  --no-ai               Disable AI enhancement
  --api-key API_KEY     Google API key for Gemini 2 (if not set in environment)
  --parallel PARALLEL   Number of parallel workers (default: CPU count)
  --verbose, -v         Enable verbose output
```

### Using AI Enhancement

To use the AI enhancement feature, you need to provide a Google API key for Gemini 2:

```bash
# Set API key as environment variable
export GOOGLE_API_KEY=your_api_key_here

# Or provide it as a command-line option
python pdf2md.py input.pdf --api-key your_api_key_here
```

### Configuration File

You can customize the converter behavior using a JSON configuration file:

```bash
python pdf2md.py input.pdf --config config.json
```

Example configuration file (`config.json`):

```json
{
  "output": {
    "markdown_extension": ".md",
    "image_format": "png",
    "image_quality": 90,
    "images_dir": "images"
  },
  "extraction": {
    "min_text_confidence": 0.7,
    "force_ocr": false
  },
  "ocr": {
    "language": "eng",
    "config": "--psm 6",
    "dpi": 300
  },
  "ai": {
    "enabled": true,
    "model": "gemini-pro",
    "temperature": 0.2,
    "chunk_size": 4000
  },
  "parallel": {
    "max_workers": null,
    "use_processes": true,
    "chunk_size": 1,
    "show_progress": true
  }
}
```

## Examples

### Convert a PDF with Default Settings

```bash
python pdf2md.py document.pdf -o output
```

### Convert a PDF with OCR Enabled

```bash
python pdf2md.py scanned_document.pdf -o output --ocr
```

### Convert a PDF with Maximum Performance

```bash
python pdf2md.py large_document.pdf -o output --parallel 8
```

### Convert a PDF without AI Enhancement

```bash
python pdf2md.py document.pdf -o output --no-ai
```

## Architecture

The converter is built with a modular architecture:

1. **TextExtractor**: Extracts text content from PDF pages with layout preservation
2. **ImageExtractor**: Identifies and extracts images from PDF pages
3. **OCRProcessor**: Processes pages or regions that require OCR
4. **ParallelExecutor**: Manages parallel processing of PDF pages
5. **AIEnhancer**: Post-processes Markdown using Google Gemini 2
6. **PDFToMarkdownConverter**: Main class that integrates all components

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [pdfminer.six](https://github.com/pdfminer/pdfminer.six) for PDF text extraction
- [pdf2image](https://github.com/Belval/pdf2image) for PDF to image conversion
- [pytesseract](https://github.com/madmaze/pytesseract) for OCR capabilities
- [Google Generative AI](https://ai.google.dev/) for AI enhancement
