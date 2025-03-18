# PDF to Markdown Converter

A Python program that converts PDF documents to Markdown format with high accuracy, supporting image extraction, OCR, and AI enhancement via Google's Gemini API.

## Features

- **Accurate Text Extraction**: Uses PyMuPDF for efficient text extraction with layout preservation
- **Image Handling**: Extracts images from PDFs and saves them to a separate folder with proper references
- **OCR Support**: Integrates Tesseract OCR for handling scanned documents or image-based PDFs
- **AI Enhancement**: Uses Google's Gemini API to improve formatting accuracy and structure
- **User-Friendly Interface**: Simple command-line interface with flexible configuration options

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR (for OCR capabilities)
- Poppler (for PDF processing)

#### Windows Installation of Prerequisites

1. **Tesseract OCR**:

   - Download the installer from the [UB Mannheim repository](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add the installation directory to your PATH environment variable

2. **Poppler**:
   - Download pre-built binaries from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases/)
   - Add the bin directory to your PATH environment variable

#### Linux Installation of Prerequisites

```bash
# Install Tesseract OCR and Poppler
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

### Docker Installation

A Dockerfile is provided for easy setup:

```bash
# Build the Docker image
docker build -t pdf_to_markdown_image -f Dockerfile .

# Run a container
docker run -it --name pdf_to_markdown \
  -v "${PWD}/pdf_to_markdown:/app/pdf_to_markdown" \
  -p 8080:80 \
  pdf_to_markdown_image
```

### Install Python Dependencies

```bash
# Install Python dependencies
pip install pytesseract pdf2image pdfminer.six PyPDF2 Pillow google-generativeai tqdm PyMuPDF
```

Alternatively, install the package directly:

```bash
pip install .
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
usage: pdf2md.py [-h] [--output OUTPUT] [--images-dir IMAGES_DIR] [--ocr] [--no-ai] [--skip-ai] [--api-key API_KEY] [--parallel PARALLEL] [--verbose]
                 pdf_file

Convert PDF to Markdown with high accuracy

positional arguments:
  pdf_file              Path to the PDF file to convert

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Output directory (default: current directory)
  --images-dir IMAGES_DIR
                        Images directory name (default: images)
  --ocr                 Force OCR for all pages
  --no-ai               Disable AI enhancement
  --skip-ai             Skip AI enhancement even if API key is provided
  --api-key API_KEY     Google API key for Gemini API
  --parallel PARALLEL   Number of parallel workers (default: 1)
  --verbose, -v         Enable verbose output
```

### Using AI Enhancement

To use the AI enhancement feature, you need to provide a Google API key for Gemini:

```bash
# Set API key as environment variable
export GOOGLE_API_KEY=your_api_key_here

# Or provide it as a command-line option
python pdf2md.py input.pdf --api-key your_api_key_here
```

If you encounter rate limiting issues with the API, you can use the `--skip-ai` flag:

```bash
python pdf2md.py input.pdf --api-key your_api_key_here --skip-ai
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

### Convert a PDF without AI Enhancement

```bash
python pdf2md.py document.pdf -o output --no-ai
```

## Known Limitations

- Large PDFs may face API limitations when using AI enhancement
- Some complex document layouts might require manual adjustment after conversion
- The parallel processing feature has been simplified for reliability

## Troubleshooting

### API Quota Errors

If you encounter a `429 Resource has been exhausted` error, you've hit the Google API quota limit. Solutions:

- Wait and try again later
- Use `--skip-ai` flag to skip AI enhancement
- Request an increased quota from Google

### Text Extraction Issues

If the converter produces empty or poorly formatted output:

- Try enabling OCR with `--ocr` flag
- Check if the PDF contains actual text (not just images)
- Verify the PDF can be opened properly in other applications

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) for PDF processing
- [pdf2image](https://github.com/Belval/pdf2image) for PDF to image conversion
- [pytesseract](https://github.com/madmaze/pytesseract) for OCR capabilities
- [Google Generative AI](https://ai.google.dev/) for AI enhancement
