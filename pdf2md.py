#!/usr/bin/env python3
"""
PDF to Markdown Converter

A Python program that converts PDF documents to Markdown format with high accuracy,
supporting image extraction, OCR, parallel processing, and AI enhancement.
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF
import google.generativeai as genai
from tqdm import tqdm
import re

# Import component modules
from src.text_extractor import TextExtractor
from src.image_extractor import ImageExtractor
from src.ocr_processor import OCRProcessor
from src.parallel_processor import ParallelExecutor, PDFParallelProcessor
from src.ai_enhancer import AIEnhancer


class PDFToMarkdownConverter:
    """
    Main class for the PDF to Markdown converter application.
    Integrates all components and provides a command-line interface.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the converter with configuration.

        Args:
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self._init_components()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        default_config = {
            "output": {
                "markdown_extension": ".md",
                "image_format": "png",
                "image_quality": 90,
                "images_dir": "images"
            },
            "extraction": {
                "min_text_confidence": 0.7,
                "force_ocr": False
            },
            "ocr": {
                "language": "eng",
                "config": "--psm 6",
                "dpi": 300
            },
            "ai": {
                "enabled": True,
                "model": "gemini-pro",
                "temperature": 0.2,
                "chunk_size": 4000
            },
            "parallel": {
                "max_workers": None,  # None = CPU count
                "use_processes": True,
                "chunk_size": 1,
                "show_progress": True
            }
        }

        # If config path is provided, load and merge with defaults
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)

                # Merge user config with defaults (recursive)
                self._merge_configs(default_config, user_config)
            except Exception as e:
                print(f"Error loading configuration file: {e}")
                print("Using default configuration.")

        return default_config

    def _merge_configs(self, default_config: Dict[str, Any], user_config: Dict[str, Any]) -> None:
        """
        Recursively merge user configuration into default configuration.

        Args:
            default_config: Default configuration dictionary (modified in-place)
            user_config: User configuration dictionary
        """
        for key, value in user_config.items():
            if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                self._merge_configs(default_config[key], value)
            else:
                default_config[key] = value

    def _init_components(self) -> None:
        """Initialize all components with appropriate configuration."""
        # Text extractor
        self.text_extractor = TextExtractor(
            config=self.config.get("extraction", {}))

        # OCR processor
        self.ocr_processor = OCRProcessor(config=self.config.get("ocr", {}))

        # AI enhancer (initialized later if needed)
        self.ai_enhancer = None

    def convert(self, pdf_path: str, output_dir: str, options: Dict[str, Any] = None) -> str:
        """
        Convert a PDF document to Markdown.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save output files
            options: Additional options to override configuration

        Returns:
            Path to the generated Markdown file
        """
        options = options or {}

        # Validate input file
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get output filename
        pdf_filename = os.path.basename(pdf_path)
        markdown_filename = os.path.splitext(
            pdf_filename)[0] + self.config["output"]["markdown_extension"]
        markdown_path = os.path.join(output_dir, markdown_filename)

        # Initialize image extractor with output directory
        image_extractor = ImageExtractor(
            output_dir=output_dir,
            config={
                "images_dir": self.config["output"]["images_dir"],
                "image_format": self.config["output"]["image_format"],
                "image_quality": self.config["output"]["image_quality"]
            }
        )

        # Initialize parallel processor
        processor = PDFParallelProcessor(
            text_extractor=self.text_extractor,
            image_extractor=image_extractor,
            ocr_processor=self.ocr_processor,
            config=self.config.get("parallel", {})
        )

        print(f"Converting PDF to Markdown: {pdf_path}")
        start_time = time.time()

        # Process the PDF
        results = processor.process_pdf(pdf_path, output_dir)

        # Combine results into a single Markdown file
        with open(markdown_path, 'w', encoding='utf-8') as f:
            for page_num in sorted(results.keys()):
                page_result = results[page_num]
                if 'markdown' in page_result:
                    f.write(page_result['markdown'])
                    f.write("\n\n")

        # Apply AI enhancement if enabled
        use_ai = options.get('use_ai', self.config["ai"]["enabled"])
        if use_ai:
            api_key = options.get(
                'api_key') or os.environ.get('GOOGLE_API_KEY')
            if api_key:
                try:
                    # Initialize AI enhancer if not already done
                    if not self.ai_enhancer:
                        self.ai_enhancer = AIEnhancer(
                            api_key=api_key,
                            config=self.config.get("ai", {})
                        )

                    # Create a temporary file for the enhanced output
                    enhanced_path = markdown_path + ".enhanced"

                    print("Enhancing Markdown with AI...")
                    success = self.ai_enhancer.enhance_document(
                        markdown_path, enhanced_path)

                    if success:
                        # Replace the original file with the enhanced version
                        os.replace(enhanced_path, markdown_path)
                        print("AI enhancement completed successfully.")
                    else:
                        print("AI enhancement failed. Using unenhanced output.")
                except Exception as e:
                    print(f"Error during AI enhancement: {e}")
                    print("Using unenhanced output.")
            else:
                print(
                    "AI enhancement enabled but no API key provided. Using unenhanced output.")

        end_time = time.time()
        print(f"Conversion completed in {end_time - start_time:.2f} seconds.")
        print(f"Output saved to: {markdown_path}")

        return markdown_path


def setup_gemini(api_key):
    """Configure the Gemini API with the provided key"""
    genai.configure(api_key=api_key)
    # Use the correct model - Gemini models in the latest API version
    # Try this model instead of gemini-pro
    return genai.GenerativeModel('gemini-1.5-pro')


def enhance_with_ai(markdown_text, model):
    """Use Google's Gemini API to enhance the markdown formatting"""
    try:
        prompt = f"""
        Please improve the following markdown text to have better formatting. 
        Fix any formatting issues, organize headings properly, and make the content more readable.
        Format tables properly if they exist. Do not add or remove any content, just improve the formatting.
        
        Here's the markdown text:
        
        {markdown_text}
        """

        response = model.generate_content(prompt)
        enhanced_text = response.text
        return enhanced_text
    except Exception as e:
        print(f"Error during AI enhancement: {e}")
        print("Continuing with original text due to API error...")
        return markdown_text  # Return original text if API enhancement fails


def enhance_with_ai_in_chunks(markdown_text, model, chunk_size=10000):
    """Process large markdown document in chunks to avoid API limits"""

    # If the text is small enough, process it as a single chunk
    if len(markdown_text) <= chunk_size:
        return enhance_with_ai(markdown_text, model)

    # Split by page markers (assuming format like "## Page X")
    import re
    pages = re.split(r'(## Page \d+)', markdown_text)

    # Recombine with the headers
    chunks = []
    current_chunk = ""

    i = 0
    while i < len(pages):
        # If this is a header and there's a next element
        if i < len(pages) - 1 and pages[i].startswith('## Page'):
            header = pages[i]
            content = pages[i+1]
            page_text = header + content

            # If adding this page exceeds chunk size, start a new chunk
            if len(current_chunk) + len(page_text) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = page_text
            else:
                current_chunk += page_text

            i += 2  # Skip the next element since we've processed it
        else:
            # Handle edge case (content without header)
            if len(current_chunk) + len(pages[i]) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = pages[i]
            else:
                current_chunk += pages[i]
            i += 1

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)

    # Process each chunk
    enhanced_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        enhanced_chunk = enhance_with_ai(chunk, model)
        enhanced_chunks.append(enhanced_chunk)
        # Add delay between chunks to avoid rate limiting
        if i < len(chunks) - 1:
            time.sleep(2)

    # Combine the enhanced chunks
    return "".join(enhanced_chunks)


def clean_markdown_output(markdown_text):
    """Clean up common formatting issues in converted markdown text"""
    # Remove markdown code block markers
    cleaned_text = re.sub(r'```markdown\s*', '', markdown_text)
    cleaned_text = re.sub(r'```\s*', '', cleaned_text)

    # Fix multiple consecutive blank lines (replace with at most 2)
    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)

    # Remove any HTML comments for filepath references
    cleaned_text = re.sub(r'<!-- filepath:.*?-->', '', cleaned_text)

    return cleaned_text


def convert_pdf_to_markdown(pdf_path, output_path=None, images_dir=None, use_ai=True, api_key=None, verbose=False):
    """
    Convert a PDF file to Markdown format

    Args:
        pdf_path: Path to the PDF file
        output_path: Output file path (optional, defaults to PDF name + .md)
        images_dir: Directory to store extracted images (optional)
        use_ai: Whether to use AI enhancement
        api_key: API key for Google's Gemini
        verbose: Whether to print verbose output

    Returns:
        The path to the output file
    """
    start_time = time.time()

    # Set default output path if not provided
    if output_path is None:
        pdf_name = os.path.basename(pdf_path)
        pdf_name_without_ext = os.path.splitext(pdf_name)[0]
        output_path = f"{pdf_name_without_ext}.md"

    # If output_path is a directory, append the PDF filename
    if os.path.isdir(output_path):
        pdf_name = os.path.basename(pdf_path)
        pdf_name_without_ext = os.path.splitext(pdf_name)[0]
        output_path = os.path.join(output_path, f"{pdf_name_without_ext}.md")

    # Create images directory if needed
    if images_dir is None and output_path is not None:
        images_dir = os.path.join(os.path.dirname(output_path), "images")

    os.makedirs(images_dir, exist_ok=True)

    if verbose:
        print(f"Converting PDF to Markdown: {pdf_path}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Open the PDF
    doc = fitz.open(pdf_path)

    # Initialize markdown content
    markdown = []

    # Process each page
    total_pages = doc.page_count
    for page_num in tqdm(range(total_pages), desc="Processing pages", unit="page"):
        page = doc[page_num]
        text = page.get_text()

        # Add page header
        markdown.append(f"\n## Page {page_num+1}\n")

        # Add page text
        if text:
            markdown.append(text)
        else:
            markdown.append("*No text found on this page*")

        # Extract images if needed (simplified)
        # This is a placeholder - complete image extraction would require more code

    # Close the document
    doc.close()

    # Combine all markdown content
    markdown_text = "\n".join(markdown)

    # Enhance with AI if requested
    if use_ai and api_key and not args.skip_ai:
        if verbose:
            print("Enhancing Markdown with AI...")

        model = setup_gemini(api_key)
        markdown_text = enhance_with_ai(markdown_text, model)

        if verbose:
            print("AI enhancement completed successfully.")
    else:
        if verbose and api_key and args.skip_ai:
            print("AI enhancement skipped as requested.")

    # Clean the markdown output
    markdown_text = clean_markdown_output(markdown_text)

    # Write markdown to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    end_time = time.time()
    if verbose:
        print(f"Conversion completed in {end_time - start_time:.2f} seconds.")
        print(f"Output saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown with high accuracy")
    parser.add_argument("pdf_file", help="Path to the PDF file to convert")
    parser.add_argument(
        "--output", "-o", help="Output file path or directory", default=".")
    parser.add_argument(
        "--images-dir", help="Directory to store extracted images")
    parser.add_argument("--ocr", action="store_true",
                        help="Use OCR for text extraction (not implemented in this version)")
    parser.add_argument("--no-ai", action="store_true",
                        help="Disable AI enhancement")
    parser.add_argument("--api-key", help="API key for Google's Gemini")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel processes (not used in this version)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--skip-ai", action="store_true",
                        help="Skip AI enhancement even if API key is provided")

    args = parser.parse_args()

    # Validate PDF file exists
    if not os.path.exists(args.pdf_file):
        print(f"Error: PDF file not found - {args.pdf_file}")
        return 1

    # Convert the PDF
    convert_pdf_to_markdown(
        pdf_path=args.pdf_file,
        output_path=args.output,
        images_dir=args.images_dir,
        use_ai=not args.no_ai,
        api_key=args.api_key,
        verbose=args.verbose
    )

    print(f"PDF successfully converted to Markdown: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
