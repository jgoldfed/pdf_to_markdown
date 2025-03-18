"""
PDF to Markdown Converter - Parallel Processing Module

This module handles parallel processing of PDF pages to improve performance
for large documents, managing task distribution and result collection.
"""

import os
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import multiprocessing
from tqdm import tqdm


def process_page_wrapper(args):
    # Extract arguments
    page_num, pdf_path, extractor, ocr_processor, ai_enhancer, output_dir, images_dir, force_ocr = args
    # Your existing processing logic here
    return result


# Add these module-level functions at the top of the file after the imports
def process_item_wrapper(args):
    """Wrapper for processing a single item."""
    item, func, extra_args, extra_kwargs = args
    return func(item, *extra_args, **extra_kwargs)


def process_indexed_item_wrapper(args):
    """Wrapper for processing an indexed item."""
    indexed_item, func, extra_args, extra_kwargs = args
    index, item = indexed_item
    try:
        # CHANGE THIS LINE - Remove the extra asterisk
        # Changed from *extra_kwargs
        result = func(item, *extra_args, extra_kwargs)
        return index, result
    except Exception as e:
        # Return the error with the index
        return index, {"error": str(e)}


class ParallelExecutor:
    """
    Manages parallel processing of PDF pages using concurrent.futures.
    Handles task scheduling, resource management, and progress tracking.
    """

    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = True,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ParallelExecutor with configuration.

        Args:
            max_workers: Maximum number of worker processes/threads (None = CPU count)
            use_processes: Whether to use processes (True) or threads (False)
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.max_workers = max_workers or self.config.get(
            'max_workers') or multiprocessing.cpu_count()
        self.use_processes = use_processes or self.config.get(
            'use_processes', True)
        self.chunk_size = self.config.get('chunk_size', 1)
        self.show_progress = self.config.get('show_progress', True)

        # Validate max_workers
        if self.max_workers <= 0:
            self.max_workers = multiprocessing.cpu_count()

        # Executor type based on configuration
        self.executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

    def process_items(self, func: Callable, items: List[Any], *args, **kwargs) -> List[Any]:
        """
        Process a list of items in parallel using the provided function.
        """
        if not items:
            return []

        # Create argument tuples for each item
        args_list = [(item, func, args, kwargs) for item in items]

        # Process items in parallel
        with self.executor_class(max_workers=self.max_workers) as executor:
            # Set up progress tracking if enabled
            if self.show_progress:
                results = list(tqdm(
                    executor.map(process_item_wrapper, args_list,
                                 chunksize=self.chunk_size),
                    total=len(items),
                    desc="Processing pages",
                    unit="page"
                ))
            else:
                results = list(executor.map(
                    process_item_wrapper, args_list, chunksize=self.chunk_size))

        return results

    def process_items_with_index(self, func: Callable, items: List[Any], *args, **kwargs) -> Dict[int, Any]:
        """
        Process a list of items in parallel using the provided function,
        preserving the index of each item in the result.
        """
        if not items:
            return {}

        # Create a list of (index, item) tuples
        indexed_items = list(enumerate(items))

        # Create argument tuples for each indexed item
        args_list = [(indexed_item, func, args, kwargs)
                     for indexed_item in indexed_items]

        # Process items in parallel
        results = {}

        with self.executor_class(max_workers=self.max_workers) as executor:
            # Set up progress tracking if enabled
            if self.show_progress:
                future_to_index = {
                    # Getting index from (index, item)
                    executor.submit(process_indexed_item_wrapper, arg): arg[0][0]
                    for arg in args_list
                }

                # Create progress bar
                pbar = tqdm(total=len(items),
                            desc="Processing pages", unit="page")

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_index):
                    index, result = future.result()
                    results[index] = result
                    pbar.update(1)

                pbar.close()
            else:
                # Without progress tracking, use map for simplicity
                for index, result in executor.map(process_indexed_item_wrapper, args_list):
                    results[index] = result

        return results

    def process_pdf_pages(self, func, pdf_path, num_pages, output_dir, ai_enhancer=None, images_dir=None):
        """Process PDF pages in parallel."""
        # Set default images directory if not provided
        if images_dir is None:
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

        # Create a list of page numbers
        page_numbers = list(range(num_pages))

        # Process pages with index to preserve page numbers
        return self.process_items_with_index(
            func=func,
            items=[page_numbers],  # List of page numbers
            pdf_path=pdf_path,     # Pass other args as kwargs
            output_dir=output_dir
        )


class PDFParallelProcessor:
    """
    High-level wrapper for parallel processing of PDF documents,
    coordinating the conversion process across multiple components.
    """

    def __init__(self, text_extractor=None, image_extractor=None, ocr_processor=None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the PDFParallelProcessor with components and configuration.

        Args:
            text_extractor: TextExtractor instance
            image_extractor: ImageExtractor instance
            ocr_processor: OCRProcessor instance
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.text_extractor = text_extractor
        self.image_extractor = image_extractor
        self.ocr_processor = ocr_processor

        # Initialize parallel executor
        max_workers = self.config.get('max_workers')
        use_processes = self.config.get('use_processes', True)
        self.executor = ParallelExecutor(
            max_workers, use_processes, self.config)

    def process_pdf(self, pdf_path: str, output_dir: str) -> Dict[int, Dict[str, Any]]:
        """
        Process a PDF document in parallel, extracting text, images, and performing OCR as needed.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory where output will be saved

        Returns:
            Dictionary mapping page numbers to page results
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get number of pages in the PDF
        import fitz  # PyMuPDF
        pdf_document = fitz.open(pdf_path)
        num_pages = len(pdf_document)
        pdf_document.close()

        # Process all pages in parallel
        results = self.executor.process_pdf_pages(
            self._process_page,
            pdf_path,
            num_pages,
            output_dir=output_dir
        )

        # Sort results by page number
        sorted_results = {page_num: results[page_num]
                          for page_num in sorted(results.keys())}

        return sorted_results

    def _process_page(self, pdf_path: str, page_num: int, output_dir: str) -> Dict[str, Any]:
        """
        Process a single page of a PDF document.

        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (0-based)
            output_dir: Directory where output will be saved

        Returns:
            Dictionary containing page processing results
        """
        result = {
            'page_num': page_num,
            'text': None,
            'images': None,
            'ocr_used': False,
            'markdown': None
        }

        try:
            # Extract text
            if self.text_extractor:
                text = self.text_extractor.extract_page(pdf_path, page_num)
                result['text'] = text

                # Check if OCR is needed
                needs_ocr = self.text_extractor.needs_ocr(
                    pdf_path, page_num, text)
                result['ocr_needed'] = needs_ocr

                # Perform OCR if needed and OCR processor is available
                if needs_ocr and self.ocr_processor:
                    ocr_text = self.ocr_processor.process_page(
                        pdf_path, page_num)

                    # If OCR text is better than extracted text, use it
                    if len(ocr_text.strip()) > len(text.strip()):
                        result['text'] = ocr_text
                        result['ocr_used'] = True

            # Extract images
            if self.image_extractor:
                images = self.image_extractor.extract_images_from_page(
                    pdf_path, page_num)
                result['images'] = images

                # Generate markdown references for images
                if images:
                    result['image_markdown'] = self.image_extractor.get_markdown_references(
                        page_num, images)

            # Generate markdown for the page
            markdown = result.get('text', '')

            # Add image references if available
            if result.get('image_markdown'):
                markdown += result['image_markdown']

            result['markdown'] = markdown

        except Exception as e:
            result['error'] = str(e)

        return result


# Simple test function
def test_parallel_processing(pdf_path, output_dir, max_workers=None):
    """Test the parallel processing functionality."""
    from text_extractor import TextExtractor
    from image_extractor import ImageExtractor
    from ocr_processor import OCRProcessor

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize components
    text_extractor = TextExtractor()
    image_extractor = ImageExtractor(output_dir)
    ocr_processor = OCRProcessor()

    # Configure parallel processor
    config = {
        'max_workers': max_workers or multiprocessing.cpu_count(),
        'use_processes': True,
        'show_progress': True
    }

    processor = PDFParallelProcessor(
        text_extractor=text_extractor,
        image_extractor=image_extractor,
        ocr_processor=ocr_processor,
        config=config
    )

    try:
        # Process the PDF
        start_time = time.time()
        results = processor.process_pdf(pdf_path, output_dir)
        end_time = time.time()

        # Report results
        print(
            f"Successfully processed {len(results)} pages in {end_time - start_time:.2f} seconds.")
        print(f"Using {config['max_workers']} worker processes.")

        # Save a sample of the first page result
        if results:
            first_page = min(results.keys())
            sample_path = os.path.join(
                output_dir, f"page_{first_page+1}_sample.md")
            with open(sample_path, 'w') as f:
                f.write(results[first_page]['markdown'])
            print(f"Sample output saved to {sample_path}")

        return True
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        return False


if __name__ == "__main__":
    # This will be executed when the module is run directly
    import sys

    if len(sys.argv) > 2:
        pdf_path = sys.argv[1]
        output_dir = sys.argv[2]
        max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else None
        test_parallel_processing(pdf_path, output_dir, max_workers)
    else:
        print("Please provide a PDF file path and output directory as arguments.")
