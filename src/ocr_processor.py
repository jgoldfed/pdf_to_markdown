"""
PDF to Markdown Converter - OCR Module

This module handles Optical Character Recognition (OCR) for PDF pages
that require it, integrating with the text extraction process.
"""

import os
import re
from typing import Dict, List, Tuple, Optional, Any
import tempfile
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import io


class OCRProcessor:
    """
    Processes PDF pages or regions that require OCR.
    Uses Tesseract OCR via pytesseract.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OCRProcessor with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.language = self.config.get('language', 'eng')
        self.dpi = self.config.get('dpi', 300)
        self.tesseract_config = self.config.get('tesseract_config', '--psm 6')
        self.temp_dir = self.config.get('temp_dir', tempfile.gettempdir())
        
        # Verify Tesseract is installed and accessible
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(f"Tesseract OCR is not properly installed or accessible: {e}")

    def process_page(self, pdf_path: str, page_num: int) -> str:
        """
        Perform OCR on a specific page in a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (0-based)
            
        Returns:
            Extracted text from OCR
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Convert PDF page to image
        images = convert_from_path(
            pdf_path, 
            first_page=page_num+1, 
            last_page=page_num+1,
            dpi=self.dpi
        )
        
        if not images:
            return ""
        
        # Process the page image with OCR
        page_image = images[0]
        return self._perform_ocr(page_image)
    
    def process_image(self, image: Image.Image) -> str:
        """
        Perform OCR on an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text from OCR
        """
        return self._perform_ocr(image)
    
    def _perform_ocr(self, image: Image.Image) -> str:
        """
        Perform OCR on an image using Tesseract.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text from OCR
        """
        # Preprocess the image for better OCR results
        processed_image = self._preprocess_image(image)
        
        # Perform OCR
        try:
            text = pytesseract.image_to_string(
                processed_image,
                lang=self.language,
                config=self.tesseract_config
            )
            
            # Post-process the OCR result
            text = self._postprocess_text(text)
            
            return text
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess the image to improve OCR accuracy.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image object
        """
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply preprocessing based on configuration
        if self.config.get('enhance_contrast', True):
            # Enhance contrast using histogram equalization
            img_array = np.array(image)
            hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * 255 / cdf[-1]
            img_equalized = np.interp(img_array.flatten(), bins[:-1], cdf_normalized)
            image = Image.fromarray(img_equalized.reshape(img_array.shape).astype('uint8'))
        
        if self.config.get('denoise', True):
            # Simple denoising by slight blurring
            image = image.filter(Image.BLUR)
            image = image.filter(Image.SHARPEN)  # Sharpen after blur to maintain edges
        
        if self.config.get('deskew', True):
            # Deskewing is complex and requires additional libraries
            # For a production system, consider using cv2 for deskewing
            pass
        
        return image
    
    def _postprocess_text(self, text: str) -> str:
        """
        Post-process OCR text to improve quality.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Processed text
        """
        if not text:
            return text
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = text.replace('|', 'I')  # Pipe to I
        text = text.replace('l', 'l')  # Lowercase L to lowercase L (font fix)
        
        # Fix line breaks
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Single newlines become spaces
        text = re.sub(r'\n{3,}', '\n\n', text)  # More than 2 newlines become 2
        
        # Fix common punctuation errors
        text = text.replace(' ,', ',')
        text = text.replace(' .', '.')
        text = text.replace(' :', ':')
        text = text.replace(' ;', ';')
        text = text.replace(' !', '!')
        text = text.replace(' ?', '?')
        
        # Detect and format headings
        lines = text.split('\n')
        processed_lines = []
        
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                processed_lines.append(line)
                continue
            
            # Check if line might be a heading (shorter, standalone line)
            is_heading = False
            if len(line.strip()) < 100:  # Potential heading (not too long)
                prev_empty = i == 0 or not lines[i-1].strip()
                next_empty = i == len(lines)-1 or not lines[i+1].strip()
                
                if prev_empty and next_empty:
                    # Standalone line surrounded by empty lines - likely a heading
                    is_heading = True
            
            if is_heading:
                # Format as a heading based on characteristics
                if len(line) < 20:  # Very short - likely a main heading
                    processed_lines.append(f"# {line.strip()}")
                elif len(line) < 50:  # Medium length - likely a subheading
                    processed_lines.append(f"## {line.strip()}")
                else:  # Longer but still a potential heading
                    processed_lines.append(f"### {line.strip()}")
            else:
                processed_lines.append(line)
        
        text = '\n'.join(processed_lines)
        
        return text
    
    def process_page_with_regions(self, pdf_path: str, page_num: int, 
                                 regions: List[Tuple[int, int, int, int]] = None) -> str:
        """
        Perform OCR on specific regions of a PDF page.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (0-based)
            regions: List of regions as (x1, y1, x2, y2) tuples, or None for the whole page
            
        Returns:
            Extracted text from OCR
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Convert PDF page to image
        images = convert_from_path(
            pdf_path, 
            first_page=page_num+1, 
            last_page=page_num+1,
            dpi=self.dpi
        )
        
        if not images:
            return ""
        
        page_image = images[0]
        
        if not regions:
            # Process the whole page
            return self._perform_ocr(page_image)
        
        # Process each region and combine results
        combined_text = ""
        
        for region in regions:
            x1, y1, x2, y2 = region
            # Crop the image to the region
            region_image = page_image.crop((x1, y1, x2, y2))
            # Perform OCR on the region
            region_text = self._perform_ocr(region_image)
            combined_text += region_text + "\n\n"
        
        return combined_text.strip()


# Simple test function
def test_ocr(pdf_path, page_num=0):
    """Test the OCR functionality."""
    ocr_processor = OCRProcessor()
    try:
        # Perform OCR on the specified page
        text = ocr_processor.process_page(pdf_path, page_num)
        
        print(f"Successfully performed OCR on page {page_num+1}.")
        print("\nSample OCR result:")
        print(text[:500] + "..." if len(text) > 500 else text)
        
        return True
    except Exception as e:
        print(f"Error performing OCR: {e}")
        return False


if __name__ == "__main__":
    # This will be executed when the module is run directly
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        page_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        test_ocr(pdf_path, page_num)
    else:
        print("Please provide a PDF file path as an argument.")
