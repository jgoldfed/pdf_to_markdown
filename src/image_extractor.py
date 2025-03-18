"""
PDF to Markdown Converter - Image Extraction Module

This module handles the extraction of images from PDF documents,
saving them to a designated output folder and generating
appropriate markdown references.
"""

import os
import re
from typing import Dict, List, Tuple, Optional, Any, Set
import uuid
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image
import io
import fitz  # PyMuPDF


class ImageExtractor:
    """
    Extracts images from PDF pages and saves them to a designated output folder.
    Uses pdf2image and PyMuPDF for comprehensive image extraction.
    """

    def __init__(self, output_dir: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ImageExtractor with output directory and configuration.
        
        Args:
            output_dir: Directory where extracted images will be saved
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, self.config.get('images_dir', 'images'))
        self.image_format = self.config.get('image_format', 'png')
        self.image_quality = self.config.get('image_quality', 90)
        self.min_image_size = self.config.get('min_image_size', 100)  # Minimum width/height in pixels
        
        # Create images directory if it doesn't exist
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Track extracted images to avoid duplicates
        self.extracted_images: Set[str] = set()

    def extract_images_from_pdf(self, pdf_path: str) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extract images from all pages in a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary mapping page numbers (0-based) to lists of image info dictionaries
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        result = {}
        
        # Open the PDF with PyMuPDF
        pdf_document = fitz.open(pdf_path)
        
        # Process each page
        for page_num, page in enumerate(pdf_document):
            page_images = self.extract_images_from_page(pdf_path, page_num, page)
            if page_images:
                result[page_num] = page_images
        
        pdf_document.close()
        return result
    
    def extract_images_from_page(self, pdf_path: str, page_num: int, 
                                page: Optional[fitz.Page] = None) -> List[Dict[str, Any]]:
        """
        Extract images from a specific page in a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (0-based)
            page: Optional PyMuPDF page object (if already loaded)
            
        Returns:
            List of dictionaries containing image information
        """
        extracted_images = []
        
        # Method 1: Extract images using PyMuPDF
        pymupdf_images = self._extract_images_with_pymupdf(pdf_path, page_num, page)
        extracted_images.extend(pymupdf_images)
        
        # If no images were found with PyMuPDF or additional extraction is needed
        if not pymupdf_images or self.config.get('use_pdf2image', True):
            # Method 2: Convert page to image and extract embedded images
            pdf2image_images = self._extract_images_with_pdf2image(pdf_path, page_num)
            
            # Only add images that don't overlap significantly with already extracted ones
            for img_info in pdf2image_images:
                if not self._is_duplicate_image(img_info, pymupdf_images):
                    extracted_images.append(img_info)
        
        return extracted_images
    
    def _extract_images_with_pymupdf(self, pdf_path: str, page_num: int, 
                                    page: Optional[fitz.Page] = None) -> List[Dict[str, Any]]:
        """
        Extract images from a page using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (0-based)
            page: Optional PyMuPDF page object (if already loaded)
            
        Returns:
            List of dictionaries containing image information
        """
        extracted_images = []
        
        # Open the PDF if page is not provided
        close_pdf = False
        if page is None:
            pdf_document = fitz.open(pdf_path)
            page = pdf_document[page_num]
            close_pdf = True
        
        # Get image list
        image_list = page.get_images(full=True)
        
        # Process each image
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            
            if base_image:
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Load image data
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Skip small images (likely icons or decorations)
                    if image.width < self.min_image_size or image.height < self.min_image_size:
                        continue
                    
                    # Generate a unique filename
                    image_filename = f"page_{page_num+1}_img_{img_index+1}.{self.image_format}"
                    image_path = os.path.join(self.images_dir, image_filename)
                    
                    # Convert to desired format and save
                    if self.image_format.lower() in ['jpg', 'jpeg']:
                        # Convert to RGB if saving as JPEG (JPEG doesn't support alpha channel)
                        if image.mode in ['RGBA', 'LA'] or (image.mode == 'P' and 'transparency' in image.info):
                            image = image.convert('RGB')
                        image.save(image_path, quality=self.image_quality)
                    else:
                        image.save(image_path)
                    
                    # Get image position on the page
                    # This is an approximation as PyMuPDF doesn't directly provide position
                    # We'll use the image reference to find its location
                    rect = None
                    for img_rect in page.get_image_rects():
                        if img_rect[0] == xref:
                            rect = img_rect[1]
                            break
                    
                    # Create image info dictionary
                    image_info = {
                        'filename': image_filename,
                        'path': image_path,
                        'width': image.width,
                        'height': image.height,
                        'format': self.image_format,
                        'page': page_num,
                        'rect': rect,
                        'md_reference': f"![Image from page {page_num+1}](images/{image_filename})"
                    }
                    
                    extracted_images.append(image_info)
                    self.extracted_images.add(image_path)
                    
                except Exception as e:
                    print(f"Error processing image: {e}")
        
        # Close the PDF if we opened it
        if close_pdf:
            page.parent.close()
        
        return extracted_images
    
    def _extract_images_with_pdf2image(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract images by converting the page to an image using pdf2image.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (0-based)
            
        Returns:
            List of dictionaries containing image information
        """
        extracted_images = []
        
        try:
            # Convert the specific page to an image
            images = convert_from_path(
                pdf_path, 
                first_page=page_num+1, 
                last_page=page_num+1,
                dpi=300,  # Higher DPI for better quality
                fmt=self.image_format
            )
            
            if images:
                # Save the whole page as an image
                page_image = images[0]
                image_filename = f"page_{page_num+1}_full.{self.image_format}"
                image_path = os.path.join(self.images_dir, image_filename)
                
                # Convert to desired format and save
                if self.image_format.lower() in ['jpg', 'jpeg']:
                    # Convert to RGB if saving as JPEG
                    if page_image.mode in ['RGBA', 'LA'] or (page_image.mode == 'P' and 'transparency' in page_image.info):
                        page_image = page_image.convert('RGB')
                    page_image.save(image_path, quality=self.image_quality)
                else:
                    page_image.save(image_path)
                
                # Create image info dictionary
                image_info = {
                    'filename': image_filename,
                    'path': image_path,
                    'width': page_image.width,
                    'height': page_image.height,
                    'format': self.image_format,
                    'page': page_num,
                    'is_full_page': True,
                    'md_reference': f"![Page {page_num+1}](images/{image_filename})"
                }
                
                extracted_images.append(image_info)
                self.extracted_images.add(image_path)
                
        except Exception as e:
            print(f"Error converting page to image: {e}")
        
        return extracted_images
    
    def _is_duplicate_image(self, new_image: Dict[str, Any], 
                           existing_images: List[Dict[str, Any]]) -> bool:
        """
        Check if a new image significantly overlaps with any existing image.
        
        Args:
            new_image: New image info dictionary
            existing_images: List of existing image info dictionaries
            
        Returns:
            True if the new image is likely a duplicate, False otherwise
        """
        # If it's a full page image and we already have other images, it's not a duplicate
        if new_image.get('is_full_page', False) and existing_images:
            return False
        
        # If we don't have position information, we can't determine overlap
        if 'rect' not in new_image or new_image['rect'] is None:
            return False
        
        for img in existing_images:
            if 'rect' not in img or img['rect'] is None:
                continue
            
            # Calculate overlap between rectangles
            rect1 = new_image['rect']
            rect2 = img['rect']
            
            # Check for overlap
            if (rect1.x0 < rect2.x1 and rect1.x1 > rect2.x0 and
                rect1.y0 < rect2.y1 and rect1.y1 > rect2.y0):
                
                # Calculate overlap area
                overlap_width = min(rect1.x1, rect2.x1) - max(rect1.x0, rect2.x0)
                overlap_height = min(rect1.y1, rect2.y1) - max(rect1.y0, rect2.y0)
                overlap_area = overlap_width * overlap_height
                
                # Calculate areas of both rectangles
                area1 = (rect1.x1 - rect1.x0) * (rect1.y1 - rect1.y0)
                area2 = (rect2.x1 - rect2.x0) * (rect2.y1 - rect2.y0)
                
                # If overlap is more than 50% of either image, consider it a duplicate
                if overlap_area > 0.5 * min(area1, area2):
                    return True
        
        return False
    
    def get_markdown_references(self, page_num: int, images: List[Dict[str, Any]]) -> str:
        """
        Generate Markdown references for the extracted images.
        
        Args:
            page_num: Page number (0-based)
            images: List of image info dictionaries
            
        Returns:
            Markdown text with image references
        """
        if not images:
            return ""
        
        markdown = "\n\n"
        
        # Add each image reference
        for img in images:
            # Skip full page images if we have other images from the same page
            if img.get('is_full_page', False) and len(images) > 1:
                continue
            
            # Add the image reference
            markdown += img['md_reference'] + "\n\n"
        
        return markdown


# Simple test function
def test_image_extraction(pdf_path, output_dir):
    """Test the image extraction functionality."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = ImageExtractor(output_dir)
    try:
        # Extract images from all pages
        images = extractor.extract_images_from_pdf(pdf_path)
        
        total_images = sum(len(page_images) for page_images in images.values())
        print(f"Successfully extracted {total_images} images from {len(images)} pages.")
        
        # Print sample markdown references
        if images:
            first_page = min(images.keys())
            print("\nSample Markdown references:")
            print(extractor.get_markdown_references(first_page, images[first_page]))
        
        return True
    except Exception as e:
        print(f"Error extracting images: {e}")
        return False


if __name__ == "__main__":
    # This will be executed when the module is run directly
    import sys
    
    if len(sys.argv) > 2:
        pdf_path = sys.argv[1]
        output_dir = sys.argv[2]
        test_image_extraction(pdf_path, output_dir)
    else:
        print("Please provide a PDF file path and output directory as arguments.")
