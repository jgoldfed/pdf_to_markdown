"""
PDF to Markdown Converter - Text Extraction Module

This module handles the extraction of text content from PDF documents,
preserving formatting and layout as much as possible.
"""

import os
import re
from typing import Dict, List, Tuple, Optional, Any

from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTTextBox, LTTextLine, LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
import io


class TextExtractor:
    """
    Extracts text content from PDF pages with layout preservation.
    Uses pdfminer.six for accurate text extraction.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TextExtractor with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.min_text_confidence = self.config.get('min_text_confidence', 0.7)
        
        # Configure layout analysis parameters
        self.laparams = LAParams(
            line_margin=self.config.get('line_margin', 0.5),
            char_margin=self.config.get('char_margin', 2.0),
            word_margin=self.config.get('word_margin', 0.1),
            boxes_flow=self.config.get('boxes_flow', 0.5),
            detect_vertical=self.config.get('detect_vertical', True)
        )

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """
        Extract text from all pages in a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary mapping page numbers (0-based) to extracted text
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        result = {}
        
        # Extract text from each page
        for page_num, page_text in enumerate(self._extract_pages_with_layout(pdf_path)):
            result[page_num] = page_text
            
        return result
    
    def extract_page(self, pdf_path: str, page_num: int) -> str:
        """
        Extract text from a specific page in a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (0-based)
            
        Returns:
            Extracted text from the specified page
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract text from the specified page
        output_string = io.StringIO()
        with open(pdf_path, 'rb') as f:
            resource_manager = PDFResourceManager()
            device = TextConverter(resource_manager, output_string, laparams=self.laparams)
            interpreter = PDFPageInterpreter(resource_manager, device)
            
            for i, page in enumerate(PDFPage.get_pages(f)):
                if i == page_num:
                    interpreter.process_page(page)
                    break
            
            device.close()
        
        text = output_string.getvalue()
        output_string.close()
        
        # Process the extracted text to improve formatting
        text = self._process_extracted_text(text)
        
        return text
    
    def _extract_pages_with_layout(self, pdf_path: str) -> List[str]:
        """
        Extract text from all pages with layout preservation.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of extracted text strings, one per page
        """
        pages_text = []
        
        # Extract text with layout analysis
        for page_layout in extract_pages(pdf_path, laparams=self.laparams):
            page_text = ""
            
            # Process text elements in the page
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text = element.get_text()
                    page_text += text
            
            # Process the extracted text to improve formatting
            page_text = self._process_extracted_text(page_text)
            pages_text.append(page_text)
        
        return pages_text
    
    def _process_extracted_text(self, text: str) -> str:
        """
        Process extracted text to improve formatting for Markdown conversion.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Processed text with improved formatting
        """
        # Remove excessive newlines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
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
        
        # Detect and format lists
        text = self._format_lists(text)
        
        # Detect and format code blocks or preformatted text
        text = self._format_code_blocks(text)
        
        return text
    
    def _format_lists(self, text: str) -> str:
        """
        Detect and format lists in the extracted text.
        
        Args:
            text: Extracted text
            
        Returns:
            Text with formatted lists
        """
        lines = text.split('\n')
        processed_lines = []
        
        # Common list markers
        bullet_patterns = [
            r'^\s*[•●■◦○□]',  # Unicode bullets
            r'^\s*[-*]',      # Hyphen or asterisk
            r'^\s*\(\d+\)',   # (1), (2), etc.
            r'^\s*\d+\.',     # 1., 2., etc.
            r'^\s*[a-z]\.',   # a., b., etc.
            r'^\s*[A-Z]\.',   # A., B., etc.
        ]
        
        for line in lines:
            # Check if line matches any list pattern
            is_list_item = False
            for pattern in bullet_patterns:
                if re.match(pattern, line):
                    is_list_item = True
                    break
            
            if is_list_item:
                # Convert to Markdown list format
                # For numbered lists, preserve the numbering
                if re.match(r'^\s*\d+\.', line):
                    processed_line = line
                # For other types, convert to bullet list
                else:
                    # Remove the original bullet and add Markdown bullet
                    processed_line = re.sub(r'^\s*[•●■◦○□\-*\(\d+\)][.\s]*', '- ', line)
                    processed_line = re.sub(r'^\s*[a-zA-Z]\.', '- ', processed_line)
                
                processed_lines.append(processed_line)
            else:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _format_code_blocks(self, text: str) -> str:
        """
        Detect and format code blocks in the extracted text.
        
        Args:
            text: Extracted text
            
        Returns:
            Text with formatted code blocks
        """
        lines = text.split('\n')
        processed_lines = []
        in_code_block = False
        
        for i, line in enumerate(lines):
            # Heuristic: lines with consistent indentation and special characters
            # might be code
            if (line.startswith('    ') and 
                ('{' in line or '}' in line or '(' in line or ')' in line or 
                 '=' in line or ';' in line or '//' in line or '#' in line)):
                
                if not in_code_block:
                    # Start a new code block
                    processed_lines.append('```')
                    in_code_block = True
                
                # Remove the indentation for code blocks
                processed_lines.append(line.strip())
            else:
                if in_code_block:
                    # End the code block
                    processed_lines.append('```')
                    in_code_block = False
                
                processed_lines.append(line)
        
        # Close any open code block
        if in_code_block:
            processed_lines.append('```')
        
        return '\n'.join(processed_lines)
    
    def needs_ocr(self, pdf_path: str, page_num: int, extracted_text: str) -> bool:
        """
        Determine if OCR is needed for a page based on the extracted text quality.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number
            extracted_text: Text already extracted from the page
            
        Returns:
            True if OCR is recommended, False otherwise
        """
        # If no text was extracted, OCR is definitely needed
        if not extracted_text.strip():
            return True
        
        # Check text density (characters per page)
        # A very low character count might indicate poor text extraction
        char_count = len(extracted_text)
        if char_count < 100:  # Arbitrary threshold
            return True
        
        # Check for common OCR indicators in the text
        ocr_indicators = ['�', '□', '■', '?', '¿', '¶']
        indicator_count = sum(extracted_text.count(c) for c in ocr_indicators)
        
        # If more than 5% of characters are indicators, suggest OCR
        if indicator_count > 0 and indicator_count / char_count > 0.05:
            return True
        
        # Check for text recognition confidence (simplified heuristic)
        # Count words that look like they might be misrecognized
        words = re.findall(r'\b\w+\b', extracted_text)
        suspicious_words = sum(1 for word in words if len(word) > 2 and (
            # Words with unusual character combinations
            re.search(r'[a-z][A-Z]', word) or
            # Words with digits mixed with letters
            re.search(r'[a-zA-Z][0-9]|[0-9][a-zA-Z]', word) or
            # Very long words (might be multiple words merged)
            len(word) > 20
        ))
        
        # If more than 10% of words look suspicious, suggest OCR
        if words and suspicious_words / len(words) > 0.1:
            return True
        
        return False


# Simple test function
def test_text_extraction(pdf_path):
    """Test the text extraction functionality."""
    extractor = TextExtractor()
    try:
        # Extract text from all pages
        pages = extractor.extract_text_from_pdf(pdf_path)
        
        print(f"Successfully extracted text from {len(pages)} pages.")
        
        # Print a sample from the first page
        if pages:
            first_page = pages[0]
            print("\nSample from first page:")
            print(first_page[:500] + "..." if len(first_page) > 500 else first_page)
            
            # Check if OCR might be needed
            needs_ocr = extractor.needs_ocr(pdf_path, 0, first_page)
            print(f"\nOCR recommended: {needs_ocr}")
        
        return True
    except Exception as e:
        print(f"Error extracting text: {e}")
        return False


if __name__ == "__main__":
    # This will be executed when the module is run directly
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        test_text_extraction(pdf_path)
    else:
        print("Please provide a PDF file path as an argument.")
