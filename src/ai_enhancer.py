"""
PDF to Markdown Converter - AI Enhancement Module

This module uses Google's Gemini 2 API to enhance the quality and accuracy
of the converted Markdown, improving formatting and structure.
"""

import os
import re
import time
from typing import Dict, List, Tuple, Optional, Any
import json

import google.generativeai as genai
from tqdm import tqdm


class AIEnhancer:
    """
    Enhances Markdown using Google's Gemini 2 API to improve formatting accuracy and structure.
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AIEnhancer with API key and configuration.
        
        Args:
            api_key: Google API key for Gemini 2 (if None, will look in environment variables)
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Get API key from config, parameter, or environment variable
        self.api_key = api_key or self.config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
        
        if not self.api_key:
            raise ValueError("Google API key is required. Provide it as a parameter, in config, or set GOOGLE_API_KEY environment variable.")
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        
        # Set model parameters
        self.model_name = self.config.get('model', 'gemini-pro')
        self.temperature = self.config.get('temperature', 0.2)
        self.max_output_tokens = self.config.get('max_output_tokens', 8192)
        self.top_p = self.config.get('top_p', 0.95)
        self.top_k = self.config.get('top_k', 40)
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_output_tokens": self.max_output_tokens,
            }
        )
        
        # Set chunk size for processing large documents
        self.chunk_size = self.config.get('chunk_size', 4000)
        
        # Set retry parameters
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 2)

    def enhance_markdown(self, markdown: str) -> str:
        """
        Enhance Markdown using Gemini 2 to improve formatting and structure.
        
        Args:
            markdown: Original Markdown text
            
        Returns:
            Enhanced Markdown text
        """
        if not markdown.strip():
            return markdown
        
        # For large documents, process in chunks
        if len(markdown) > self.chunk_size:
            return self._process_large_document(markdown)
        
        # For smaller documents, process in one go
        return self._enhance_chunk(markdown)
    
    def _process_large_document(self, markdown: str) -> str:
        """
        Process a large document by splitting it into chunks and enhancing each chunk.
        
        Args:
            markdown: Original Markdown text
            
        Returns:
            Enhanced Markdown text
        """
        # Split the document into chunks
        chunks = self._split_into_chunks(markdown)
        
        # Process each chunk
        enhanced_chunks = []
        for i, chunk in enumerate(tqdm(chunks, desc="Enhancing document chunks", unit="chunk")):
            enhanced_chunk = self._enhance_chunk(chunk)
            enhanced_chunks.append(enhanced_chunk)
        
        # Combine the enhanced chunks
        return "\n\n".join(enhanced_chunks)
    
    def _split_into_chunks(self, markdown: str) -> List[str]:
        """
        Split a large Markdown document into chunks for processing.
        
        Args:
            markdown: Original Markdown text
            
        Returns:
            List of Markdown chunks
        """
        # Split by headings to preserve document structure
        heading_pattern = r'^(#{1,6}\s.+)$'
        lines = markdown.split('\n')
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            # Check if this is a heading
            is_heading = re.match(heading_pattern, line)
            
            # If we've reached the chunk size limit and this is a heading, start a new chunk
            if current_length >= self.chunk_size and is_heading:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Add the line to the current chunk
            current_chunk.append(line)
            current_length += len(line) + 1  # +1 for the newline
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _enhance_chunk(self, markdown: str) -> str:
        """
        Enhance a chunk of Markdown using Gemini 2.
        
        Args:
            markdown: Original Markdown chunk
            
        Returns:
            Enhanced Markdown chunk
        """
        prompt = self._create_enhancement_prompt(markdown)
        
        # Try to get a response with retries
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)
                
                # Extract the enhanced markdown from the response
                if hasattr(response, 'text'):
                    enhanced_markdown = response.text
                else:
                    # Handle different response formats
                    enhanced_markdown = str(response)
                
                # Clean up the response if needed
                enhanced_markdown = self._clean_response(enhanced_markdown)
                
                return enhanced_markdown
            
            except Exception as e:
                print(f"Error enhancing markdown (attempt {attempt+1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    # Wait before retrying
                    time.sleep(self.retry_delay)
                else:
                    # If all retries failed, return the original markdown
                    print("All enhancement attempts failed. Returning original markdown.")
                    return markdown
    
    def _create_enhancement_prompt(self, markdown: str) -> str:
        """
        Create a prompt for the Gemini 2 model to enhance the Markdown.
        
        Args:
            markdown: Original Markdown text
            
        Returns:
            Prompt for the Gemini 2 model
        """
        return f"""You are an expert Markdown formatter and editor. Your task is to enhance the following Markdown content to improve its formatting, structure, and readability while preserving all original information.

Please follow these guidelines:
1. Preserve all original content and meaning
2. Improve heading structure and hierarchy if needed
3. Format lists properly (bullet points and numbered lists)
4. Format code blocks with appropriate syntax highlighting
5. Ensure proper spacing between sections
6. Format tables correctly if present
7. Preserve all image references
8. Fix any obvious formatting errors
9. Do not add any new content or remove any existing content
10. Return only the enhanced Markdown without any explanations or comments

Here is the Markdown content to enhance:

```markdown
{markdown}
```

Enhanced Markdown:"""
    
    def _clean_response(self, response: str) -> str:
        """
        Clean up the response from the Gemini 2 model.
        
        Args:
            response: Response from the Gemini 2 model
            
        Returns:
            Cleaned response
        """
        # Remove any markdown code block markers that might be in the response
        response = re.sub(r'^```markdown\s*', '', response, flags=re.MULTILINE)
        response = re.sub(r'^```\s*$', '', response, flags=re.MULTILINE)
        
        # Remove any explanations or comments the model might have added
        response = re.sub(r'^Enhanced Markdown:\s*', '', response, flags=re.MULTILINE)
        
        return response.strip()
    
    def enhance_document(self, input_path: str, output_path: str) -> bool:
        """
        Enhance an entire Markdown document file.
        
        Args:
            input_path: Path to the input Markdown file
            output_path: Path to save the enhanced Markdown file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the input file
            with open(input_path, 'r', encoding='utf-8') as f:
                markdown = f.read()
            
            # Enhance the markdown
            enhanced_markdown = self.enhance_markdown(markdown)
            
            # Write the enhanced markdown to the output file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_markdown)
            
            return True
        
        except Exception as e:
            print(f"Error enhancing document: {e}")
            return False


# Simple test function
def test_ai_enhancement(markdown_path, output_path, api_key=None):
    """Test the AI enhancement functionality."""
    try:
        # Use API key from environment if not provided
        api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        
        if not api_key:
            print("Google API key is required. Set GOOGLE_API_KEY environment variable or provide as parameter.")
            return False
        
        # Initialize the AI enhancer
        enhancer = AIEnhancer(api_key=api_key)
        
        # Enhance the document
        success = enhancer.enhance_document(markdown_path, output_path)
        
        if success:
            print(f"Successfully enhanced Markdown document.")
            print(f"Enhanced document saved to: {output_path}")
            
            # Print a sample of the enhanced document
            with open(output_path, 'r', encoding='utf-8') as f:
                sample = f.read(1000)
            print("\nSample of enhanced document:")
            print(sample + "..." if len(sample) >= 1000 else sample)
        
        return success
    
    except Exception as e:
        print(f"Error testing AI enhancement: {e}")
        return False


if __name__ == "__main__":
    # This will be executed when the module is run directly
    import sys
    
    if len(sys.argv) > 2:
        markdown_path = sys.argv[1]
        output_path = sys.argv[2]
        api_key = sys.argv[3] if len(sys.argv) > 3 else None
        test_ai_enhancement(markdown_path, output_path, api_key)
    else:
        print("Please provide input and output file paths as arguments.")
