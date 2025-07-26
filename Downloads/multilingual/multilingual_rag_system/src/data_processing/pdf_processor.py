"""
PDF text extraction module using EasyOCR for Bengali documents.

This module handles PDF processing with OCR capabilities specifically
optimized for Bengali text recognition and extraction.
"""

import easyocr
import cv2
import numpy as np
from pdf2image import convert_from_path
from typing import List, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles PDF text extraction using EasyOCR for multilingual documents.
    
    This class is specifically designed to handle Bengali literature PDFs
    with advanced OCR preprocessing for better text recognition accuracy.
    """
    
    def __init__(self, languages: List[str] = ['en', 'bn'], use_gpu: bool = True):
        """
        Initialize the PDF processor with OCR capabilities.
        
        Args:
            languages (List[str]): List of language codes for OCR recognition
            use_gpu (bool): Whether to use GPU acceleration for OCR
        """
        self.languages = languages
        self.use_gpu = use_gpu
        self.reader = None
        self._initialize_ocr_reader()
        
    def _initialize_ocr_reader(self) -> None:
        """Initialize the EasyOCR reader with specified languages."""
        try:
            logger.info(f"Initializing OCR reader for languages: {self.languages}")
            self.reader = easyocr.Reader(
                self.languages, 
                gpu=self.use_gpu,
                verbose=False
            )
            logger.info("OCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR reader: {str(e)}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str, dpi: int = 300) -> List[str]:
        """
        Extract text from PDF using OCR with preprocessing.
        
        Args:
            pdf_path (str): Path to the PDF file
            dpi (int): DPI for PDF to image conversion (higher = better quality)
            
        Returns:
            List[str]: List of extracted text from each page
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        try:
            logger.info(f"Starting PDF text extraction from: {pdf_path}")
            
            # Convert PDF pages to images
            pages = convert_from_path(pdf_path, dpi=dpi)
            extracted_texts = []
            
            logger.info(f"Processing {len(pages)} pages from PDF")
            
            for page_num, page in enumerate(pages, 1):
                logger.info(f"Processing page {page_num}/{len(pages)}")
                
                # Convert PIL image to numpy array
                page_array = np.array(page)
                
                # Preprocess image for better OCR results
                processed_image = self._preprocess_image(page_array)
                
                # Extract text using EasyOCR
                ocr_results = self.reader.readtext(processed_image)
                
                # Combine all text from the page with confidence filtering
                page_text = self._extract_text_from_ocr_results(ocr_results)
                
                if page_text.strip():  # Only add non-empty pages
                    extracted_texts.append(page_text)
                    logger.debug(f"Page {page_num}: Extracted {len(page_text)} characters")
                else:
                    logger.warning(f"Page {page_num}: No text extracted")
                
            logger.info(f"Successfully extracted text from {len(extracted_texts)} pages")
            return extracted_texts
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results on Bengali text.
        
        Args:
            image (np.ndarray): Input image as numpy array
            
        Returns:
            np.ndarray: Preprocessed image optimized for OCR
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply adaptive thresholding for better text clarity
            thresh = cv2.adaptiveThreshold(
                blurred, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11, 
                2
            )
            
            # Optional: Morphological operations to clean up the image
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}, using original image")
            return image
    
    def _extract_text_from_ocr_results(
        self, 
        ocr_results: List[Tuple], 
        confidence_threshold: float = 0.3
    ) -> str:
        """
        Extract and filter text from OCR results based on confidence.
        
        Args:
            ocr_results (List[Tuple]): OCR results from EasyOCR
            confidence_threshold (float): Minimum confidence score to include text
            
        Returns:
            str: Filtered and combined text
        """
        filtered_texts = []
        
        for result in ocr_results:
            if len(result) >= 3:
                bbox, text, confidence = result[0], result[1], result[2]
                
                # Filter by confidence and text length
                if confidence >= confidence_threshold and len(text.strip()) > 1:
                    filtered_texts.append(text.strip())
        
        return " ".join(filtered_texts)
    
    def extract_text_from_images(self, image_paths: List[str]) -> List[str]:
        """
        Extract text from a list of image files.
        
        Args:
            image_paths (List[str]): List of paths to image files
            
        Returns:
            List[str]: Extracted text from each image
        """
        extracted_texts = []
        
        for image_path in image_paths:
            try:
                logger.info(f"Processing image: {image_path}")
                
                # Read image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not read image: {image_path}")
                    continue
                
                # Preprocess and extract text
                processed_image = self._preprocess_image(image)
                ocr_results = self.reader.readtext(processed_image)
                text = self._extract_text_from_ocr_results(ocr_results)
                
                extracted_texts.append(text)
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {str(e)}")
                extracted_texts.append("")
        
        return extracted_texts
