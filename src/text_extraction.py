import fitz  # PyMuPDF
import numpy as np
import easyocr
from PIL import Image
import logging
from typing import List
import os
import tempfile
import sys
import io
import gc
import time

# Configure logging
logging.basicConfig(level=logging.INFO, filename='logs/rag_pipeline.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import pdf2image, but fall back to PyMuPDF if not available
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
    logger.info("pdf2image is available for high-quality OCR")
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image or poppler not available, falling back to PyMuPDF for image extraction")

# Check if we're on Windows and provide instructions for poppler
if sys.platform.startswith('win') and PDF2IMAGE_AVAILABLE:
    try:
        # Test if poppler is installed by trying to get version info
        from pdf2image.exceptions import PDFInfoNotInstalledError
        try:
            from pdf2image import pdfinfo_from_path
            pdfinfo_from_path("test")
        except PDFInfoNotInstalledError:
            logger.warning(
                "Poppler is not installed or not in PATH. "
                "Please download poppler for Windows from https://github.com/oschwartz10612/poppler-windows/releases/ "
                "and add the bin directory to your PATH."
            )
            PDF2IMAGE_AVAILABLE = False
    except Exception as e:
        logger.warning(f"Error checking poppler installation: {e}")
        PDF2IMAGE_AVAILABLE = False

def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """
    Extract text from a PDF file by converting all pages to images and applying OCR.
    This approach handles non-Unicode fonts better than direct text extraction.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        List[str]: List of extracted text per page.
    
    Raises:
        FileNotFoundError: If the PDF file is not found.
        Exception: For other extraction errors.
    """
    try:
        if PDF2IMAGE_AVAILABLE:
            # Use pdf2image for high-quality OCR if available
            return _extract_with_pdf2image(pdf_path)
        else:
            # Fall back to PyMuPDF if pdf2image is not available
            return _extract_with_pymupdf(pdf_path)
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        raise
    except Exception as e:
        logger.error(f"Text extraction failed: {str(e)}")
        raise

def _extract_with_pdf2image(pdf_path: str) -> List[str]:
    """Extract text using pdf2image and easyocr for best OCR quality."""
    reader = easyocr.Reader(['bn', 'en'], gpu=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Converting PDF to images with pdf2image: {pdf_path}")
        images = convert_from_path(
            pdf_path,
            dpi=200,
            output_folder=temp_dir,
            fmt='png',
            thread_count=1
        )
        pages = []
        for i, img in enumerate(images, 1):
            logger.info(f"Applying EasyOCR on page {i}")
            img_array = np.array(img)
            result = reader.readtext(img_array, detail=0, paragraph=True)
            text = '\n'.join(result)
            pages.append(text)
            img = None
            if i % 5 == 0:
                gc.collect()
                time.sleep(0.1)
    logger.info(f"Extracted {len(pages)} pages from {pdf_path} using pdf2image + easyocr")
    return pages

def _extract_with_pymupdf(pdf_path: str) -> List[str]:
    """Extract text using PyMuPDF and easyocr as fallback."""
    reader = easyocr.Reader(['bn', 'en'], gpu=True)
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, 1):
        logger.info(f"Converting page {page_num} to image with PyMuPDF")
        pix = page.get_pixmap(dpi=200)
        try:
            if hasattr(pix, 'samples'):
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            else:
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
            img_array = np.array(img)
            result = reader.readtext(img_array, detail=0, paragraph=True)
            text = '\n'.join(result)
            pages.append(text)
            logger.info(f"Applied EasyOCR on page {page_num}")
            img = None
            pix = None
            if page_num % 5 == 0:
                gc.collect()
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {str(e)}")
            pages.append(f"[OCR ERROR ON PAGE {page_num}]")
    doc.close()
    logger.info(f"Extracted {len(pages)} pages from {pdf_path} using PyMuPDF + easyocr")
    return pages