from src.text_extraction import extract_text_from_pdf
import os

def test_extraction():
    pdf_path = "data/HSC26-Bangla1st-Paper.pdf"
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    print(f"Extracting text from {pdf_path}...")
    pages = extract_text_from_pdf(pdf_path)
    
    print(f"Extracted {len(pages)} pages")
    
    # Print a sample of the first page
    if pages:
        print("\nSample of first page:")
        print(pages[0][:500])
        
        # Save the extracted text to a file for inspection
        with open("extracted_text_sample.txt", "w", encoding="utf-8") as f:
            f.write(pages[0][:1000])
        print("\nSaved first 1000 characters to extracted_text_sample.txt")

if __name__ == "__main__":
    test_extraction()