import nltk
import os
import sys

# Create directory for NLTK data if it doesn't exist
local_nltk_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(local_nltk_dir, exist_ok=True)

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Set NLTK data path to include local directory
nltk.data.path.append(local_nltk_dir)

# Download required NLTK data packages
print("Downloading NLTK data packages...")

# General tokenization data
nltk.download('punkt', download_dir=local_nltk_dir)

# Additional useful resources
nltk.download('stopwords', download_dir=local_nltk_dir)
nltk.download('wordnet', download_dir=local_nltk_dir)

print("\nNLTK data downloaded successfully!")
print(f"NLTK data is stored in: {local_nltk_dir}")
print("NLTK data path includes:")
for path in nltk.data.path:
    print(f"  - {path}")

# Test Bengali tokenization using our custom tokenizer
print("\nTesting Bengali tokenization using custom tokenizer...")

try:
    # Try to import our custom Bengali tokenizer
    sys.path.append(os.path.join(os.getcwd(), 'multilingual-rag'))
    from src.bengali_tokenizer import bengali_sent_tokenize, bengali_word_tokenize
    
    # Test sentence tokenization
    test_text = "এটি একটি পরীক্ষা বাক্য। আমি বাংলায় কথা বলি।"
    sentences = bengali_sent_tokenize(test_text)
    print(f"\nBengali sentence tokenization test successful!")
    print(f"Input: {test_text}")
    print(f"Tokenized into {len(sentences)} sentences:")
    for i, sent in enumerate(sentences):
        print(f"  {i+1}. {sent}")
    
    # Test word tokenization
    words = bengali_word_tokenize(sentences[0])
    print(f"\nBengali word tokenization test successful!")
    print(f"Input: {sentences[0]}")
    print(f"Tokenized into {len(words)} words: {words}")
    
    print("\nAll tests completed successfully!")
    print("The system is now ready to process Bengali text.")
except ImportError as e:
    print(f"\nCould not import custom Bengali tokenizer: {e}")
    print("Please ensure the multilingual-rag/src/bengali_tokenizer.py file exists.")
except Exception as e:
    print(f"\nUnexpected error during Bengali tokenization test: {e}")
    print("The system may still work with English text, but Bengali text processing might have issues.")