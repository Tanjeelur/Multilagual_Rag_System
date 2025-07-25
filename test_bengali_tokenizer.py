import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the custom Bengali tokenizer
try:
    # Try with underscore
    from multilingual_rag.src.bengali_tokenizer import bengali_sent_tokenize, bengali_word_tokenize
except ImportError:
    # Direct import as fallback
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multilingual-rag'))
    from src.bengali_tokenizer import bengali_sent_tokenize, bengali_word_tokenize

# Test Bengali sentence tokenization
print("Testing Bengali sentence tokenization...")
test_sentences = [
    "এটি একটি পরীক্ষা বাক্য।",
    "আমি বাংলায় কথা বলি। আপনি কি বাংলায় কথা বলেন?",
    "বাংলাদেশের রাজধানী ঢাকা। এটি একটি সুন্দর শহর।"
]

for i, text in enumerate(test_sentences):
    print(f"\nTest {i+1}: {text}")
    sentences = bengali_sent_tokenize(text)
    print(f"Tokenized into {len(sentences)} sentences:")
    for j, sent in enumerate(sentences):
        print(f"  {j+1}. {sent}")

# Test Bengali word tokenization
print("\n\nTesting Bengali word tokenization...")
test_text = "আমি বাংলায় কথা বলি।"
print(f"\nText: {test_text}")
words = bengali_word_tokenize(test_text)
print(f"Tokenized into {len(words)} words:")
print(words)

print("\nTokenization tests completed!")