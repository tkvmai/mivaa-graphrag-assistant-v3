import nltk

try:
    # Attempt to find the punkt tokenizer models
    # This is just to check if it's already available in a known path
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' resource already seems to be downloaded.")
except LookupError:
    # If not found, download it
    print("NLTK 'punkt' resource not found. Downloading...")
    nltk.download('punkt')
    print("NLTK 'punkt' resource downloaded successfully.")

# Optional: Verify by using it
try:
    text = "This is a sample sentence. This is another one."
    sentences = nltk.sent_tokenize(text)
    print("\nVerification successful. Example tokenization:")
    print(sentences)
except Exception as e:
    print(f"\nVerification failed. Error: {e}")
    print("Please ensure the download completed without errors and check your environment.")