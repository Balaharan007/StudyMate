import nltk
from nltk.tokenize import word_tokenize

# Download only required resource
nltk.download('punkt')

def tokenize_text(text):
    return word_tokenize(text)
