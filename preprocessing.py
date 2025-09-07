import re
import string
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (run once)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize once
stop_words = set(stopwords.words('english'))
stop_words.remove("not")  # Keep "not" for sentiment
lemmatizer = WordNetLemmatizer()

def expand_nt_contractions(text):
    """Expand n't contractions"""
    return re.sub(r"(\b\w+)n['']t\b", r"\1 not", text)

def preprocess_text(text):
    """Clean and preprocess text - exactly from your notebook"""
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # 3. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    
    # 4. Remove emojis and special characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # 5. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 6. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 7. Expand nt Contractions
    text = expand_nt_contractions(text)
    
    # 8. Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

def normalize_para(text):
    """Lemmatize text - exactly from your notebook"""
    words = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word, pos="v") for word in words]
    return " ".join(lemmas)

def full_preprocessing(text):
    """Complete preprocessing pipeline"""
    cleaned = preprocess_text(text)
    normalized = normalize_para(cleaned)
    return normalized