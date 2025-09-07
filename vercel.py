import streamlit as st
import joblib
import numpy as np
import re
import string
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk

# Ensure punkt and punkt_tab are available
nltk.download("punkt")
nltk.download("punkt_tab")


# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        return True
    except:
        return False

# Initialize NLTK components
@st.cache_resource
def init_nltk():
    download_nltk_data()
    stop_words = set(stopwords.words('english'))
    stop_words.remove("not")
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

# Load models
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load('bow_vectorizer.pkl')
        classifier = joblib.load('bow_classifier.pkl')
        return vectorizer, classifier
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please upload bow_vectorizer.pkl and bow_classifier.pkl files")
        return None, None

# Preprocessing functions
def expand_nt_contractions(text):
    return re.sub(r"(\b\w+)n['']t\b", r"\1 not", text)

def preprocess_text(text, stop_words):
    # Lowercasing
    text = text.lower()
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    
    # Remove emojis and special characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Expand nt Contractions
    text = expand_nt_contractions(text)
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

def normalize_para(text, lemmatizer):
    words = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word, pos="v") for word in words]
    return " ".join(lemmas)

def full_preprocessing(text, stop_words, lemmatizer):
    cleaned = preprocess_text(text, stop_words)
    normalized = normalize_para(cleaned, lemmatizer)
    return normalized

def get_top_features(text, vectorizer, classifier, top_n=10):
    """Get top influential words for the prediction"""
    try:
        # Transform text
        text_vector = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()
        coefficients = classifier.coef_[0]
        
        # Get features present in text
        text_features = text_vector.toarray()[0]
        influential_words = []
        
        for i, (feature, coef, present) in enumerate(zip(feature_names, coefficients, text_features)):
            if present > 0:
                influence_score = coef * present
                influential_words.append((feature, influence_score))
        
        # Sort by absolute influence
        influential_words.sort(key=lambda x: abs(x[1]), reverse=True)
        return influential_words[:top_n]
    except:
        return []

# Main Streamlit App
def main():
    # Page config
    st.set_page_config(
        page_title="🎭 Sentiment Analysis Tool",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize components
    stop_words, lemmatizer = init_nltk()
    vectorizer, classifier = load_models()
    
    # App header
    st.title("🎭 Sentiment Analysis Tool")
    st.markdown("### Analyze the emotional tone of any text using AI")
    st.markdown("---")
    
    # Sidebar info
    with st.sidebar:
        st.header("📊 Model Information")
        st.info("""
        **Model Type**: Bag of Words + Logistic Regression
        
        **Training Data**: 10,000 IMDB movie reviews
        
        **Accuracy**: 87% on test set
        
        **Features**: 16,174 unique words
        """)
        
        st.header("🔧 How it Works")
        st.markdown("""
        1. **Text Cleaning**: Removes HTML, URLs, punctuation
        2. **Normalization**: Lemmatization and stopword removal
        3. **Vectorization**: Converts to numerical features
        4. **Prediction**: Uses trained model to classify sentiment
        """)
    
    # Main content
    if vectorizer is None or classifier is None:
        st.error("⚠️ Models not loaded. Please ensure model files are available.")
        st.stop()
    
    # Input section
    st.header("📝 Enter Your Text")
    user_input = st.text_area(
        "Type or paste your comment/review here:",
        height=150,
        placeholder="Enter any text here... (movie review, product feedback, social media post, etc.)"
    )
    
    # Prediction button
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                try:
                    # Preprocess text
                    original_text = user_input
                    processed_text = full_preprocessing(user_input, stop_words, lemmatizer)
                    
                    # Make prediction
                    text_vector = vectorizer.transform([processed_text])
                    prediction = classifier.predict(text_vector)[0]
                    prediction_proba = classifier.predict_proba(text_vector)[0]
                    
                    # Calculate confidence scores
                    negative_confidence = prediction_proba[0] * 100
                    positive_confidence = prediction_proba[1] * 100
                    
                    # Determine sentiment
                    sentiment = "Positive 😊" if prediction == 1 else "Negative 😔"
                    confidence = max(negative_confidence, positive_confidence)
                    
                    # Display results
                    st.markdown("---")
                    st.header("📊 Analysis Results")
                    
                    # Main result
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if prediction == 1:
                            st.success(f"**Sentiment**: {sentiment}")
                        else:
                            st.error(f"**Sentiment**: {sentiment}")
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    with col3:
                        if prediction == 1:
                            st.metric("Positivity", f"{positive_confidence:.1f}%")
                        else:
                            st.metric("Negativity", f"{negative_confidence:.1f}%")
                    
                    # Confidence breakdown
                    st.subheader("📈 Confidence Breakdown")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.progress(positive_confidence/100)
                        st.write(f"**Positive**: {positive_confidence:.2f}%")
                    
                    with col2:
                        st.progress(negative_confidence/100)
                        st.write(f"**Negative**: {negative_confidence:.2f}%")
                    
                    # Text processing details
                    with st.expander("🔍 View Processing Details"):
                        st.subheader("Original Text")
                        st.write(f'"{original_text}"')
                        
                        st.subheader("After Preprocessing")
                        st.code(processed_text)
                        st.caption("This shows your text after cleaning, removing stopwords, and lemmatization")
                    
                    # Top influential words
                    top_words = get_top_features(processed_text, vectorizer, classifier)
                    
                    if top_words:
                        with st.expander("📝 Most Influential Words"):
                            st.subheader("Words that influenced this prediction:")
                            
                            # Create DataFrame for better display
                            words_df = pd.DataFrame(top_words, columns=['Word', 'Influence Score'])
                            words_df['Impact'] = words_df['Influence Score'].apply(
                                lambda x: 'Positive' if x > 0 else 'Negative'
                            )
                            words_df['Influence Score'] = words_df['Influence Score'].round(4)
                            
                            # Display as colored metrics
                            cols = st.columns(5)
                            for i, (word, score, impact) in enumerate(words_df.values[:10]):
                                with cols[i % 5]:
                                    if impact == 'Positive':
                                        st.success(f"**{word}**\n+{abs(score):.3f}")
                                    else:
                                        st.error(f"**{word}**\n-{abs(score):.3f}")
                            
                            st.caption("Green = pushes toward positive, Red = pushes toward negative")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        else:
            st.warning("⚠️ Please enter some text to analyze!")
    
    # Example texts
    st.markdown("---")
    st.header("💡 Try These Examples")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📽️ Positive Movie Review", use_container_width=True):
            st.session_state['example_text'] = "This movie was absolutely fantastic! The acting was superb, the plot was engaging, and the cinematography was breathtaking. I loved every minute of it and would definitely recommend it to everyone. A true masterpiece!"
    
    with col2:
        if st.button("📱 Negative Product Review", use_container_width=True):
            st.session_state['example_text'] = "I'm really disappointed with this product. The quality is poor, it broke after just a few days, and customer service was unhelpful. Definitely not worth the money. I regret buying this."
    
    # Load example text if selected
    if 'example_text' in st.session_state:
        st.text_area("Example text loaded:", value=st.session_state['example_text'], key="example_display")
        if st.button("🔄 Use This Example", use_container_width=True):
            st.rerun()

if __name__ == "__main__":
    main()
