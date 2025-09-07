from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from preprocessing import full_preprocessing

app = Flask(__name__)

# Load your trained model and vectorizer
try:
    vectorizer = joblib.load('bow_vectorizer.pkl')
    classifier = joblib.load('bow_classifier.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    vectorizer = None
    classifier = None

@app.route('/')
def home():
    """Home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """Predict sentiment for user input"""
    try:
        # Get user input
        user_comment = request.form['comment']
        
        if not user_comment.strip():
            return render_template('result.html', 
                                 error="Please enter a comment to analyze")
        
        # Preprocess exactly like in your notebook
        original_comment = user_comment
        cleaned_comment = full_preprocessing(user_comment)
        
        # Transform using your vectorizer
        comment_vector = vectorizer.transform([cleaned_comment])
        
        # Predict
        prediction = classifier.predict(comment_vector)[0]
        prediction_proba = classifier.predict_proba(comment_vector)[0]
        
        # Get confidence scores
        negative_confidence = prediction_proba[0] * 100
        positive_confidence = prediction_proba[1] * 100
        
        # Determine sentiment
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = max(negative_confidence, positive_confidence)
        
        # Get top influential words (optional)
        feature_names = vectorizer.get_feature_names_out()
        coefficients = classifier.coef_[0]
        
        # Find words that influenced this prediction
        comment_features = comment_vector.toarray()[0]
        influential_words = []
        
        for i, (feature, coef, present) in enumerate(zip(feature_names, coefficients, comment_features)):
            if present > 0:  # Word is present in the comment
                influence_score = coef * present
                influential_words.append((feature, influence_score))
        
        # Sort by absolute influence
        influential_words.sort(key=lambda x: abs(x[1]), reverse=True)
        top_words = influential_words[:10]  # Top 10 influential words
        
        return render_template('result.html',
                             original_comment=original_comment,
                             cleaned_comment=cleaned_comment,
                             sentiment=sentiment,
                             confidence=round(confidence, 2),
                             positive_confidence=round(positive_confidence, 2),
                             negative_confidence=round(negative_confidence, 2),
                             top_words=top_words)
                             
    except Exception as e:
        return render_template('result.html', 
                             error=f"Error analyzing sentiment: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)