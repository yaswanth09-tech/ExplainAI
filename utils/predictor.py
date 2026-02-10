"""
Sentiment Prediction and Explanation Utilities
Handles model predictions and generates AI-powered explanations
"""

import pickle
import numpy as np
import re
from typing import Dict, List, Tuple
import os


class SentimentPredictor:
    """
    Production-ready sentiment prediction engine with explanations
    """
    
    def __init__(self, model_path='model/sentiment_model.pkl', 
                 vectorizer_path='model/vectorizer.pkl'):
        """
        Load trained model and vectorizer
        
        Args:
            model_path: Path to saved model
            vectorizer_path: Path to saved vectorizer
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
            
    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a single text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with prediction results
        """
        # Vectorize text
        text_vectorized = self.vectorizer.transform([text])
        
        # Get prediction
        prediction = self.model.predict(text_vectorized)[0]
        
        # Get probability scores
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(text_vectorized)[0]
            confidence = float(np.max(probabilities))
            
            # Create probability distribution
            prob_dict = {
                class_name: float(prob) 
                for class_name, prob in zip(self.model.classes_, probabilities)
            }
        else:
            # For models without predict_proba (like LinearSVC)
            decision = self.model.decision_function(text_vectorized)[0]
            confidence = float(np.max(np.abs(decision)))
            prob_dict = {prediction: confidence}
        
        return {
            'sentiment': prediction,
            'confidence': confidence,
            'probabilities': prob_dict,
            'text': text
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]
    
    def get_important_words(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Extract most important words for the prediction
        
        Args:
            text: Input text
            top_n: Number of top words to return
            
        Returns:
            List of (word, importance_score) tuples
        """
        # Vectorize text
        text_vectorized = self.vectorizer.transform([text])
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get feature indices with non-zero values
        non_zero_indices = text_vectorized.nonzero()[1]
        
        if len(non_zero_indices) == 0:
            return []
        
        # Get model coefficients
        if hasattr(self.model, 'coef_'):
            # For binary classification
            if len(self.model.coef_.shape) == 1:
                coef = self.model.coef_
            else:
                # Get coefficients for predicted class
                prediction = self.model.predict(text_vectorized)[0]
                class_idx = np.where(self.model.classes_ == prediction)[0][0]
                coef = self.model.coef_[class_idx]
            
            # Calculate importance scores
            importance_scores = []
            for idx in non_zero_indices:
                word = feature_names[idx]
                tfidf_score = text_vectorized[0, idx]
                model_weight = coef[idx]
                importance = abs(tfidf_score * model_weight)
                importance_scores.append((word, float(importance)))
            
            # Sort by importance and return top N
            importance_scores.sort(key=lambda x: x[1], reverse=True)
            return importance_scores[:top_n]
        
        return []


class AIExplainer:
    """
    Generates human-readable explanations using Gemini API
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize AI explainer with API key
        
        Args:
            api_key: Gemini API key (optional, can be set via environment)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
    def generate_explanation(self, text: str, sentiment: str, 
                           confidence: float, important_words: List[Tuple[str, float]] = None) -> str:
        """
        Generate AI-powered explanation for sentiment prediction
        
        Args:
            text: Original text
            sentiment: Predicted sentiment
            confidence: Prediction confidence
            important_words: List of important words and scores
            
        Returns:
            Human-readable explanation string
        """
        if not self.api_key:
            return self._generate_rule_based_explanation(text, sentiment, important_words)
        
        try:
            import google.generativeai as genai
            
            # Configure API
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # Create prompt
            words_str = ", ".join([f"'{word}'" for word, _ in (important_words or [])[:5]])
            
            prompt = f"""You are an AI sentiment analysis explainer. Explain why the following text was classified as {sentiment.upper()}.

Text: "{text}"

Key influential words: {words_str}
Confidence: {confidence:.2%}

Provide a brief, clear explanation (2-3 sentences max) that:
1. States the sentiment clearly
2. Mentions specific words or phrases that indicate this sentiment
3. Sounds natural and conversational

Do NOT use technical terms. Write as if explaining to a non-technical person."""

            # Generate explanation
            response = model.generate_content(prompt)
            explanation = response.text.strip()
            
            # Clean up the explanation
            explanation = self._clean_explanation(explanation)
            
            return explanation
            
        except Exception as e:
            print(f"AI explanation generation failed: {e}")
            return self._generate_rule_based_explanation(text, sentiment, important_words)
    
    def _clean_explanation(self, explanation: str) -> str:
        """Clean and format AI-generated explanation"""
        # Remove markdown formatting
        explanation = re.sub(r'\*\*(.*?)\*\*', r'\1', explanation)
        explanation = re.sub(r'\*(.*?)\*', r'\1', explanation)
        
        # Remove bullet points
        explanation = re.sub(r'^[\*\-]\s+', '', explanation, flags=re.MULTILINE)
        
        # Ensure proper spacing
        explanation = ' '.join(explanation.split())
        
        return explanation
    
    def _generate_rule_based_explanation(self, text: str, sentiment: str, 
                                        important_words: List[Tuple[str, float]] = None) -> str:
        """
        Generate rule-based explanation when API is unavailable
        
        Args:
            text: Original text
            sentiment: Predicted sentiment
            important_words: Important words list
            
        Returns:
            Rule-based explanation
        """
        if not important_words or len(important_words) == 0:
            return f"This text is classified as {sentiment} based on overall language patterns."
        
        # Get top 3 words
        top_words = [word for word, _ in important_words[:3]]
        words_str = ", ".join([f"'{word}'" for word in top_words])
        
        sentiment_indicators = {
            'positive': 'optimistic language and positive expressions',
            'negative': 'critical language and negative expressions',
            'neutral': 'balanced language without strong emotional indicators'
        }
        
        indicator = sentiment_indicators.get(sentiment, 'the language used')
        
        explanation = (
            f"This text expresses a {sentiment} sentiment because it contains "
            f"key words like {words_str} that indicate {indicator}. "
            f"The overall tone and word choice strongly suggest a {sentiment} perspective."
        )
        
        return explanation


def get_sentiment_color(sentiment: str) -> str:
    """
    Get color code for sentiment
    
    Args:
        sentiment: Sentiment label
        
    Returns:
        Hex color code
    """
    colors = {
        'positive': '#10b981',  # Green
        'negative': '#ef4444',  # Red
        'neutral': '#6b7280'    # Gray
    }
    return colors.get(sentiment.lower(), '#6b7280')


def get_sentiment_emoji(sentiment: str) -> str:
    """
    Get emoji for sentiment
    
    Args:
        sentiment: Sentiment label
        
    Returns:
        Emoji character
    """
    emojis = {
        'positive': '😊',
        'negative': '😞',
        'neutral': '😐'
    }
    return emojis.get(sentiment.lower(), '😐')


def format_confidence(confidence: float) -> str:
    """
    Format confidence score with color indicator
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Formatted confidence string
    """
    percentage = confidence * 100
    
    if percentage >= 90:
        level = "Very High"
        color = "#10b981"
    elif percentage >= 70:
        level = "High"
        color = "#3b82f6"
    elif percentage >= 50:
        level = "Moderate"
        color = "#f59e0b"
    else:
        level = "Low"
        color = "#ef4444"
    
    return f"{percentage:.1f}% ({level})", color