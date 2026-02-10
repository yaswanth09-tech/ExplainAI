"""
Sentiment Analysis Model Training Pipeline
Trains a production-ready classifier with comprehensive evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
import pickle
import warnings
warnings.filterwarnings('ignore')


class SentimentModelTrainer:
    """
    Production-grade sentiment analysis model trainer
    Supports multiple algorithms and comprehensive evaluation
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize the trainer with vectorization parameters
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: N-gram range for feature extraction
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        self.model = None
        self.model_type = None
        
    def prepare_data(self, texts, labels):
        """
        Prepare and split data for training
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels
            
        Returns:
            Train-test split of features and labels
        """
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Vectorize text
        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)
        
        return X_train, X_test, y_train, y_test, X_test_text
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        print("Training Logistic Regression model...")
        self.model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        self.model_type = "Logistic Regression"
        
    def train_linear_svm(self, X_train, y_train):
        """Train Linear SVM model"""
        print("Training Linear SVM model...")
        self.model = LinearSVC(
            max_iter=1000,
            C=1.0,
            random_state=42,
            dual=False
        )
        self.model.fit(X_train, y_train)
        self.model_type = "Linear SVM"
        
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return metrics, y_pred
    
    def get_top_features(self, n_features=10):
        """
        Extract top features for each sentiment class
        
        Args:
            n_features: Number of top features to extract
            
        Returns:
            Dictionary of top features per class
        """
        feature_names = self.vectorizer.get_feature_names_out()
        
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            
            top_features = {}
            classes = self.model.classes_
            
            for idx, sentiment in enumerate(classes):
                if len(coef.shape) > 1:
                    top_indices = np.argsort(coef[idx])[-n_features:][::-1]
                else:
                    top_indices = np.argsort(coef)[-n_features:][::-1]
                    
                top_features[sentiment] = [
                    (feature_names[i], coef[idx][i] if len(coef.shape) > 1 else coef[i])
                    for i in top_indices
                ]
            
            return top_features
        return {}
    
    def save_model(self, model_path='model/sentiment_model.pkl', 
                   vectorizer_path='model/vectorizer.pkl'):
        """Save trained model and vectorizer"""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"✓ Model saved to {model_path}")
        print(f"✓ Vectorizer saved to {vectorizer_path}")


def create_sample_dataset():
    """
    Create a sample dataset for demonstration
    In production, replace with your actual dataset
    """
    # Sample reviews - expanded dataset
    positive_reviews = [
        "This product is absolutely amazing! Best purchase ever.",
        "Excellent quality and fast shipping. Highly recommend!",
        "Love it! Exceeded all my expectations.",
        "Outstanding service and great value for money.",
        "Perfect! Exactly what I was looking for.",
        "Fantastic product, will definitely buy again.",
        "Super happy with this purchase. Five stars!",
        "Incredible quality and excellent customer support.",
        "Best product in its category. Totally worth it!",
        "Amazing! Fast delivery and great packaging.",
        "Wonderful experience from start to finish.",
        "Top-notch quality and reasonable price.",
        "Exceeded expectations in every way possible.",
        "Brilliant product! Highly satisfied with my purchase.",
        "Outstanding! Would recommend to everyone.",
    ] * 20  # Multiply for larger dataset
    
    negative_reviews = [
        "Terrible quality. Complete waste of money.",
        "Very disappointed. Product broke after one use.",
        "Awful experience. Would not recommend to anyone.",
        "Poor quality and terrible customer service.",
        "Horrible! Nothing like the description.",
        "Worst purchase I've ever made. Totally useless.",
        "Completely dissatisfied. Want my money back.",
        "Bad quality and slow shipping. Very unhappy.",
        "Don't buy this! Total disappointment.",
        "Cheap materials and poor craftsmanship.",
        "Regret buying this. Absolute waste.",
        "Terrible product. Broke immediately.",
        "Very poor quality. Not worth the price.",
        "Disappointing in every aspect. Avoid!",
        "Worst experience ever. Never again.",
    ] * 20
    
    neutral_reviews = [
        "It's okay. Nothing special but does the job.",
        "Average product. Met basic expectations.",
        "Decent quality for the price.",
        "It works fine. Not great, not terrible.",
        "Acceptable product. Could be better.",
        "Fair quality. Gets the job done.",
        "Mediocre. Nothing to write home about.",
        "It's alright. Standard quality.",
        "Okay for the price. Nothing more.",
        "Average experience overall.",
    ] * 30
    
    # Combine data
    texts = positive_reviews + negative_reviews + neutral_reviews
    labels = (['positive'] * len(positive_reviews) + 
              ['negative'] * len(negative_reviews) + 
              ['neutral'] * len(neutral_reviews))
    
    return texts, labels


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("SENTIMENT ANALYSIS MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Load or create dataset
    print("\n📊 Loading dataset...")
    texts, labels = create_sample_dataset()
    print(f"✓ Loaded {len(texts)} samples")
    print(f"  - Positive: {labels.count('positive')}")
    print(f"  - Negative: {labels.count('negative')}")
    print(f"  - Neutral: {labels.count('neutral')}")
    
    # Initialize trainer
    trainer = SentimentModelTrainer(max_features=5000, ngram_range=(1, 2))
    
    # Prepare data
    print("\n🔧 Preparing data...")
    X_train, X_test, y_train, y_test, X_test_text = trainer.prepare_data(texts, labels)
    print(f"✓ Training samples: {X_train.shape[0]}")
    print(f"✓ Testing samples: {X_test.shape[0]}")
    print(f"✓ Features extracted: {X_train.shape[1]}")
    
    # Train model - Logistic Regression (best for probability estimates)
    print("\n🤖 Training model...")
    trainer.train_logistic_regression(X_train, y_train)
    
    # Evaluate
    print("\n📈 Evaluating model performance...")
    metrics, y_pred = trainer.evaluate_model(X_test, y_test)
    
    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}")
    print(metrics['confusion_matrix'])
    
    print(f"\n{'='*60}")
    print("DETAILED CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(metrics['classification_report'])
    
    # Top features
    print(f"\n{'='*60}")
    print("TOP PREDICTIVE FEATURES")
    print(f"{'='*60}")
    top_features = trainer.get_top_features(n_features=10)
    for sentiment, features in top_features.items():
        print(f"\n{sentiment.upper()}:")
        for word, score in features[:5]:
            print(f"  • {word}: {score:.4f}")
    
    # Save model
    print(f"\n{'='*60}")
    print("SAVING MODEL")
    print(f"{'='*60}")
    trainer.save_model()
    
    print("\n✅ Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()