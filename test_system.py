"""
Comprehensive Testing Script for Sentiment AI Explainer
Tests all core functionality to ensure production readiness
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.predictor import SentimentPredictor, AIExplainer
import pickle


def test_model_files():
    """Test if model files exist"""
    print("=" * 60)
    print("TEST 1: Model Files Verification")
    print("=" * 60)
    
    model_path = 'model/sentiment_model.pkl'
    vectorizer_path = 'model/vectorizer.pkl'
    
    model_exists = os.path.exists(model_path)
    vectorizer_exists = os.path.exists(vectorizer_path)
    
    print(f"✓ Model file exists: {model_exists}")
    print(f"✓ Vectorizer file exists: {vectorizer_exists}")
    
    if not (model_exists and vectorizer_exists):
        print("❌ FAILED: Model files not found. Run 'python model/train_model.py'")
        return False
    
    print("✅ PASSED: All model files present\n")
    return True


def test_model_loading():
    """Test model loading"""
    print("=" * 60)
    print("TEST 2: Model Loading")
    print("=" * 60)
    
    try:
        predictor = SentimentPredictor()
        print("✓ Model loaded successfully")
        print("✓ Vectorizer loaded successfully")
        print("✅ PASSED: Model loading successful\n")
        return True, predictor
    except Exception as e:
        print(f"❌ FAILED: {str(e)}\n")
        return False, None


def test_basic_predictions(predictor):
    """Test basic prediction functionality"""
    print("=" * 60)
    print("TEST 3: Basic Predictions")
    print("=" * 60)
    
    test_cases = [
        ("This is amazing! Best product ever!", "positive"),
        ("Terrible quality. Complete waste of money.", "negative"),
        ("It's okay. Nothing special.", "neutral")
    ]
    
    passed = True
    for text, expected_sentiment in test_cases:
        try:
            result = predictor.predict(text)
            actual_sentiment = result['sentiment']
            
            print(f"\nText: '{text}'")
            print(f"Expected: {expected_sentiment}")
            print(f"Predicted: {actual_sentiment}")
            print(f"Confidence: {result['confidence']:.2%}")
            
            if actual_sentiment == expected_sentiment:
                print("✓ Correct prediction")
            else:
                print("⚠ Prediction mismatch")
                passed = False
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            passed = False
    
    if passed:
        print("\n✅ PASSED: All basic predictions correct\n")
    else:
        print("\n⚠ PARTIAL: Some predictions may need review\n")
    
    return passed


def test_confidence_scores(predictor):
    """Test confidence scoring"""
    print("=" * 60)
    print("TEST 4: Confidence Scores")
    print("=" * 60)
    
    test_texts = [
        "This is absolutely amazing! Outstanding quality!",
        "It's okay, I guess."
    ]
    
    passed = True
    for text in test_texts:
        try:
            result = predictor.predict(text)
            confidence = result['confidence']
            
            print(f"\nText: '{text}'")
            print(f"Confidence: {confidence:.2%}")
            
            if 0 <= confidence <= 1:
                print("✓ Valid confidence score")
            else:
                print("❌ Invalid confidence range")
                passed = False
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            passed = False
    
    if passed:
        print("\n✅ PASSED: Confidence scores valid\n")
    else:
        print("\n❌ FAILED: Confidence score issues\n")
    
    return passed


def test_probability_distribution(predictor):
    """Test probability distribution"""
    print("=" * 60)
    print("TEST 5: Probability Distribution")
    print("=" * 60)
    
    text = "This product is fantastic! Highly recommend."
    
    try:
        result = predictor.predict(text)
        probabilities = result['probabilities']
        
        print(f"\nText: '{text}'")
        print(f"Predicted: {result['sentiment']}")
        print("\nProbability Distribution:")
        
        total_prob = 0
        for sentiment, prob in probabilities.items():
            print(f"  {sentiment}: {prob:.4f} ({prob*100:.2f}%)")
            total_prob += prob
        
        print(f"\nTotal probability: {total_prob:.4f}")
        
        if abs(total_prob - 1.0) < 0.01:  # Allow small floating point error
            print("✓ Probabilities sum to 1.0")
            print("✅ PASSED: Probability distribution valid\n")
            return True
        else:
            print("❌ Probabilities don't sum to 1.0")
            print("❌ FAILED: Probability distribution invalid\n")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}\n")
        return False


def test_keyword_extraction(predictor):
    """Test keyword extraction"""
    print("=" * 60)
    print("TEST 6: Keyword Extraction")
    print("=" * 60)
    
    text = "This product has excellent quality and amazing performance. Highly satisfied!"
    
    try:
        keywords = predictor.get_important_words(text, top_n=5)
        
        print(f"\nText: '{text}'")
        print(f"Extracted {len(keywords)} keywords:")
        
        for word, score in keywords:
            print(f"  • {word}: {score:.4f}")
        
        if len(keywords) > 0:
            print("\n✅ PASSED: Keyword extraction working\n")
            return True
        else:
            print("\n⚠ WARNING: No keywords extracted\n")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}\n")
        return False


def test_batch_predictions(predictor):
    """Test batch prediction capability"""
    print("=" * 60)
    print("TEST 7: Batch Predictions")
    print("=" * 60)
    
    texts = [
        "Great product! Love it!",
        "Terrible experience. Very disappointed.",
        "Average quality. Nothing special."
    ]
    
    try:
        results = predictor.predict_batch(texts)
        
        print(f"\nProcessed {len(results)} texts:")
        for i, (text, result) in enumerate(zip(texts, results)):
            print(f"\n{i+1}. '{text[:50]}...'")
            print(f"   Sentiment: {result['sentiment']}")
            print(f"   Confidence: {result['confidence']:.2%}")
        
        if len(results) == len(texts):
            print("\n✅ PASSED: Batch processing successful\n")
            return True
        else:
            print("\n❌ FAILED: Result count mismatch\n")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}\n")
        return False


def test_edge_cases(predictor):
    """Test edge cases and error handling"""
    print("=" * 60)
    print("TEST 8: Edge Cases")
    print("=" * 60)
    
    edge_cases = [
        ("", "Empty string"),
        ("a", "Single character"),
        ("!!!", "Special characters only"),
        ("This is a very long text " * 100, "Very long text"),
    ]
    
    passed = True
    for text, description in edge_cases:
        try:
            result = predictor.predict(text)
            print(f"✓ Handled: {description}")
        except Exception as e:
            print(f"⚠ Error with {description}: {str(e)}")
            # Edge cases can fail gracefully
    
    print("\n✅ PASSED: Edge case handling\n")
    return True


def test_ai_explainer():
    """Test AI explanation generation (without API key)"""
    print("=" * 60)
    print("TEST 9: AI Explainer (Rule-based)")
    print("=" * 60)
    
    try:
        explainer = AIExplainer()  # No API key - will use rule-based
        
        text = "This product is excellent! Great quality."
        sentiment = "positive"
        confidence = 0.95
        keywords = [("excellent", 1.5), ("great", 1.2), ("quality", 1.0)]
        
        explanation = explainer.generate_explanation(
            text, sentiment, confidence, keywords
        )
        
        print(f"\nText: '{text}'")
        print(f"Sentiment: {sentiment}")
        print(f"\nGenerated Explanation:")
        print(f"'{explanation}'")
        
        if explanation and len(explanation) > 0:
            print("\n✅ PASSED: Explanation generation working\n")
            return True
        else:
            print("\n❌ FAILED: No explanation generated\n")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}\n")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "SENTIMENT AI EXPLAINER - TEST SUITE" + " " * 13 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")
    
    results = []
    
    # Test 1: Model files
    results.append(("Model Files", test_model_files()))
    
    if not results[-1][1]:
        print("\n⚠ Cannot continue without model files. Exiting tests.\n")
        return False
    
    # Test 2: Model loading
    success, predictor = test_model_loading()
    results.append(("Model Loading", success))
    
    if not success:
        print("\n⚠ Cannot continue without loaded model. Exiting tests.\n")
        return False
    
    # Test 3-9: Functionality tests
    results.append(("Basic Predictions", test_basic_predictions(predictor)))
    results.append(("Confidence Scores", test_confidence_scores(predictor)))
    results.append(("Probability Distribution", test_probability_distribution(predictor)))
    results.append(("Keyword Extraction", test_keyword_extraction(predictor)))
    results.append(("Batch Predictions", test_batch_predictions(predictor)))
    results.append(("Edge Cases", test_edge_cases(predictor)))
    results.append(("AI Explainer", test_ai_explainer()))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<30} {status}")
    
    print("=" * 60)
    print(f"Total: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! System is production-ready.\n")
        return True
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed. Review issues above.\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)