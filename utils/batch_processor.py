"""
Batch Sentiment Analysis Processor
Process multiple texts efficiently and export results
"""

import pandas as pd
import sys
import os
from typing import List, Dict
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.predictor import SentimentPredictor, AIExplainer


class BatchProcessor:
    """
    Batch processing for sentiment analysis
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize batch processor
        
        Args:
            api_key: Optional Gemini API key for explanations
        """
        self.predictor = SentimentPredictor()
        self.explainer = AIExplainer(api_key=api_key)
        
    def process_texts(self, texts: List[str], 
                     include_explanations: bool = False) -> pd.DataFrame:
        """
        Process multiple texts
        
        Args:
            texts: List of texts to analyze
            include_explanations: Whether to generate AI explanations
            
        Returns:
            DataFrame with results
        """
        results = []
        
        print(f"Processing {len(texts)} texts...")
        
        for idx, text in enumerate(texts):
            print(f"  [{idx+1}/{len(texts)}] Processing...", end='\r')
            
            # Get prediction
            result = self.predictor.predict(text)
            
            # Get important words
            important_words = self.predictor.get_important_words(text, top_n=5)
            top_words = [word for word, _ in important_words[:3]]
            
            # Prepare result dict
            result_dict = {
                'text': text,
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'top_keywords': ', '.join(top_words)
            }
            
            # Add probabilities
            for sentiment, prob in result['probabilities'].items():
                result_dict[f'prob_{sentiment}'] = prob
            
            # Generate explanation if requested
            if include_explanations:
                explanation = self.explainer.generate_explanation(
                    text, 
                    result['sentiment'],
                    result['confidence'],
                    important_words
                )
                result_dict['explanation'] = explanation
            
            results.append(result_dict)
        
        print(f"\n✓ Completed processing {len(texts)} texts")
        
        return pd.DataFrame(results)
    
    def process_csv(self, input_path: str, text_column: str,
                   output_path: str = None, include_explanations: bool = False):
        """
        Process texts from CSV file
        
        Args:
            input_path: Path to input CSV
            text_column: Name of column containing text
            output_path: Path for output CSV (optional)
            include_explanations: Whether to include AI explanations
        """
        print(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")
        
        texts = df[text_column].tolist()
        
        # Process texts
        results_df = self.process_texts(texts, include_explanations)
        
        # Combine with original data
        output_df = pd.concat([df, results_df], axis=1)
        
        # Save results
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'data/sentiment_results_{timestamp}.csv'
        
        output_df.to_csv(output_path, index=False)
        print(f"✓ Results saved to {output_path}")
        
        return output_df
    
    def generate_report(self, results_df: pd.DataFrame) -> Dict:
        """
        Generate summary report from results
        
        Args:
            results_df: DataFrame with analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        report = {
            'total_texts': len(results_df),
            'sentiment_distribution': results_df['sentiment'].value_counts().to_dict(),
            'average_confidence': results_df['confidence'].mean(),
            'confidence_by_sentiment': results_df.groupby('sentiment')['confidence'].mean().to_dict()
        }
        
        return report


def main():
    """Main batch processing script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Sentiment Analysis')
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('--text-column', default='text', 
                       help='Name of column containing text (default: text)')
    parser.add_argument('--output', help='Output CSV file path')
    parser.add_argument('--explanations', action='store_true',
                       help='Include AI explanations (slower)')
    parser.add_argument('--api-key', help='Gemini API key for explanations')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BatchProcessor(api_key=args.api_key or os.getenv('GEMINI_API_KEY'))
    
    # Process CSV
    results = processor.process_csv(
        args.input,
        args.text_column,
        args.output,
        args.explanations
    )
    
    # Generate report
    report = processor.generate_report(results)
    
    print("\n" + "="*60)
    print("BATCH PROCESSING REPORT")
    print("="*60)
    print(f"Total texts processed: {report['total_texts']}")
    print(f"Average confidence: {report['average_confidence']:.2%}")
    print("\nSentiment Distribution:")
    for sentiment, count in report['sentiment_distribution'].items():
        print(f"  {sentiment}: {count} ({count/report['total_texts']*100:.1f}%)")
    print("\nAverage Confidence by Sentiment:")
    for sentiment, conf in report['confidence_by_sentiment'].items():
        print(f"  {sentiment}: {conf:.2%}")
    print("="*60)


if __name__ == "__main__":
    main()