import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, Any, List, Tuple
import logging
import re
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with NLTK components"""
        try:
            # Download required NLTK data
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            self.sia = SentimentIntensityAnalyzer()
            
            # Load custom sentiment modifiers
            self.sentiment_modifiers = self._load_sentiment_modifiers()
            
            # Initialize sentiment thresholds
            self.strong_threshold = 0.6
            self.moderate_threshold = 0.3
            self.weak_threshold = 0.1
            
            # Initialize confidence thresholds
            self.high_confidence_threshold = 0.8
            self.medium_confidence_threshold = 0.5
            
            logger.info("SentimentAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing SentimentAnalyzer: {str(e)}")
            raise

    def _load_sentiment_modifiers(self) -> Dict[str, float]:
        """Load custom sentiment modifiers from JSON file"""
        try:
            modifiers_path = os.path.join(os.path.dirname(__file__), 'sentiment_modifiers.json')
            if os.path.exists(modifiers_path):
                with open(modifiers_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading sentiment modifiers: {str(e)}")
            return {}

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters but keep important punctuation
            text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text

    def _calculate_sentiment_score(self, text: str) -> Tuple[float, Dict[str, float]]:
        """Calculate sentiment score with detailed analysis"""
        try:
            # Get base sentiment scores
            scores = self.sia.polarity_scores(text)
            
            # Apply custom modifiers
            compound_score = scores['compound']
            for modifier, weight in self.sentiment_modifiers.items():
                if modifier in text.lower():
                    compound_score *= weight
            
            # Normalize score to [-1, 1] range
            compound_score = max(min(compound_score, 1.0), -1.0)
            
            return compound_score, scores
        except Exception as e:
            logger.error(f"Error calculating sentiment score: {str(e)}")
            return 0.0, {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

    def _calculate_confidence(self, text: str, sentiment_score: float) -> float:
        """Calculate confidence level in the sentiment analysis"""
        try:
            # Base confidence on text length and sentiment strength
            text_length = len(text.split())
            sentiment_strength = abs(sentiment_score)
            
            # Longer texts with stronger sentiment get higher confidence
            length_factor = min(text_length / 20, 1.0)  # Cap at 20 words
            strength_factor = sentiment_strength
            
            # Calculate final confidence
            confidence = (length_factor * 0.4 + strength_factor * 0.6)
            
            return min(max(confidence, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0

    def _determine_sentiment_category(self, score: float) -> str:
        """Determine the sentiment category based on score"""
        if score >= self.strong_threshold:
            return "STRONGLY_POSITIVE"
        elif score >= self.moderate_threshold:
            return "MODERATELY_POSITIVE"
        elif score >= self.weak_threshold:
            return "SLIGHTLY_POSITIVE"
        elif score <= -self.strong_threshold:
            return "STRONGLY_NEGATIVE"
        elif score <= -self.moderate_threshold:
            return "MODERATELY_NEGATIVE"
        elif score <= -self.weak_threshold:
            return "SLIGHTLY_NEGATIVE"
        else:
            return "NEUTRAL"

    def _determine_confidence_level(self, confidence: float) -> str:
        """Determine the confidence level based on score"""
        if confidence >= self.high_confidence_threshold:
            return "HIGH"
        elif confidence >= self.medium_confidence_threshold:
            return "MEDIUM"
        else:
            return "LOW"

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text with enhanced features"""
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Calculate sentiment score
            sentiment_score, detailed_scores = self._calculate_sentiment_score(processed_text)
            
            # Calculate confidence
            confidence = self._calculate_confidence(processed_text, sentiment_score)
            
            # Determine categories
            sentiment_category = self._determine_sentiment_category(sentiment_score)
            confidence_level = self._determine_confidence_level(confidence)
            
            # Prepare analysis results
            analysis = {
                'text': text,
                'processed_text': processed_text,
                'sentiment_score': sentiment_score,
                'sentiment_category': sentiment_category,
                'confidence': confidence,
                'confidence_level': confidence_level,
                'detailed_scores': detailed_scores,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Sentiment analysis completed for text: {text[:50]}...")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                'text': text,
                'sentiment_score': 0.0,
                'sentiment_category': 'NEUTRAL',
                'confidence': 0.0,
                'confidence_level': 'LOW',
                'detailed_scores': {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0},
                'timestamp': datetime.now().isoformat()
            } 