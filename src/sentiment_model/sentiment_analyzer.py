from textblob import TextBlob
from typing import Dict, Tuple, List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

class SentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            
        # Sentiment thresholds
        self.positive_threshold = 0.2
        self.negative_threshold = -0.2
        
        # Load stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Keywords that might indicate importance
        self.importance_keywords = {
            'breakthrough': 1.5,
            'record': 1.3,
            'surge': 1.4,
            'plunge': -1.4,
            'crisis': -1.5,
            'revolutionary': 1.5,
            'disaster': -1.5,
            'milestone': 1.3,
            'setback': -1.3,
            'innovation': 1.4,
            'lawsuit': -1.2,
            'partnership': 1.2,
            'acquisition': 1.2,
            'merger': 1.2,
            'bankruptcy': -1.5
        }
        
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing special characters and converting to lowercase.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()
        
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of important keywords
        """
        words = word_tokenize(text.lower())
        return [word for word in words if word not in self.stop_words]
        
    def _calculate_keyword_impact(self, keywords: List[str]) -> float:
        """
        Calculate the impact of important keywords on sentiment.
        
        Args:
            keywords (List[str]): List of keywords
            
        Returns:
            float: Keyword impact score
        """
        impact = 0.0
        for keyword in keywords:
            if keyword in self.importance_keywords:
                impact += self.importance_keywords[keyword]
        return impact
        
    def analyze_sentiment(self, text: str) -> Tuple[float, str, Dict]:
        """
        Analyze the sentiment of a given text using TextBlob and custom analysis.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Tuple[float, str, Dict]: A tuple containing the sentiment score, category, and detailed analysis
        """
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Create TextBlob object
        blob = TextBlob(processed_text)
        
        # Get the base polarity score (-1 to 1)
        base_sentiment = blob.sentiment.polarity
        
        # Extract and analyze keywords
        keywords = self._extract_keywords(processed_text)
        keyword_impact = self._calculate_keyword_impact(keywords)
        
        # Calculate final sentiment score
        sentiment_score = base_sentiment + (keyword_impact * 0.2)  # Weight keyword impact
        sentiment_score = max(min(sentiment_score, 1.0), -1.0)  # Clamp between -1 and 1
        
        # Categorize the sentiment
        if sentiment_score > self.positive_threshold:
            category = "POSITIVE"
        elif sentiment_score < self.negative_threshold:
            category = "NEGATIVE"
        else:
            category = "NEUTRAL"
            
        # Prepare detailed analysis
        analysis = {
            'base_sentiment': base_sentiment,
            'keyword_impact': keyword_impact,
            'important_keywords': [k for k in keywords if k in self.importance_keywords],
            'subjectivity': blob.sentiment.subjectivity,
            'word_count': len(keywords)
        }
            
        return sentiment_score, category, analysis
    
    def analyze_headline(self, headline: str, company: str) -> Dict:
        """
        Analyze the sentiment of a headline specifically for a mentioned company.
        
        Args:
            headline (str): The news headline
            company (str): The company name to analyze sentiment for
            
        Returns:
            Dict: Dictionary containing sentiment analysis results
        """
        sentiment_score, category, analysis = self.analyze_sentiment(headline)
        
        return {
            'headline': headline,
            'company': company,
            'sentiment_score': sentiment_score,
            'sentiment_category': category,
            'confidence': abs(sentiment_score) * (1 + analysis['subjectivity']),  # Higher subjectivity increases confidence
            'analysis': analysis
        } 