import pytest
from src.event_filter.company_detector import CompanyDetector
from src.sentiment_model.sentiment_analyzer import SentimentAnalyzer
from src.trade_decision.trade_analyzer import TradeAnalyzer

def test_company_detection():
    detector = CompanyDetector()
    
    # Test exact match
    companies = detector.detect_companies("Apple announces new iPhone")
    assert "Apple" in companies
    assert companies["Apple"] == "AAPL"
    
    # Test possessive form
    companies = detector.detect_companies("Tesla's stock price rises")
    assert "Tesla" in companies
    assert companies["Tesla"] == "TSLA"
    
    # Test no match
    companies = detector.detect_companies("Random news headline")
    assert len(companies) == 0

def test_sentiment_analysis():
    analyzer = SentimentAnalyzer()
    
    # Test positive sentiment
    score, category = analyzer.analyze_sentiment("Great news for investors as company reports record profits")
    assert category == "POSITIVE"
    assert score > 0
    
    # Test negative sentiment
    score, category = analyzer.analyze_sentiment("Company faces major losses and layoffs")
    assert category == "NEGATIVE"
    assert score < 0
    
    # Test neutral sentiment
    score, category = analyzer.analyze_sentiment("Company announces quarterly results")
    assert category == "NEUTRAL"
    assert abs(score) < 0.2

def test_trade_decision():
    analyzer = TradeAnalyzer()
    
    # Test positive sentiment with volume spike
    sentiment_data = {
        'sentiment_score': 0.5,
        'confidence': 0.6
    }
    decision = analyzer.make_trade_decision(sentiment_data, "AAPL")
    assert decision['decision'] in ["BUY", "WATCH"]  # WATCH if no volume spike
    
    # Test negative sentiment
    sentiment_data = {
        'sentiment_score': -0.5,
        'confidence': 0.6
    }
    decision = analyzer.make_trade_decision(sentiment_data, "TSLA")
    assert decision['decision'] in ["SELL", "WATCH"]  # WATCH if no volume spike
    
    # Test weak sentiment
    sentiment_data = {
        'sentiment_score': 0.1,
        'confidence': 0.2
    }
    decision = analyzer.make_trade_decision(sentiment_data, "MSFT")
    assert decision['decision'] == "WATCH" 