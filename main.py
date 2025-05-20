import logging
from datetime import datetime
from typing import Dict, List, Optional
import json
import os
from src.event_filter.company_detector import CompanyDetector
from src.event_filter.sentiment_analyzer import SentimentAnalyzer
from src.trade_decision.trade_analyzer import TradeAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsSignalTrader:
    def __init__(self):
        """Initialize the trading bot with all components"""
        try:
            self.company_detector = CompanyDetector()
            self.sentiment_analyzer = SentimentAnalyzer()
            self.trade_analyzer = TradeAnalyzer()
            
            # Initialize results storage
            self.results_dir = 'results'
            os.makedirs(self.results_dir, exist_ok=True)
            
            logger.info("NewsSignalTrader initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NewsSignalTrader: {str(e)}")
            raise

    def process_headline(self, headline: str) -> List[Dict]:
        """Process a news headline and make trading decisions"""
        try:
            # Detect companies
            companies = self.company_detector.detect_companies(headline)
            if not companies:
                logger.info(f"No relevant companies detected in headline: {headline}")
                return []

            # Analyze sentiment
            sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(headline)
            
            # Make trading decisions for each company
            decisions = []
            for company in companies:
                decision = self.trade_analyzer.make_trade_decision(
                    sentiment_analysis,
                    company['ticker']
                )
                decision['company'] = company['name']
                decision['headline'] = headline
                decisions.append(decision)
                
            logger.info(f"Processed headline: {headline}")
            return decisions
            
        except Exception as e:
            logger.error(f"Error processing headline: {str(e)}")
            return []

    def save_results(self, decisions: List[Dict], filename: Optional[str] = None) -> str:
        """Save trading decisions to a JSON file"""
        try:
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'trading_decisions_{timestamp}.json'
            
            filepath = os.path.join(self.results_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(decisions, f, indent=2)
                
            logger.info(f"Results saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return ""

    def run_analysis(self, headlines: List[str]) -> List[Dict]:
        """Run analysis on multiple headlines"""
        try:
            all_decisions = []
            
            for headline in headlines:
                decisions = self.process_headline(headline)
                all_decisions.extend(decisions)
                
            # Save results
            self.save_results(all_decisions)
            
            return all_decisions
            
        except Exception as e:
            logger.error(f"Error running analysis: {str(e)}")
            return []

def main():
    """Main function to run the trading bot"""
    try:
        # Initialize the trading bot
        trader = NewsSignalTrader()
        
        # Example headlines
        headlines = [
            "Apple announces new iPhone with breakthrough AI features",
            "Tesla stock surges after record quarterly deliveries",
            "Microsoft acquires AI startup for $10 billion",
            "JPMorgan warns of potential market correction",
            "Amazon expands into healthcare with new acquisition",
            "Meta faces regulatory scrutiny over data practices",
            "Visa and PayPal announce strategic partnership for digital payments",
            "Netflix announces price increase amid strong subscriber growth",
            "Disney+ surpasses Netflix in international subscribers",
            "Uber expands into food delivery with major acquisition",
            "Samsung and TSMC announce joint venture for advanced chip manufacturing",
            "PayPal adds Ethereum support, crypto markets rally",
            "Amazon faces new competition from Pinduoduo in international markets"
        ]
        
        # Run analysis
        decisions = trader.run_analysis(headlines)
        
        # Print results
        for decision in decisions:
            print(f"\nCompany: {decision['company']}")
            print(f"Ticker: {decision['ticker']}")
            print(f"Decision: {decision['decision']}")
            print(f"Risk Level: {decision['risk_level']}")
            print(f"Market Trend: {decision['market_trend']}")
            print(f"Reason: {decision['reason']}")
            print("-" * 50)
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main() 