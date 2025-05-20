import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import NewsSignalTrader
import json
from datetime import datetime
import pandas as pd
from typing import List, Dict
import logging
import numpy as np
import unittest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingBotTrial:
    def __init__(self):
        """Initialize the trading bot trial"""
        self.trader = NewsSignalTrader()
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)

    def run_scenario_trials(self):
        """Run various trading scenarios"""
        scenarios = [
            # US Tech Companies
            {
                "headline": "Apple announces new iPhone with breakthrough AI features",
                "expected_companies": ["Apple"]
            },
            {
                "headline": "Tesla stock surges after record quarterly deliveries",
                "expected_companies": ["Tesla"]
            },
            {
                "headline": "Microsoft acquires AI startup for $10 billion",
                "expected_companies": ["Microsoft"]
            },
            
            # International Companies
            {
                "headline": "Toyota announces major electric vehicle investment",
                "expected_companies": ["Toyota"]
            },
            {
                "headline": "Samsung unveils new chip manufacturing technology",
                "expected_companies": ["Samsung"]
            },
            {
                "headline": "Baidu launches new AI research center",
                "expected_companies": ["Baidu"]
            },
            
            # Cryptocurrency
            {
                "headline": "Bitcoin reaches new all-time high",
                "expected_companies": ["Bitcoin"]
            },
            {
                "headline": "Ethereum network upgrade successful",
                "expected_companies": ["Ethereum"]
            },
            
            # Market-wide Events
            {
                "headline": "Federal Reserve announces interest rate decision",
                "expected_companies": []
            },
            {
                "headline": "US inflation data shows unexpected increase",
                "expected_companies": []
            },
            
            # Mergers & Acquisitions
            {
                "headline": "Microsoft acquires gaming studio for $5 billion",
                "expected_companies": ["Microsoft"]
            },
            {
                "headline": "Amazon expands into healthcare with new acquisition",
                "expected_companies": ["Amazon"]
            },
            
            # CEO Changes
            {
                "headline": "Disney appoints new CEO",
                "expected_companies": ["Disney"]
            },
            {
                "headline": "Intel CEO announces retirement",
                "expected_companies": ["Intel"]
            },
            
            # Regulatory Actions
            {
                "headline": "Google faces new antitrust investigation",
                "expected_companies": ["Google"]
            },
            {
                "headline": "Meta fined for data privacy violations",
                "expected_companies": ["Meta"]
            },
            
            # Tech Launches
            {
                "headline": "Apple unveils new MacBook Pro with M3 chip",
                "expected_companies": ["Apple"]
            },
            {
                "headline": "NVIDIA announces next-generation AI chips",
                "expected_companies": ["NVIDIA"]
            },
            
            # Lawsuits & Scandals
            {
                "headline": "Tesla faces new lawsuit over autopilot safety",
                "expected_companies": ["Tesla"]
            },
            {
                "headline": "Amazon warehouse workers file class action lawsuit",
                "expected_companies": ["Amazon"]
            },
            
            # Semiconductor Industry
            {
                "headline": "TSMC announces new chip manufacturing plant",
                "expected_companies": ["TSMC"]
            },
            {
                "headline": "Intel reveals breakthrough in chip technology",
                "expected_companies": ["Intel"]
            },
            
            # Chinese Tech
            {
                "headline": "Pinduoduo reports strong quarterly growth",
                "expected_companies": ["Pinduoduo"]
            },
            {
                "headline": "JD.com expands into new markets",
                "expected_companies": ["JD.com"]
            },
            
            # Fintech
            {
                "headline": "PayPal launches new cryptocurrency features",
                "expected_companies": ["PayPal"]
            },
            {
                "headline": "Visa partners with fintech startups",
                "expected_companies": ["Visa"]
            },
            
            # Entertainment & Streaming
            {
                "headline": "Netflix announces price increase",
                "expected_companies": ["Netflix"]
            },
            {
                "headline": "Disney+ adds new content lineup",
                "expected_companies": ["Disney"]
            },
            
            # Ride-sharing & Mobility
            {
                "headline": "Uber expands into new cities",
                "expected_companies": ["Uber"]
            },
            
            # International Market Events
            {
                "headline": "Samsung and TSMC announce joint venture",
                "expected_companies": ["Samsung", "TSMC"]
            },
            
            # Cryptocurrency Integration
            {
                "headline": "PayPal adds Ethereum support",
                "expected_companies": ["PayPal", "Ethereum"]
            },
            
            # E-commerce Competition
            {
                "headline": "Amazon faces new competition from Pinduoduo",
                "expected_companies": ["Amazon", "Pinduoduo"]
            },
            
            # AI and Technology
            {
                "headline": "Baidu announces breakthrough in AI technology",
                "expected_companies": ["Baidu"]
            },
            {
                "headline": "ASML reports strong demand for chip equipment",
                "expected_companies": ["ASML"]
            }
        ]
        
        results = []
        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"\nTrial {i}: {scenario['headline']}")
            
            # Process headline
            decisions = self.trader.process_headline(scenario['headline'])
            
            # Log results
            logger.info(f"Expected companies: {scenario['expected_companies']}")
            logger.info(f"Detected companies: {[d['company'] for d in decisions]}")
            logger.info(f"Decisions made: {len(decisions)}")
            
            for decision in decisions:
                logger.info(f"\nDecision for {decision['company']} ({decision['ticker']}):")
                logger.info(f"Action: {decision['decision']}")
                logger.info(f"Risk Level: {decision['risk_level']}")
                logger.info(f"Market Trend: {decision['market_trend']}")
                logger.info(f"Reason: {decision['reason']}")
            
            results.append({
                'trial_number': i,
                'headline': scenario['headline'],
                'expected_companies': scenario['expected_companies'],
                'detected_companies': [d['company'] for d in decisions],
                'decisions': decisions
            })
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'trial_results_{timestamp}.json'
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"\nResults saved to {filepath}")
        return results

def main():
    """Run the trading bot trials"""
    try:
        trial = TradingBotTrial()
        results = trial.run_scenario_trials()
        
        # Print summary
        total_trials = len(results)
        total_decisions = sum(len(r['decisions']) for r in results)
        
        print(f"\nTrial Summary:")
        print(f"Total trials: {total_trials}")
        print(f"Total decisions made: {total_decisions}")
        print(f"Average decisions per trial: {total_decisions/total_trials:.2f}")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main() 