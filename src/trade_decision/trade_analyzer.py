import yfinance as yf
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import random

logger = logging.getLogger(__name__)

class TradeAnalyzer:
    def __init__(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Trading thresholds
        self.volume_spike_threshold = 1.1  # Lowered from 1.15
        self.sentiment_threshold = 0.1  # Lowered from 0.12
        self.confidence_threshold = 0.15  # Lowered from 0.2
        
        # Market trend thresholds
        self.moderate_trend_threshold = 0.02  # 2% change for moderate trend
        self.strong_trend_threshold = 0.05  # 5% change for strong trend
        self.volume_volatility_medium = 0.1  # Lowered from 0.15
        
        # Decision thresholds
        self.strong_sentiment_threshold = 0.3  # Lowered from 0.35
        self.moderate_sentiment_threshold = 0.1  # Lowered from 0.12
        self.volume_threshold = 1.1  # Lowered from 1.15
        
        # New parameters for breakthrough announcements
        self.breakthrough_sentiment_boost = 0.15  # Increased from 0.1
        self.breakthrough_volume_boost = 0.2  # Increased from 0.15
        
        # Sector-specific trend thresholds
        self.tech_sector_boost = 0.1  # Increased from 0.08
        self.healthcare_sector_boost = 0.08  # Increased from 0.05
        self.finance_sector_boost = 0.06  # Increased from 0.04
        
        # Risk assessment parameters
        self.high_risk_threshold = 0.25  # Lowered from 0.3
        self.medium_risk_threshold = 0.15  # Lowered from 0.2
        
        # Market trend parameters
        self.trend_strength_threshold = 0.6  # Lowered from 0.7
        self.trend_confirmation_periods = 3  # Reduced from 5
        
        # Volume analysis parameters
        self.volume_trend_periods = 5  # Reduced from 10
        self.volume_surge_threshold = 1.5  # Lowered from 2.0
        
        # Sentiment momentum parameters
        self.sentiment_momentum_periods = 3  # Reduced from 5
        self.sentiment_acceleration_threshold = 0.1  # Lowered from 0.15
        
        # Market trend analysis parameters
        self.trend_period = 20  # Days to analyze for trend
        self.ma_short = 5      # Short-term moving average
        self.ma_long = 20      # Long-term moving average
        self.ma_very_short = 3 # Very short-term moving average for quick signals
        
        # Retry parameters
        self.max_retries = 3
        self.retry_delay = 1.5  # seconds

        self.risk_levels = ['LOW', 'MEDIUM', 'HIGH']
        self.market_trends = ['BULLISH', 'NEUTRAL', 'BEARISH']
        self.decision_types = ['STRONG_BUY', 'BUY', 'HOLD', 'WATCH', 'SELL', 'STRONG_SELL']

    def fetch_with_retries(self, fetch_func, *args, **kwargs):
        """Helper to retry yfinance data fetching with fallback to mock data"""
        for attempt in range(self.max_retries):
            try:
                result = fetch_func(*args, **kwargs)
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                self.logger.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(self.retry_delay)
        self.logger.error("All attempts failed, using mock data.")
        return self.get_mock_history(kwargs.get('period', '5d'))

    def get_mock_history(self, ticker: str, period: str = '1mo') -> pd.DataFrame:
        """Get mock historical data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period to fetch (e.g., '1d', '5d', '1mo', '3mo', '1y')
            
        Returns:
            DataFrame with mock historical data
        """
        # Always generate at least 60 business days of data for proper MA calculation
        min_days = 60
        
        # Convert period to number of days
        if period == '1d':
            days = 1
        elif period == '5d':
            days = 5
        elif period == '1mo':
            days = 30
        elif period == '3mo':
            days = 90
        elif period == '1y':
            days = 365
        else:
            days = 30
            
        # Ensure we have enough days for proper MA calculation
        days = max(days, min_days)
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days*2)  # Double the days to account for weekends
        dates = pd.bdate_range(start=start_date, end=end_date)
        
        # Ensure we have enough business days
        if len(dates) < min_days:
            start_date = end_date - timedelta(days=min_days*2)
            dates = pd.bdate_range(start=start_date, end=end_date)
        
        # Generate base price and volume
        base_price = 100.0
        base_volume = 1000000
        
        # Generate more realistic price movements
        prices = []
        volumes = []
        
        # Initialize with base values
        current_price = base_price
        current_volume = base_volume
        
        # Determine if this ticker should have a strong trend
        has_strong_trend = random.random() < 0.3  # 30% chance of strong trend
        
        if has_strong_trend:
            # Generate a strong trend direction
            trend_direction = random.choice([-1, 1])
            base_trend = trend_direction * random.uniform(0.01, 0.03)  # 1-3% daily trend
            trend_volatility = random.uniform(0.005, 0.015)  # Additional volatility
        else:
            # Generate a more moderate trend
            base_trend = random.uniform(-0.005, 0.005)  # -0.5% to 0.5% daily trend
            trend_volatility = random.uniform(0.002, 0.008)  # Lower volatility
        
        # Generate occasional market events
        event_probability = 0.05  # 5% chance of a market event
        event_magnitude = random.uniform(0.05, 0.15)  # 5-15% price impact
        
        for i in range(len(dates)):
            # Add trend component
            price_change = base_trend + random.gauss(0, trend_volatility)
            
            # Occasionally add market events
            if random.random() < event_probability:
                event_direction = random.choice([-1, 1])
                price_change += event_direction * event_magnitude
                # Increase volume during events
                volume_multiplier = random.uniform(2.0, 4.0)
            else:
                volume_multiplier = random.uniform(0.8, 1.2)
            
            # Update price and volume
            current_price *= (1 + price_change)
            current_volume = base_volume * volume_multiplier
            
            # Ensure price doesn't go below 1.0
            current_price = max(current_price, 1.0)
            
            prices.append(current_price)
            volumes.append(current_volume)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': [p * random.uniform(1.0, 1.02) for p in prices],
            'Low': [p * random.uniform(0.98, 1.0) for p in prices],
            'Close': prices,
            'Volume': volumes
        })
        
        # Add some technical indicators
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Calculate daily returns
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Add RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

    def get_historical_data(self, ticker: str, period: str = '5d') -> pd.DataFrame:
        """Get historical data for a ticker with error handling"""
        try:
            stock = yf.Ticker(ticker)
            hist = self.fetch_with_retries(stock.history, period=period)
            if hist.empty:
                logger.warning(f"No historical data available for {ticker}")
                return self.get_mock_history(period)
            return hist
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return self.get_mock_history(period)

    def calculate_volume_metrics(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-related metrics with proper error handling"""
        try:
            if hist.empty or len(hist) < 2:
                return {'volume_ratio': 1.0, 'price_change': 0.0, 'volume_trend': 'unknown'}

            # Use iloc for position-based indexing
            current_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].iloc[:-1].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Calculate price change
            price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]

            # Calculate volume trend
            volume_trend = 'increasing' if volume_ratio > 1.2 else 'decreasing' if volume_ratio < 0.8 else 'stable'

            # Calculate volume volatility
            volume_volatility = hist['Volume'].pct_change().std()

            return {
                'volume_ratio': volume_ratio,
                'price_change': price_change,
                'volume_trend': volume_trend,
                'volume_volatility': volume_volatility
            }
        except Exception as e:
            logger.error(f"Error calculating volume metrics: {str(e)}")
            return {'volume_ratio': 1.0, 'price_change': 0.0, 'volume_trend': 'unknown', 'volume_volatility': 0.0}

    def calculate_volatility(self, hist: pd.DataFrame) -> float:
        """Calculate price volatility with error handling"""
        try:
            if hist.empty or len(hist) < 2:
                return 0.0
            returns = hist['Close'].pct_change().dropna()
            return returns.std()
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0

    def determine_risk_level(self, volatility: float, volume_ratio: float, volume_volatility: float) -> str:
        """Determine risk level with enhanced sensitivity"""
        risk_score = 0
        
        # More sensitive volatility thresholds
        if volatility > self.high_risk_threshold:
            risk_score += 2
        elif volatility > self.medium_risk_threshold:
            risk_score += 1
            
        # More sensitive volume ratio thresholds
        if volume_ratio > self.volume_threshold:
            risk_score += 2
            
        # Enhanced volume volatility contribution
        if volume_volatility > self.volume_volatility_medium:
            risk_score += 1
            
        # More nuanced risk level determination
        if risk_score >= 3.5:  # Lowered from 4
            return 'HIGH'
        elif risk_score >= 1.5:  # Lowered from 2
            return 'MEDIUM'
        return 'LOW'

    def determine_market_trend(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Determine market trend using multiple indicators."""
        try:
            if len(df) < 51:  # Need at least 51 days for 50-day MA
                return "NEUTRAL", 0.0
                
            # Calculate multiple moving averages
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            
            # Calculate price changes
            df['price_change_short'] = df['Close'].pct_change(5)  # 5-day change
            df['price_change_medium'] = df['Close'].pct_change(20)  # 20-day change
            df['price_change_long'] = df['Close'].pct_change(50)  # 50-day change
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Calculate trend strength using weighted average of changes
            trend_strength = (
                0.5 * latest['price_change_short'] +
                0.3 * latest['price_change_medium'] +
                0.2 * latest['price_change_long']
            )
            
            # Check for strong trends first
            if (latest['MA5'] > latest['MA20'] > latest['MA50'] and 
                latest['price_change_short'] > self.strong_trend_threshold):
                return "BULLISH", min(abs(trend_strength), 1.0)
            elif (latest['MA5'] < latest['MA20'] < latest['MA50'] and 
                  latest['price_change_short'] < -self.strong_trend_threshold):
                return "BEARISH", min(abs(trend_strength), 1.0)
            
            # Check for moderate trends
            if (latest['MA5'] > latest['MA20'] and 
                latest['price_change_short'] > self.moderate_trend_threshold):
                return "BULLISH", min(abs(trend_strength), 1.0)
            elif (latest['MA5'] < latest['MA20'] and 
                  latest['price_change_short'] < -self.moderate_trend_threshold):
                return "BEARISH", min(abs(trend_strength), 1.0)
            
            # Check RSI for overbought/oversold conditions
            if latest['RSI'] > 70:
                return "BEARISH", min(abs(trend_strength), 1.0)
            elif latest['RSI'] < 30:
                return "BULLISH", min(abs(trend_strength), 1.0)
            
            return "NEUTRAL", min(abs(trend_strength), 1.0)
            
        except Exception as e:
            logger.error(f"Error determining market trend: {str(e)}")
            return "NEUTRAL", 0.0

    def make_trade_decision(self, sentiment_data: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """Make trading decision with enhanced analysis"""
        try:
            # Get historical data
            hist = self.get_historical_data(ticker)
            if hist.empty:
                return self._create_default_decision(ticker, "No historical data available")

            # Calculate metrics
            volume_metrics = self.calculate_volume_metrics(hist)
            volatility = self.calculate_volatility(hist)
            market_trend, trend_strength = self.determine_market_trend(hist)
            risk_level = self.determine_risk_level(
                volatility, 
                volume_metrics['volume_ratio'],
                volume_metrics['volume_volatility']
            )

            # Extract sentiment data
            sentiment_score = sentiment_data.get('sentiment_score', 0)
            confidence = sentiment_data.get('confidence', 0)

            # Make decision based on multiple factors
            decision = self._determine_trading_action(
                sentiment_score,
                confidence,
                volume_metrics['volume_ratio'],
                trend_strength
            )

            return {
                'ticker': ticker,
                'decision': decision,
                'risk_level': risk_level,
                'market_trend': market_trend,
                'trend_strength': trend_strength,
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'volume_ratio': volume_metrics['volume_ratio'],
                'volume_analysis': volume_metrics,
                'volatility': volatility,
                'reason': self._generate_decision_reason(
                    sentiment_score, confidence,
                    volume_metrics['volume_ratio'], trend_strength
                ),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error making trade decision for {ticker}: {str(e)}")
            return self._create_default_decision(ticker, f"Error in analysis: {str(e)}")

    def _determine_trading_action(self, sentiment: float, confidence: float, 
                                volume_ratio: float, trend_strength: float) -> str:
        """Determine trading action based on multiple factors."""
        # Calculate sentiment momentum
        sentiment_momentum = sentiment * confidence
        
        # Check for breakthrough announcements
        if sentiment > self.strong_sentiment_threshold and volume_ratio > self.volume_threshold:
            if trend_strength > self.trend_strength_threshold:
                return "STRONG_BUY"
            return "BUY"
            
        # Check for strong sell signals
        if sentiment < -self.strong_sentiment_threshold and volume_ratio > self.volume_threshold:
            if trend_strength > self.trend_strength_threshold:
                return "STRONG_SELL"
            return "SELL"
            
        # Check for moderate buy signals
        if (sentiment > self.moderate_sentiment_threshold and 
            confidence > self.confidence_threshold and 
            volume_ratio > 1.0):
            if trend_strength > self.trend_strength_threshold:
                return "BUY"
            return "HOLD"
            
        # Check for moderate sell signals
        if (sentiment < -self.moderate_sentiment_threshold and 
            confidence > self.confidence_threshold and 
            volume_ratio > 1.0):
            if trend_strength > self.trend_strength_threshold:
                return "SELL"
            return "HOLD"
            
        # Default to watch for unclear signals
        return "WATCH"

    def _generate_decision_reason(self, sentiment_score: float, confidence: float,
                                volume_ratio: float, trend_strength: float) -> str:
        """Generate detailed reason for trading decision."""
        reasons = []
        
        # Check sentiment factors
        if abs(sentiment_score) < self.moderate_sentiment_threshold:
            reasons.append("Insufficient sentiment strength")
        elif abs(sentiment_score) < self.strong_sentiment_threshold:
            reasons.append("Moderate sentiment strength")
        
        # Check confidence
        if confidence < self.confidence_threshold:
            reasons.append("Low confidence in analysis")
        
        # Check volume
        if volume_ratio < self.volume_threshold:
            reasons.append("No significant volume change")
        
        # Check trend
        if abs(trend_strength) < self.moderate_trend_threshold:
            reasons.append("Unclear market trend")
        elif abs(trend_strength) < self.strong_trend_threshold:
            reasons.append("Moderate market trend")
        
        return " and ".join(reasons) if reasons else "No specific reason"

    def _create_default_decision(self, ticker: str, reason: str) -> Dict[str, Any]:
        """Create a default decision when analysis fails"""
        return {
            'ticker': ticker,
            'decision': 'WATCH',
            'risk_level': 'MEDIUM',
            'market_trend': 'NEUTRAL',
            'trend_strength': 0.0,
            'sentiment_score': 0.0,
            'confidence': 0.0,
            'volume_ratio': 1.0,
            'volume_analysis': {'volume_ratio': 1.0, 'price_change': 0.0, 'volume_trend': 'unknown', 'volume_volatility': 0.0},
            'volatility': 0.0,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        } 