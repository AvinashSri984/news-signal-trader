# News Signal Trader

An AI-powered trading bot that analyzes news headlines, detects company mentions, performs sentiment analysis, and makes trading decisions based on news impact and stock volume spikes.

## Features

- News headline scanning and company mention detection
- Sentiment analysis using TextBlob
- Stock volume spike detection using yFinance
- Automated trading decision making (BUY/SELL/WATCH)
- Modular architecture for easy extension and maintenance

## Project Structure

```
news-signal-trader/
├── src/
│   ├── event_filter/        # News filtering and company detection
│   ├── sentiment_model/     # Sentiment analysis implementation
│   └── trade_decision/      # Trading decision logic
├── tests/                   # Unit tests
├── data/                    # Data storage
├── main.py                  # Main application entry point
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/news-signal-trader.git
cd news-signal-trader
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

## Usage

Run the main application:
```bash
python main.py
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details