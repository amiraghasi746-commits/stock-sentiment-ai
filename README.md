# 📈 Stock Market Sentiment Analyzer

> **AI-powered NLP engine for financial market sentiment analysis**

[![CI/CD](https://github.com/your-username/stock-sentiment-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/stock-sentiment-ai/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)]()

Analyze news headlines, tweets, and financial text to generate actionable market sentiment signals. Built with a domain-specific financial NLP pipeline that handles negation, intensifiers, and multi-ticker aggregation — **zero external dependencies**.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Financial NLP** | 85+ domain-specific lexicon terms with calibrated weights |
| 🔄 **Negation Handling** | Window-based negation detection ("does not surge" → bearish) |
| ⚡ **Intensity Amplification** | "very", "massively", "extremely" boost signal strength |
| 📊 **Multi-Ticker Pipeline** | Aggregate sentiment across multiple stocks simultaneously |
| 🎯 **Calibrated Confidence** | Sigmoid-calibrated probabilities, not raw scores |
| 🚀 **Zero Dependencies** | Pure Python 3.10+ stdlib only |
| 🖥️ **CLI + Python API** | Use from terminal or import as a library |

---

## 🚀 Quick Start

### Install

```bash
git clone https://github.com/your-username/stock-sentiment-ai
cd stock-sentiment-ai
pip install -e .
```

### Python API

```python
from stock_sentiment import SentimentAnalyzer, AnalysisPipeline

# Analyze a single headline
analyzer = SentimentAnalyzer()
result = analyzer.analyze("Apple surges on record earnings beat", ticker="AAPL")
print(result.label)       # SentimentLabel.BULLISH
print(result.confidence)  # 0.83
print(result.scores)      # {'bullish': 0.83, 'bearish': 0.05, 'neutral': 0.12}

# Multi-ticker pipeline
pipeline = AnalysisPipeline()
signals = pipeline.run_multi_ticker({
    "AAPL": ["Apple beats earnings", "iPhone demand surges", "record buyback"],
    "TSLA": ["Tesla crashes on miss", "delivery numbers disappoint badly"],
    "NVDA": ["NVIDIA soars on AI demand", "record data center revenue"],
})
print(pipeline.generate_report(signals))
```

### CLI

```bash
# Analyze a single headline
python -m stock_sentiment analyze "Apple surges on record earnings" --ticker AAPL

# Run on a file of headlines (one per line)
python -m stock_sentiment pipeline data/samples/aapl_headlines.txt --ticker AAPL

# JSON output
python -m stock_sentiment analyze "Tesla crashes" --ticker TSLA --json

# Built-in demo
python -m stock_sentiment demo
```

---

## 📊 Example Output

```
============================================================
  STOCK MARKET SENTIMENT REPORT
  Generated: 2024-01-15 14:32:08
============================================================

📈 NVDA: BULLISH (strong, score=+0.82)
   Confidence: 87% | Samples: 3 | Bull/Bear/Neut: 100%/0%/0%

📈 AAPL: BULLISH (moderate, score=+0.54)
   Confidence: 71% | Samples: 3 | Bull/Bear/Neut: 67%/33%/0%

📉 TSLA: BEARISH (strong, score=-0.71)
   Confidence: 79% | Samples: 2 | Bull/Bear/Neut: 0%/100%/0%

============================================================
```

---

## 🏗️ Project Structure

```
stock-sentiment-ai/
├── src/stock_sentiment/
│   ├── __init__.py        # Public API exports
│   ├── __main__.py        # CLI entry point
│   ├── analyzer.py        # Core NLP engine
│   ├── pipeline.py        # Aggregation pipeline
│   └── models.py          # Data models
├── tests/
│   ├── conftest.py        # Pytest configuration
│   └── test_sentiment.py  # 30+ unit & integration tests
├── docs/
│   ├── API_REFERENCE.md   # Full API documentation
│   └── ARCHITECTURE.md    # System design & algorithm
├── notebooks/
│   └── demo.ipynb         # Interactive demo notebook
├── data/samples/          # Sample headline datasets
├── .github/workflows/
│   └── ci.yml             # CI/CD pipeline
└── pyproject.toml         # Package configuration
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=stock_sentiment --cov-report=term-missing

# Run only unit tests
pytest tests/ -k "not Integration"
```

---

## 📖 Documentation

- [API Reference](docs/API_REFERENCE.md) — Full class and method documentation
- [Architecture](docs/ARCHITECTURE.md) — System design and algorithm details
- [Demo Notebook](notebooks/demo.ipynb) — Interactive examples

---

## 🔬 How It Works

1. **Preprocess** — strip URLs, mentions, lowercase, tokenize
2. **Lexicon Scoring** — match tokens against 85+ weighted financial terms
3. **Negation Detection** — sliding window catches "not", "never", "don't" etc.
4. **Intensity Boost** — amplify scores near "very", "extremely", "massively"
5. **Length Normalize** — divide by `log(token_count)` to remove length bias
6. **Sigmoid Calibrate** — convert net score to calibrated probabilities
7. **Aggregate** — pipeline averages `sentiment_score` values across texts
8. **Signal** — composite score → BULLISH / BEARISH / NEUTRAL with STRONG / MODERATE / WEAK

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Run tests: `pytest tests/ -v`
4. Open a Pull Request
