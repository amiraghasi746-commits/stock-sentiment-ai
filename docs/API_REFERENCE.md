[Uploading API_REFERENCE.md…]()
# API Reference

## `SentimentAnalyzer`

```python
from stock_sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer(negation_window=3)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `negation_window` | `int` | `3` | Number of tokens before a term to search for negation words |

---

### `analyze(text, ticker=None, source="unknown") → SentimentResult`

Analyze the sentiment of a single financial text.

**Arguments:**
- `text` *(str)*: Input text — headline, tweet, post, etc.
- `ticker` *(str, optional)*: Stock ticker symbol. Auto-extracted from text (via `$TICKER` pattern) if not provided.
- `source` *(str)*: Label for the data source. Default: `"unknown"`.

**Returns:** [`SentimentResult`](#sentimentresult)

**Raises:** `ValueError` if `text` is empty or whitespace-only.

**Example:**
```python
result = analyzer.analyze(
    "Apple surges on record earnings beat",
    ticker="AAPL",
    source="reuters"
)
print(result.label)       # SentimentLabel.BULLISH
print(result.confidence)  # e.g. 0.8312
print(result.scores)      # {'bullish': 0.83, 'bearish': 0.05, 'neutral': 0.12}
```

---

### `batch_analyze(texts, ticker=None, source="batch") → list[SentimentResult]`

Analyze multiple texts in batch. Failed items are logged and skipped (no exception raised).

**Arguments:**
- `texts` *(list[str])*: List of texts to analyze.
- `ticker` *(str, optional)*: Apply same ticker to all texts.
- `source` *(str)*: Label for the data source.

**Returns:** `list[SentimentResult]`

---

## `AnalysisPipeline`

```python
from stock_sentiment import AnalysisPipeline

pipeline = AnalysisPipeline()
```

---

### `run(texts, ticker, source="mixed") → MarketSignal`

Run the full pipeline and aggregate results into a single market signal.

**Arguments:**
- `texts` *(list[str])*: List of texts to analyze.
- `ticker` *(str)*: Stock ticker symbol.
- `source` *(str)*: Data source label.

**Returns:** [`MarketSignal`](#marketsignal)

**Raises:** 
- `ValueError` if `texts` is empty.
- `RuntimeError` if all texts fail analysis.

---

### `run_multi_ticker(ticker_texts, source="mixed") → dict[str, MarketSignal]`

Run pipeline for multiple tickers simultaneously.

**Arguments:**
- `ticker_texts` *(dict[str, list[str]])*: Mapping of ticker → list of texts.
- `source` *(str)*: Data source label.

**Returns:** `dict[str, MarketSignal]`

---

### `generate_report(signals) → str`

Generate a formatted human-readable report from market signals.

**Arguments:**
- `signals` *(dict[str, MarketSignal])*: Output from `run_multi_ticker`.

**Returns:** `str` — multi-line formatted report.

---

## Data Models

### `SentimentResult`

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Original input text |
| `ticker` | `str \| None` | Extracted or provided ticker |
| `label` | `SentimentLabel` | BULLISH / BEARISH / NEUTRAL |
| `confidence` | `float` | Confidence in the label (0–1) |
| `scores` | `dict` | Probabilities: `{'bullish': ..., 'bearish': ..., 'neutral': ...}` |
| `timestamp` | `datetime` | Time of analysis |
| `source` | `str` | Data source label |
| `is_high_confidence` | `bool` *(property)* | True if confidence ≥ 0.75 |
| `sentiment_score` | `float` *(property)* | +1 (bullish) to -1 (bearish) |

---

### `MarketSignal`

| Field | Type | Description |
|-------|------|-------------|
| `ticker` | `str` | Stock ticker |
| `signal` | `SentimentLabel` | Aggregated signal direction |
| `strength` | `SignalStrength` | STRONG / MODERATE / WEAK |
| `avg_confidence` | `float` | Average confidence across samples |
| `bullish_ratio` | `float` | Fraction of bullish results |
| `bearish_ratio` | `float` | Fraction of bearish results |
| `neutral_ratio` | `float` | Fraction of neutral results |
| `sample_count` | `int` | Number of texts analyzed |
| `composite_score` | `float` | Aggregate score (-1 to +1) |
| `timestamp` | `datetime` | Time of signal generation |
| `summary` | `str` *(property)* | One-line human-readable summary |

---

### Enums

```python
class SentimentLabel(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class SignalStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
```
