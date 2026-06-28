[Uploading ARCHITECTURE.md…]()
# Architecture Overview

## System Design

```
Input Texts (headlines / tweets / posts)
         │
         ▼
┌─────────────────────┐
│   AnalysisPipeline  │  ← Orchestrator
│  run() / run_multi  │
└──────────┬──────────┘
           │  delegates to
           ▼
┌─────────────────────┐
│  SentimentAnalyzer  │  ← Core ML engine
│   analyze(text)     │
└──────────┬──────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
Preprocess    Score Computation
(tokenize,    (lexicon lookup,
 clean URLs)   negation window,
               intensity boost)
     │            │
     └─────┬──────┘
           ▼
    Sigmoid calibration
    → probability distribution
           │
           ▼
    SentimentResult
  (label, confidence, scores)
           │
           ▼ (aggregated by pipeline)
      MarketSignal
  (signal, strength, composite_score)
```

## Scoring Algorithm

### 1. Preprocessing
- Strip URLs, @mentions
- Lowercase + tokenize
- Remove punctuation

### 2. Lexicon Scoring
Each token is looked up in domain-specific lexicons:
- **Bullish lexicon**: ~40 terms with weights [0.50 – 0.95]
- **Bearish lexicon**: ~45 terms with weights [0.50 – 0.95]

### 3. Negation Handling
A sliding window of `negation_window` tokens (default: 3) is checked before each term.
- Negated bullish → contributes 60% of weight as bearish signal
- Negated bearish → contributes 60% of weight as bullish signal

### 4. Intensity Amplification
Intensifier words (very, extremely, massively, etc.) multiply the following term's weight by 1.3–1.5×.

### 5. Length Normalization
Raw scores are divided by `log(token_count)` to avoid bias toward longer texts.

### 6. Sigmoid Calibration
Net score is passed through a sigmoid function to produce calibrated probabilities:
```
p_bullish = sigmoid(net * 2.5)
p_bearish = sigmoid(-net * 2.5)
p_neutral = max(0, 1 - |net| * 0.8)
```
Probabilities are then L1-normalized to sum to 1.

### 7. Aggregation (Pipeline)
- Average of per-text `sentiment_score` values (range: -1 to +1)
- Signal threshold: composite > +0.05 → BULLISH, < -0.05 → BEARISH
- Strength determined by weighted combination of: `|composite|`, `avg_confidence`, `sample_size_factor`

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pure Python, no heavy dependencies | Deployable anywhere (Lambda, containers, edge) |
| Lexicon-based hybrid (not transformer) | Fast, interpretable, no GPU required |
| Sigmoid calibration | Avoids overconfident outputs from raw scores |
| Negation window (not full parse) | O(n) complexity, robust to informal text |
| dataclass models | Type-safe, easily serializable |
