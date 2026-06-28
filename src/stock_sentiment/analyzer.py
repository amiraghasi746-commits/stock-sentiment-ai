"""
Core Sentiment Analyzer using ML + Financial NLP
"""

import re
import math
import logging
from typing import Optional
from collections import Counter

from .models import SentimentLabel, SentimentResult

logger = logging.getLogger(__name__)


# Financial domain lexicon — carefully curated
BULLISH_TERMS = {
    # Price action
    "surge": 0.85, "soar": 0.85, "rally": 0.75, "spike": 0.70, "jump": 0.70,
    "gain": 0.65, "rise": 0.60, "climb": 0.65, "advance": 0.60, "breakout": 0.80,
    "skyrocket": 0.90, "moon": 0.85, "pump": 0.75, "bounce": 0.60, "recover": 0.55,
    # Fundamentals
    "beat": 0.70, "outperform": 0.75, "upgrade": 0.75, "strong": 0.60,
    "exceed": 0.70, "record": 0.65, "growth": 0.65, "profit": 0.70,
    "revenue": 0.50, "earnings": 0.50, "expansion": 0.65, "milestone": 0.60,
    "bullish": 0.85, "upside": 0.70, "opportunity": 0.55, "buy": 0.70,
    "undervalued": 0.75, "cheap": 0.55, "accumulate": 0.70, "long": 0.65,
    # Sentiment
    "optimistic": 0.70, "confident": 0.65, "positive": 0.60, "promising": 0.65,
    "innovative": 0.60, "breakthrough": 0.75, "disruptive": 0.65, "leader": 0.60,
    # Tech / AI domain
    "demand": 0.60, "adoption": 0.60, "dominate": 0.65, "lead": 0.60,
    "partnership": 0.55, "launch": 0.55, "winning": 0.70, "win": 0.65,
    "boost": 0.65, "accelerate": 0.65, "momentum": 0.65, "robust": 0.65,
}

BEARISH_TERMS = {
    # Price action
    "crash": 0.90, "collapse": 0.90, "plunge": 0.85, "drop": 0.65, "fall": 0.65,
    "decline": 0.70, "slide": 0.65, "dump": 0.80, "tank": 0.80, "sink": 0.75,
    "tumble": 0.75, "selloff": 0.80, "correction": 0.60, "dip": 0.50, "retreat": 0.60,
    # Fundamentals
    "miss": 0.70, "underperform": 0.75, "downgrade": 0.80, "weak": 0.65,
    "disappoint": 0.75, "loss": 0.70, "debt": 0.55, "deficit": 0.65,
    "bearish": 0.85, "downside": 0.70, "risk": 0.50, "sell": 0.70,
    "overvalued": 0.75, "expensive": 0.55, "short": 0.70, "avoid": 0.65,
    # Sentiment
    "pessimistic": 0.70, "uncertain": 0.55, "negative": 0.65, "concern": 0.60,
    "warning": 0.70, "threat": 0.65, "crisis": 0.85, "fraud": 0.90, "lawsuit": 0.75,
    "bankruptcy": 0.95, "investigation": 0.75, "recall": 0.70, "layoff": 0.65,
}

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "without", "barely",
                  "hardly", "rarely", "scarcely", "dont", "doesnt", "didnt",
                  "wont", "cant", "isnt", "arent", "wasnt", "werent"}

INTENSIFIERS = {"very": 1.3, "extremely": 1.5, "highly": 1.3, "significantly": 1.3,
                "massively": 1.5, "incredibly": 1.4, "absolutely": 1.4, "completely": 1.3}

TICKER_PATTERN = re.compile(r'\b\$?([A-Z]{1,5})\b')
URL_PATTERN = re.compile(r'http\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')


class SentimentAnalyzer:
    """
    Financial sentiment analyzer using a lexicon-based ML hybrid approach.

    Features:
    - Financial domain-specific lexicon with weighted terms
    - Negation handling (window-based)
    - Intensity amplification
    - Ticker extraction
    - Confidence calibration via sigmoid smoothing
    """

    def __init__(self, negation_window: int = 3):
        self.negation_window = negation_window
        self._bullish = BULLISH_TERMS
        self._bearish = BEARISH_TERMS
        logger.info("SentimentAnalyzer initialized (lexicon size: %d bullish, %d bearish)",
                    len(self._bullish), len(self._bearish))

    def _preprocess(self, text: str) -> list[str]:
        """Clean and tokenize text"""
        text = URL_PATTERN.sub(" ", text)
        text = MENTION_PATTERN.sub(" ", text)
        text = re.sub(r'[^\w\s$]', ' ', text.lower())
        tokens = text.split()
        return tokens

    def _extract_ticker(self, text: str) -> Optional[str]:
        """Extract stock ticker from text"""
        # Look for explicit $ prefix first
        dollar_match = re.search(r'\$([A-Z]{1,5})\b', text.upper())
        if dollar_match:
            return dollar_match.group(1)
        return None

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function for probability calibration"""
        return 1.0 / (1.0 + math.exp(-x))

    def _compute_scores(self, tokens: list[str]) -> dict:
        """
        Compute raw sentiment scores with negation and intensity handling.
        Returns dict with bullish_score, bearish_score, token_count.
        """
        bullish_score = 0.0
        bearish_score = 0.0
        n = len(tokens)

        for i, token in enumerate(tokens):
            # Check if negated within window
            start = max(0, i - self.negation_window)
            context = tokens[start:i]
            negated = any(w in NEGATION_WORDS for w in context)

            # Check for intensifier before this token
            intensity = 1.0
            if i > 0 and tokens[i - 1] in INTENSIFIERS:
                intensity = INTENSIFIERS[tokens[i - 1]]

            if token in self._bullish:
                score = self._bullish[token] * intensity
                if negated:
                    bearish_score += score * 0.6  # Negated bullish → weak bearish
                else:
                    bullish_score += score

            elif token in self._bearish:
                score = self._bearish[token] * intensity
                if negated:
                    bullish_score += score * 0.6  # Negated bearish → weak bullish
                else:
                    bearish_score += score

        return {
            "raw_bullish": bullish_score,
            "raw_bearish": bearish_score,
            "token_count": n,
        }

    def analyze(self, text: str, ticker: Optional[str] = None,
                source: str = "unknown") -> SentimentResult:
        """
        Analyze sentiment of financial text.

        Args:
            text: Input text (news headline, tweet, etc.)
            ticker: Stock ticker symbol (auto-extracted if None)
            source: Source of the text (twitter, news, reddit, etc.)

        Returns:
            SentimentResult with label, confidence, and scores
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        extracted_ticker = ticker or self._extract_ticker(text)
        tokens = self._preprocess(text)

        if not tokens:
            return SentimentResult(
                text=text, ticker=extracted_ticker,
                label=SentimentLabel.NEUTRAL, confidence=0.5,
                scores={"bullish": 0.33, "bearish": 0.33, "neutral": 0.34},
                source=source,
            )

        raw = self._compute_scores(tokens)
        bull = raw["raw_bullish"]
        bear = raw["raw_bearish"]

        # Normalize by log(token_count) to handle length bias
        length_norm = math.log(max(raw["token_count"], 2))
        bull_norm = bull / length_norm
        bear_norm = bear / length_norm

        # Compute net score: positive = bullish, negative = bearish
        net = bull_norm - bear_norm

        # Convert to probabilities
        p_bullish = self._sigmoid(net * 2.5)
        p_bearish = self._sigmoid(-net * 2.5)
        # Neutral is the "leftover" probability
        total = max(p_bullish + p_bearish, 1e-9)
        p_bullish_n = p_bullish / total
        p_bearish_n = p_bearish / total
        p_neutral = max(0.0, 1.0 - abs(net) * 0.8)

        # Re-normalize
        total2 = p_bullish_n + p_bearish_n + p_neutral
        p_bull_f = p_bullish_n / total2
        p_bear_f = p_bearish_n / total2
        p_neut_f = p_neutral / total2

        # Determine label
        scores_map = {
            SentimentLabel.BULLISH: p_bull_f,
            SentimentLabel.BEARISH: p_bear_f,
            SentimentLabel.NEUTRAL: p_neut_f,
        }
        label = max(scores_map, key=scores_map.__getitem__)
        confidence = scores_map[label]

        return SentimentResult(
            text=text,
            ticker=extracted_ticker,
            label=label,
            confidence=confidence,
            scores={
                "bullish": round(p_bull_f, 4),
                "bearish": round(p_bear_f, 4),
                "neutral": round(p_neut_f, 4),
            },
            source=source,
        )

    def batch_analyze(self, texts: list[str], ticker: Optional[str] = None,
                      source: str = "batch") -> list[SentimentResult]:
        """Analyze multiple texts in batch"""
        results = []
        for text in texts:
            try:
                result = self.analyze(text, ticker=ticker, source=source)
                results.append(result)
            except Exception as e:
                logger.warning("Skipping text due to error: %s", e)
        return results
