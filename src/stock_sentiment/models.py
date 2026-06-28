"""
Data models for Stock Sentiment Analyzer
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SentimentLabel(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


@dataclass
class SentimentResult:
    """Result of sentiment analysis on a single text"""
    text: str
    ticker: Optional[str]
    label: SentimentLabel
    confidence: float
    scores: dict
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "ticker": self.ticker,
            "label": self.label.value,
            "confidence": round(self.confidence, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.75

    @property
    def sentiment_score(self) -> float:
        """Returns a single score: +1 (bullish) to -1 (bearish)"""
        if self.label == SentimentLabel.BULLISH:
            return self.confidence
        elif self.label == SentimentLabel.BEARISH:
            return -self.confidence
        return 0.0


@dataclass
class MarketSignal:
    """Aggregated market signal from multiple sentiment results"""
    ticker: str
    signal: SentimentLabel
    strength: SignalStrength
    avg_confidence: float
    bullish_ratio: float
    bearish_ratio: float
    neutral_ratio: float
    sample_count: int
    composite_score: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "signal": self.signal.value,
            "strength": self.strength.value,
            "avg_confidence": round(self.avg_confidence, 4),
            "ratios": {
                "bullish": round(self.bullish_ratio, 4),
                "bearish": round(self.bearish_ratio, 4),
                "neutral": round(self.neutral_ratio, 4),
            },
            "sample_count": self.sample_count,
            "composite_score": round(self.composite_score, 4),
            "timestamp": self.timestamp.isoformat(),
        }

    @property
    def summary(self) -> str:
        direction = "📈" if self.signal == SentimentLabel.BULLISH else (
            "📉" if self.signal == SentimentLabel.BEARISH else "➡️"
        )
        return (
            f"{direction} {self.ticker}: {self.signal.value.upper()} "
            f"({self.strength.value}, score={self.composite_score:+.2f})"
        )
