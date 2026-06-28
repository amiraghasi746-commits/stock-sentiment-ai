"""
Analysis Pipeline — aggregates multiple SentimentResults into a MarketSignal
"""

import logging
from typing import Optional
from datetime import datetime

from .analyzer import SentimentAnalyzer
from .models import SentimentLabel, SentimentResult, MarketSignal, SignalStrength

logger = logging.getLogger(__name__)


def _determine_strength(composite: float, avg_confidence: float, n: int) -> SignalStrength:
    """Determine signal strength based on composite score, confidence and sample size"""
    abs_score = abs(composite)
    size_factor = min(n / 10.0, 1.0)  # Saturates at 10 samples

    weighted = abs_score * 0.5 + avg_confidence * 0.3 + size_factor * 0.2

    if weighted >= 0.65:
        return SignalStrength.STRONG
    elif weighted >= 0.45:
        return SignalStrength.MODERATE
    return SignalStrength.WEAK


class AnalysisPipeline:
    """
    End-to-end pipeline: text list → MarketSignal

    Usage:
        pipeline = AnalysisPipeline()
        signal = pipeline.run(texts=["AAPL surges on record earnings", ...], ticker="AAPL")
        print(signal.summary)
    """

    def __init__(self):
        self.analyzer = SentimentAnalyzer()

    def run(
        self,
        texts: list[str],
        ticker: str,
        source: str = "mixed",
    ) -> MarketSignal:
        """
        Run the full pipeline on a list of texts for a given ticker.

        Args:
            texts: List of news headlines / tweets / posts
            ticker: Stock ticker (e.g. "AAPL")
            source: Data source label

        Returns:
            MarketSignal with aggregated sentiment
        """
        if not texts:
            raise ValueError("texts list cannot be empty")

        results: list[SentimentResult] = self.analyzer.batch_analyze(
            texts, ticker=ticker, source=source
        )

        if not results:
            raise RuntimeError("All texts failed analysis — no results to aggregate")

        n = len(results)
        label_counts = {
            SentimentLabel.BULLISH: 0,
            SentimentLabel.BEARISH: 0,
            SentimentLabel.NEUTRAL: 0,
        }
        total_confidence = 0.0
        composite_score = 0.0

        for r in results:
            label_counts[r.label] += 1
            total_confidence += r.confidence
            composite_score += r.sentiment_score

        avg_confidence = total_confidence / n
        composite_score /= n  # Range: [-1, +1]

        bull_ratio = label_counts[SentimentLabel.BULLISH] / n
        bear_ratio = label_counts[SentimentLabel.BEARISH] / n
        neut_ratio = label_counts[SentimentLabel.NEUTRAL] / n

        # Final signal: dominant label, but use composite_score for ties
        if composite_score > 0.05:
            signal = SentimentLabel.BULLISH
        elif composite_score < -0.05:
            signal = SentimentLabel.BEARISH
        else:
            signal = SentimentLabel.NEUTRAL

        strength = _determine_strength(composite_score, avg_confidence, n)

        logger.info(
            "Pipeline complete: ticker=%s, n=%d, signal=%s(%s), score=%.3f",
            ticker, n, signal.value, strength.value, composite_score
        )

        return MarketSignal(
            ticker=ticker,
            signal=signal,
            strength=strength,
            avg_confidence=avg_confidence,
            bullish_ratio=bull_ratio,
            bearish_ratio=bear_ratio,
            neutral_ratio=neut_ratio,
            sample_count=n,
            composite_score=composite_score,
            timestamp=datetime.now(),
        )

    def run_multi_ticker(
        self,
        ticker_texts: dict[str, list[str]],
        source: str = "mixed",
    ) -> dict[str, MarketSignal]:
        """
        Run pipeline on multiple tickers simultaneously.

        Args:
            ticker_texts: Dict mapping ticker → list of texts
            source: Data source label

        Returns:
            Dict mapping ticker → MarketSignal
        """
        signals = {}
        for ticker, texts in ticker_texts.items():
            try:
                signals[ticker] = self.run(texts=texts, ticker=ticker, source=source)
            except Exception as e:
                logger.error("Failed to process ticker %s: %s", ticker, e)
        return signals

    def generate_report(self, signals: dict[str, MarketSignal]) -> str:
        """Generate a human-readable report from multiple market signals"""
        if not signals:
            return "No signals to report."

        lines = [
            "=" * 60,
            "  STOCK MARKET SENTIMENT REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
        ]

        # Sort: bullish first, then bearish, then neutral
        sorted_signals = sorted(
            signals.values(),
            key=lambda s: (
                0 if s.signal == SentimentLabel.BULLISH else
                1 if s.signal == SentimentLabel.BEARISH else 2,
                -abs(s.composite_score),
            )
        )

        for sig in sorted_signals:
            lines.append(sig.summary)
            lines.append(
                f"   Confidence: {sig.avg_confidence:.1%} | "
                f"Samples: {sig.sample_count} | "
                f"Bull/Bear/Neut: {sig.bullish_ratio:.0%}/{sig.bearish_ratio:.0%}/{sig.neutral_ratio:.0%}"
            )
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
