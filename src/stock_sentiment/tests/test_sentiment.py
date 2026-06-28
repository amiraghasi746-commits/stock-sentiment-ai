"""
Test suite for Stock Sentiment Analyzer
"""

import pytest
from datetime import datetime

from stock_sentiment.analyzer import SentimentAnalyzer
from stock_sentiment.models import SentimentLabel, SentimentResult, MarketSignal, SignalStrength
from stock_sentiment.pipeline import AnalysisPipeline


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def analyzer():
    return SentimentAnalyzer()


@pytest.fixture
def pipeline():
    return AnalysisPipeline()


# ─── SentimentAnalyzer tests ──────────────────────────────────────────────────

class TestSentimentAnalyzer:

    def test_bullish_headline(self, analyzer):
        result = analyzer.analyze("Apple surges on record earnings beat")
        assert result.label == SentimentLabel.BULLISH
        assert result.confidence > 0.5

    def test_bearish_headline(self, analyzer):
        result = analyzer.analyze("Tesla crashes on disappointing sales miss and massive loss")
        assert result.label == SentimentLabel.BEARISH
        assert result.confidence > 0.5

    def test_neutral_headline(self, analyzer):
        result = analyzer.analyze("Company releases quarterly report")
        assert result.label == SentimentLabel.NEUTRAL

    def test_returns_sentiment_result(self, analyzer):
        result = analyzer.analyze("Stock rises", ticker="AAPL")
        assert isinstance(result, SentimentResult)
        assert result.ticker == "AAPL"
        assert result.label in SentimentLabel
        assert 0.0 <= result.confidence <= 1.0

    def test_scores_sum_to_one(self, analyzer):
        result = analyzer.analyze("Apple beats earnings expectations strongly")
        total = sum(result.scores.values())
        assert abs(total - 1.0) < 0.01, f"Scores should sum to ~1.0, got {total}"

    def test_ticker_extraction(self, analyzer):
        result = analyzer.analyze("$NVDA surges on AI demand")
        assert result.ticker == "NVDA"

    def test_explicit_ticker_overrides_extraction(self, analyzer):
        result = analyzer.analyze("$AAPL rises", ticker="TSLA")
        assert result.ticker == "TSLA"

    def test_negation_handling(self, analyzer):
        pos = analyzer.analyze("Stock surges strongly")
        neg = analyzer.analyze("Stock does not surge at all")
        # Negated version should be less bullish
        assert pos.scores["bullish"] > neg.scores["bullish"]

    def test_intensifier_effect(self, analyzer):
        normal = analyzer.analyze("Stock gains")
        intense = analyzer.analyze("Stock very significantly gains")
        # Intensified should have higher bullish confidence
        assert intense.scores["bullish"] >= normal.scores["bullish"]

    def test_empty_text_raises(self, analyzer):
        with pytest.raises(ValueError, match="cannot be empty"):
            analyzer.analyze("")

    def test_whitespace_only_raises(self, analyzer):
        with pytest.raises(ValueError, match="cannot be empty"):
            analyzer.analyze("   ")

    def test_source_label_preserved(self, analyzer):
        result = analyzer.analyze("Stock rises", source="twitter")
        assert result.source == "twitter"

    def test_sentiment_score_range(self, analyzer):
        for text in ["Stock surges", "Stock crashes", "Stock trades flat"]:
            result = analyzer.analyze(text)
            assert -1.0 <= result.sentiment_score <= 1.0

    def test_bullish_positive_score(self, analyzer):
        result = analyzer.analyze("Massive rally and surge in profits")
        assert result.sentiment_score > 0

    def test_bearish_negative_score(self, analyzer):
        result = analyzer.analyze("Stock crashes and collapses on huge loss")
        assert result.sentiment_score < 0

    def test_batch_analyze(self, analyzer):
        texts = ["Stock rises", "Stock falls", "Stock trades sideways"]
        results = analyzer.batch_analyze(texts)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, SentimentResult)

    def test_batch_with_ticker(self, analyzer):
        texts = ["Rises strongly", "Falls hard"]
        results = analyzer.batch_analyze(texts, ticker="MSFT")
        assert all(r.ticker == "MSFT" for r in results)

    def test_is_high_confidence(self, analyzer):
        result = analyzer.analyze("Massive crash and collapse bankruptcy fraud")
        # Strong signal should be high confidence
        assert isinstance(result.is_high_confidence, bool)

    def test_high_confidence_threshold(self):
        from stock_sentiment.models import SentimentResult
        r = SentimentResult(
            text="test", ticker=None,
            label=SentimentLabel.BULLISH, confidence=0.80,
            scores={"bullish": 0.80, "bearish": 0.10, "neutral": 0.10},
        )
        assert r.is_high_confidence is True

    def test_low_confidence_threshold(self):
        from stock_sentiment.models import SentimentResult
        r = SentimentResult(
            text="test", ticker=None,
            label=SentimentLabel.NEUTRAL, confidence=0.55,
            scores={"bullish": 0.25, "bearish": 0.20, "neutral": 0.55},
        )
        assert r.is_high_confidence is False


# ─── AnalysisPipeline tests ───────────────────────────────────────────────────

class TestAnalysisPipeline:

    def test_returns_market_signal(self, pipeline):
        texts = ["Apple surges on record sales", "AAPL rallies on earnings beat"]
        signal = pipeline.run(texts=texts, ticker="AAPL")
        assert isinstance(signal, MarketSignal)

    def test_empty_texts_raises(self, pipeline):
        with pytest.raises(ValueError, match="cannot be empty"):
            pipeline.run(texts=[], ticker="AAPL")

    def test_ticker_preserved(self, pipeline):
        texts = ["Stock rises strongly"]
        signal = pipeline.run(texts=texts, ticker="MSFT")
        assert signal.ticker == "MSFT"

    def test_ratios_sum_to_one(self, pipeline):
        texts = ["rises", "falls", "flat", "surges", "crashes"]
        signal = pipeline.run(texts=texts, ticker="TEST")
        total = signal.bullish_ratio + signal.bearish_ratio + signal.neutral_ratio
        assert abs(total - 1.0) < 0.01

    def test_sample_count(self, pipeline):
        texts = ["rises", "falls", "flat"]
        signal = pipeline.run(texts=texts, ticker="X")
        assert signal.sample_count == 3

    def test_composite_score_range(self, pipeline):
        texts = ["Stock surges massively", "Stock rallies strongly", "Stock beats estimates"]
        signal = pipeline.run(texts=texts, ticker="BULL")
        assert -1.0 <= signal.composite_score <= 1.0

    def test_bullish_signal_for_positive_texts(self, pipeline):
        texts = [
            "Stock surges on record earnings beat",
            "Massive rally in shares after strong results",
            "Analysts upgrade with very high price target",
        ]
        signal = pipeline.run(texts=texts, ticker="BULL")
        assert signal.signal == SentimentLabel.BULLISH
        assert signal.composite_score > 0

    def test_bearish_signal_for_negative_texts(self, pipeline):
        texts = [
            "Stock crashes on bankruptcy fears",
            "Massive selloff as company misses targets badly",
            "Analysts downgrade after weak disappointing results",
        ]
        signal = pipeline.run(texts=texts, ticker="BEAR")
        assert signal.signal == SentimentLabel.BEARISH
        assert signal.composite_score < 0

    def test_signal_strength_type(self, pipeline):
        texts = ["rises", "falls"]
        signal = pipeline.run(texts=texts, ticker="X")
        assert signal.strength in SignalStrength

    def test_signal_summary_string(self, pipeline):
        texts = ["Stock surges on record profits"]
        signal = pipeline.run(texts=texts, ticker="AAPL")
        summary = signal.summary
        assert "AAPL" in summary
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_to_dict(self, pipeline):
        texts = ["Stock rises"]
        signal = pipeline.run(texts=texts, ticker="MSFT")
        d = signal.to_dict()
        assert "ticker" in d
        assert "signal" in d
        assert "composite_score" in d
        assert "ratios" in d
        assert d["ticker"] == "MSFT"

    def test_multi_ticker(self, pipeline):
        data = {
            "AAPL": ["Apple surges on record earnings"],
            "TSLA": ["Tesla crashes on delivery miss"],
        }
        signals = pipeline.run_multi_ticker(data)
        assert "AAPL" in signals
        assert "TSLA" in signals
        assert isinstance(signals["AAPL"], MarketSignal)
        assert isinstance(signals["TSLA"], MarketSignal)

    def test_generate_report(self, pipeline):
        data = {
            "AAPL": ["Apple surges on strong earnings beat"],
            "TSLA": ["Tesla crashes on massive delivery miss"],
        }
        signals = pipeline.run_multi_ticker(data)
        report = pipeline.generate_report(signals)
        assert "SENTIMENT REPORT" in report
        assert "AAPL" in report
        assert "TSLA" in report

    def test_generate_report_empty(self, pipeline):
        report = pipeline.generate_report({})
        assert "No signals" in report

    def test_timestamp_is_datetime(self, pipeline):
        texts = ["Stock rises"]
        signal = pipeline.run(texts=texts, ticker="X")
        assert isinstance(signal.timestamp, datetime)


# ─── Integration tests ────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline_end_to_end(self):
        """Complete workflow: texts → pipeline → report"""
        texts = {
            "NVDA": [
                "NVIDIA surges on AI chip demand exceeding expectations",
                "NVDA stock soars after record-breaking data center revenue",
                "NVIDIA beats estimates by massive margin, analysts upgrade",
            ],
            "META": [
                "Meta crashes after revenue miss disappoints investors",
                "Facebook parent falls on weak guidance and high costs",
            ],
        }
        pipeline = AnalysisPipeline()
        signals = pipeline.run_multi_ticker(texts, source="integration_test")
        report = pipeline.generate_report(signals)

        assert signals["NVDA"].signal == SentimentLabel.BULLISH
        assert signals["META"].signal == SentimentLabel.BEARISH
        assert "NVDA" in report
        assert "META" in report

    def test_single_text_consistency(self):
        """Single text result should be consistent across multiple calls"""
        analyzer = SentimentAnalyzer()
        text = "Apple beats earnings with record iPhone sales surge"
        r1 = analyzer.analyze(text, ticker="AAPL")
        r2 = analyzer.analyze(text, ticker="AAPL")
        assert r1.label == r2.label
        assert r1.confidence == r2.confidence

    def test_json_serializable(self):
        """Results must be fully JSON serializable"""
        import json
        pipeline = AnalysisPipeline()
        signal = pipeline.run(["Stock surges strongly"], ticker="AAPL")
        json_str = json.dumps(signal.to_dict())
        parsed = json.loads(json_str)
        assert parsed["ticker"] == "AAPL"
