"""
Command-line interface for Stock Sentiment Analyzer
"""

import argparse
import json
import sys
import logging
from pathlib import Path

from stock_sentiment import SentimentAnalyzer, AnalysisPipeline


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_analyze(args):
    """Analyze a single text"""
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze(args.text, ticker=args.ticker, source=args.source)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        label_emoji = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}
        emoji = label_emoji.get(result.label.value, "")
        print(f"\n{emoji}  Sentiment: {result.label.value.upper()}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Scores: Bullish={result.scores['bullish']:.2f} | "
              f"Bearish={result.scores['bearish']:.2f} | "
              f"Neutral={result.scores['neutral']:.2f}")
        if result.ticker:
            print(f"   Ticker: ${result.ticker}")
        print()


def cmd_pipeline(args):
    """Run pipeline on a file of texts"""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    texts = [line.strip() for line in input_path.read_text().splitlines() if line.strip()]
    if not texts:
        print("Error: No texts found in input file.", file=sys.stderr)
        sys.exit(1)

    pipeline = AnalysisPipeline()
    signal = pipeline.run(texts=texts, ticker=args.ticker, source="file")

    if args.json:
        print(json.dumps(signal.to_dict(), indent=2))
    else:
        print(f"\n{signal.summary}")
        print(f"  Samples analyzed: {signal.sample_count}")
        print(f"  Average confidence: {signal.avg_confidence:.1%}")
        print(f"  Composite score: {signal.composite_score:+.3f}")
        print()


def cmd_demo(args):
    """Run a built-in demo"""
    demo_data = {
        "AAPL": [
            "Apple beats earnings expectations with record iPhone sales surge",
            "AAPL stock rallies strongly after strong quarterly results",
            "Apple faces antitrust investigation from EU regulators",
            "iPhone 16 demand disappoints analysts, stock drops",
            "Apple announces massive $110B share buyback program",
        ],
        "TSLA": [
            "Tesla crashes on disappointing delivery numbers miss",
            "Elon Musk sells TSLA shares again, stock tumbles",
            "Tesla reports record revenue and strong profit growth",
            "Tesla faces recall of 200,000 vehicles over safety concern",
            "Tesla Cybertruck production ramps up significantly",
        ],
        "NVDA": [
            "NVIDIA surges on AI chip demand, revenue skyrockets",
            "NVDA stock soars after record-breaking data center growth",
            "NVIDIA announces new H200 GPU breakthrough for AI",
            "NVIDIA beats estimates by massive margin",
            "Analysts upgrade NVDA with very high price target",
        ],
    }

    pipeline = AnalysisPipeline()
    signals = pipeline.run_multi_ticker(demo_data, source="demo")
    report = pipeline.generate_report(signals)
    print(report)


def main():
    parser = argparse.ArgumentParser(
        description="🤖 Stock Market Sentiment Analyzer — AI-powered financial NLP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single headline
  python -m stock_sentiment analyze "Apple surges on record earnings" --ticker AAPL

  # Run pipeline on a file of texts
  python -m stock_sentiment pipeline headlines.txt --ticker TSLA

  # Run built-in demo
  python -m stock_sentiment demo
        """
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze subcommand
    p_analyze = subparsers.add_parser("analyze", help="Analyze a single text")
    p_analyze.add_argument("text", help="Text to analyze")
    p_analyze.add_argument("--ticker", "-t", help="Stock ticker symbol")
    p_analyze.add_argument("--source", "-s", default="cli", help="Source label")
    p_analyze.add_argument("--json", action="store_true", help="Output as JSON")
    p_analyze.set_defaults(func=cmd_analyze)

    # pipeline subcommand
    p_pipeline = subparsers.add_parser("pipeline", help="Run pipeline on a text file")
    p_pipeline.add_argument("input", help="Path to text file (one item per line)")
    p_pipeline.add_argument("--ticker", "-t", required=True, help="Stock ticker symbol")
    p_pipeline.add_argument("--json", action="store_true", help="Output as JSON")
    p_pipeline.set_defaults(func=cmd_pipeline)

    # demo subcommand
    p_demo = subparsers.add_parser("demo", help="Run built-in demo with sample data")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
