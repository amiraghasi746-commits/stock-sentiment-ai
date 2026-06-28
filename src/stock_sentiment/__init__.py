"""
Stock Market Sentiment Analyzer
AI-powered NLP tool for financial market sentiment analysis
"""

__version__ = "1.0.0"
__author__ = "Stock Sentiment AI"

from .analyzer import SentimentAnalyzer
from .models import SentimentResult, MarketSignal
from .pipeline import AnalysisPipeline

__all__ = ["SentimentAnalyzer", "SentimentResult", "MarketSignal", "AnalysisPipeline"]
