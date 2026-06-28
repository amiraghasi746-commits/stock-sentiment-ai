"""
pytest configuration and shared fixtures
"""
import sys
from pathlib import Path

# Make sure src/ is on the path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
