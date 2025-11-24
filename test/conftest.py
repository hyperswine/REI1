"""
Test configuration for REI1 parser tests
"""

from error_handling import REI1ParseError
from parsing import REI1Grammar
import pytest
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
