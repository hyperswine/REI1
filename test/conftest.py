"""
Test configuration for REI1 parser tests
"""

import pytest
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from parsing import REI1Grammar
from error_handling import REI1ParseError
