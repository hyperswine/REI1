"""
Basic parsing tests for REI1 language
Tests fundamental parsing capabilities
"""

import pytest
from parsing import REI1Grammar
from error_handling import REI1ParseError
from pyparsing import ParseException


class TestBasicParsing:
  """Test basic parsing functionality"""

  @pytest.fixture
  def grammar(self):
    """Provide a fresh grammar instance for each test"""
    return REI1Grammar()

  def test_simple_program_parsing(self, grammar):
    """Test parsing of a simple program"""
    code = "myValue = 42."
    result = grammar.program.parseString(code)
    assert len(result) == 1

  def test_function_definition_parsing(self, grammar):
    """Test parsing of function definitions"""
    code = "myFunc (x : Num) -> Num = x."
    result = grammar.program.parseString(code)
    assert len(result) == 1

    # Should be a function definition
    binding = result[0]
    assert binding[0] == 'FUNCTION_DEF'

  def test_expression_parsing(self, grammar):
    """Test parsing of expressions"""
    # Simple identifier
    result = grammar.expression.parseString("myVar")
    assert result[0][0] == 'IDENTIFIER'

    # Function call
    result = grammar.expression.parseString("f x")
    assert result[0][0] == 'FUNCTION_CALL'

  # def test_pattern_parsing(self, grammar):
  #     """Test parsing of patterns"""
  #     # Variable pattern
  #     result = grammar.pattern.parseString("x")
  #     assert result[0][0] == 'PATTERN_VAR'

  #     # Wildcard pattern
  #     result = grammar.pattern.parseString("_")
  #     assert result[0][0] == 'PATTERN_WILDCARD'


class TestTypeExpressions:
  """Test type expression parsing"""

  @pytest.fixture
  def grammar(self):
    return REI1Grammar()

  def test_function_type_parsing(self, grammar):
    """Test parsing of function types using expression parser"""
    # For now, test through function definitions that include types
    code = "myFunc (x : Num) -> String = x."
    result = grammar.program.parseString(code)
    assert len(result) == 1

    # Check that it parsed as a function definition
    binding = result[0]
    assert binding[0] == 'FUNCTION_DEF'


class TestPatterns:
  """Test pattern parsing"""

  @pytest.fixture
  def grammar(self):
    return REI1Grammar()

  def test_pattern_variable(self, grammar):
    """Test variable patterns"""
    result = grammar.pattern.parseString("x")
    assert result[0][0] == 'PATTERN_VAR'
    assert result[0][1] == 'x'

  # def test_pattern_wildcard(self, grammar):
  #     """Test wildcard patterns"""
  #     result = grammar.pattern.parseString("_")
  #     assert result[0][0] == 'PATTERN_WILDCARD'
  #     assert result[0][1] == '_'

  def test_pattern_empty_list(self, grammar):
    """Test empty list patterns"""
    result = grammar.pattern.parseString("[]")
    assert result[0][0] == 'PATTERN_LIST'
    assert result[0][1] == []


class TestCaseExpressions:
  """Test case expression parsing"""

  @pytest.fixture
  def grammar(self):
    return REI1Grammar()

  def test_simple_case(self, grammar):
    """Test simple case expressions"""
    code = """case n of
            0 => "zero",
            1 => "one",
            _ => "other"
        """
    result = grammar.expression.parseString(code)
    assert result[0][0] == 'CASE'


class TestContracts:
  """Test contract parsing"""

  @pytest.fixture
  def grammar(self):
    return REI1Grammar()

  def test_simple_contract(self, grammar):
    """Test simple contracts"""
    code = "where { pre: x > 0, post: result > 0 }"
    result = grammar.contract.parseString(code)
    assert result[0][0] == 'CONTRACT'


class TestErrorHandling:
  """Test error handling and reporting"""

  @pytest.fixture
  def grammar(self):
    return REI1Grammar()

  def test_invalid_syntax_error(self, grammar):
    """Test that invalid syntax produces helpful errors"""
    with pytest.raises(ParseException):
      grammar.program.parseString("invalid syntax here")

  def test_incomplete_function_error(self, grammar):
    """Test error on incomplete function definition"""
    with pytest.raises(ParseException):
      grammar.program.parseString("myFunc (x : Num) =")  # Missing body
