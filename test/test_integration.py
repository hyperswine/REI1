"""
Integration tests for REI1 parser using real example files
"""

import pytest
from pathlib import Path
from parsing import REI1Grammar, REI1ParseError
from error_handling import REI1ErrorHandler
from pyparsing import ParseException


class TestFileIntegration:
  """Test parsing of complete example files"""

  @pytest.fixture
  def grammar(self):
    return REI1Grammar()

  @pytest.fixture
  def examples_dir(self):
    """Get the examples directory path"""
    return Path(__file__).parent.parent / "examples"

  @pytest.fixture
  def test_files_dir(self):
    """Get the test files directory path"""
    return Path(__file__).parent.parent

  def test_contracts_patterns_file(self, grammar, test_files_dir):
    """Test the test_contracts_patterns.rei1 file"""
    test_file = test_files_dir / "test_contracts_patterns.rei1"

    if not test_file.exists():
      pytest.skip(f"Test file {test_file} not found")

    with open(test_file, 'r') as f:
      content = f.read()

    try:
      result = grammar.program.parseString(content, parseAll=True)
      # Should parse successfully and have 7 bindings
      assert len(result) == 7

      # Check that first binding is a module signature
      first_binding = result[0]
      assert first_binding[0] == 'MODULE_SIG'

    except ParseException as e:
      error_handler = REI1ErrorHandler(content, str(test_file))
      enhanced_error = error_handler.enhance_parse_exception(e)
      pytest.fail(f"Failed to parse {test_file}:\n{enhanced_error}")

  def test_both_styles_file(self, grammar, test_files_dir):
    """Test the test_both_styles.rei1 file"""
    test_file = test_files_dir / "test_both_styles.rei1"

    if not test_file.exists():
      pytest.skip(f"Test file {test_file} not found")

    with open(test_file, 'r') as f:
      content = f.read()

    try:
      result = grammar.program.parseString(content, parseAll=True)
      # Should parse successfully
      assert len(result) >= 1

      # Should be a module signature with function types
      first_binding = result[0]
      assert first_binding[0] == 'MODULE_SIG'

    except ParseException as e:
      error_handler = REI1ErrorHandler(content, str(test_file))
      enhanced_error = error_handler.enhance_parse_exception(e)
      pytest.fail(f"Failed to parse {test_file}:\n{enhanced_error}")

  def test_simple_function_file(self, grammar, test_files_dir):
    """Test the test_simple_function.rei1 file"""
    test_file = test_files_dir / "test_simple_function.rei1"

    if not test_file.exists():
      pytest.skip(f"Test file {test_file} not found")

    with open(test_file, 'r') as f:
      content = f.read()

    try:
      result = grammar.program.parseString(content, parseAll=True)
      assert len(result) >= 1

    except ParseException as e:
      error_handler = REI1ErrorHandler(content, str(test_file))
      enhanced_error = error_handler.enhance_parse_exception(e)
      pytest.fail(f"Failed to parse {test_file}:\n{enhanced_error}")

  def test_examples_directory(self, grammar, examples_dir):
    """Test all .rei1 files in the examples directory"""
    if not examples_dir.exists():
      pytest.skip("Examples directory not found")

    rei1_files = list(examples_dir.glob("**/*.rei1"))

    if not rei1_files:
      pytest.skip("No .rei1 files found in examples directory")

    # Skip design document files that aren't meant to be fully parseable
    design_docs = ['webserver.rei1',
                   'bigexample.rei1', 'actos.rei1', 'modules.rei1']

    failed_files = []

    for rei1_file in rei1_files:
      # Skip design document files
      if rei1_file.name in design_docs:
        print(f"⊘ Skipped {rei1_file.name} (design document)")
        continue

      with open(rei1_file, 'r') as f:
        content = f.read()

      try:
        # Use the parse_program method which handles preprocessing correctly
        result = grammar.parse_program(content, str(rei1_file))
        print(
            f"✓ Successfully parsed {rei1_file.name} ({len(result)} bindings)")

      except REI1ParseError as e:
        failed_files.append((rei1_file, str(e)))
        print(f"✗ Failed to parse {rei1_file.name}")
      except ParseException as e:
        error_handler = REI1ErrorHandler(content, str(rei1_file))
        enhanced_error = error_handler.enhance_parse_exception(e)
        failed_files.append((rei1_file, enhanced_error))
        print(f"✗ Failed to parse {rei1_file.name}")

    # Report any failures
    if failed_files:
      failure_msg = "Failed to parse the following files:\n"
      for file, error in failed_files:
        failure_msg += f"\n{file.name}:\n{error}\n"
      pytest.fail(failure_msg)


class TestSpecificFeatures:
  """Test specific language features in isolation"""

  @pytest.fixture
  def grammar(self):
    return REI1Grammar()

  def DISABLED_test_module_signature_parsing(self, grammar):
    """Test module signature parsing - DISABLED: Sig is now a stdlib function"""
    code = """TestModule = Sig (
            add : Num -> Num -> Num,
            concat : String -> String -> String
        )."""

    try:
      result = grammar.module_sig.parseString(code)
      assert result[0][0] == 'MODULE_SIG'

      # Check that we have the expected functions
      members = result[0][1]['members']
      assert len(members) == 2

    except ParseException as e:
      pytest.fail(f"Failed to parse module signature: {e}")

  def test_function_with_contract(self, grammar):
    """Test function definition with contract"""
    code = """safeDivide (x : Num) (y : Num) -> Num where { pre: y /= 0, post: result > 0 } = / x y"""

    try:
      result = grammar.function_def.parseString(code)
      assert result[0][0] == 'FUNCTION_DEF'

      # Check that contract exists
      func_data = result[0][1]
      assert func_data['contract'] is not None

    except ParseException as e:
      pytest.fail(f"Failed to parse function with contract: {e}")

  def test_curried_function_types(self, grammar):
    """Test curried function type parsing"""
    # Simple curried type
    result = grammar.type_expr.parseString("Num -> (Num -> Num)")
    assert result[0][0] == 'TYPE_FUNC'

    # Complex curried type
    result = grammar.type_expr.parseString("(a -> b) -> (List a -> List b)")
    assert result[0][0] == 'TYPE_FUNC'
