"""
Enhanced error handling for REI1 parser with detailed error messages
Pure functional style - no classes except for compatibility
"""

from typing import List, Optional, Dict
from pyparsing import ParseException
import re


# ============================================================================
# DATA STRUCTURES (Immutable Dictionaries)
# ============================================================================

def make_parse_error(
    message: str,
    location: int,
    line: int,
    column: int,
    expected: Optional[List[str]] = None,
    got: Optional[str] = None,
    context: Optional[str] = None,
    suggestions: Optional[List[str]] = None
) -> Dict:
    """Create an immutable parse error structure"""
    return {
        'message': message,
        'location': location,
        'line': line,
        'column': column,
        'expected': expected or [],
        'got': got,
        'context': context,
        'suggestions': suggestions or []
    }


def format_parse_error(error: Dict) -> str:
    """Format parse error as string"""
    error_msg = f"Parse error at line {error['line']}, column {error['column']}:\n"
    error_msg += f"  {error['message']}\n"

    if error['expected']:
        error_msg += f"  Expected: {', '.join(error['expected'])}\n"

    if error['got']:
        error_msg += f"  Got: {error['got']}\n"

    if error['context']:
        error_msg += f"  Context: {error['context']}\n"

    if error['suggestions']:
        error_msg += f"  Suggestions:\n"
        for suggestion in error['suggestions']:
            error_msg += f"    - {suggestion}\n"

    return error_msg


# ============================================================================
# PURE FUNCTIONS
# ============================================================================

def get_context_lines(source_text: str, line_num: int, col_num: int, context_lines: int = 2) -> str:
    """Get context lines around the error"""
    lines = source_text.split('\n')
    start_line = max(0, line_num - context_lines - 1)
    end_line = min(len(lines), line_num + context_lines)

    context_parts = []
    for i in range(start_line, end_line):
        line_prefix = f"{i+1:4d}: "
        if i == line_num - 1:  # Error line
            context_parts.append(f"{line_prefix}{lines[i]}")
            context_parts.append(f"{'':6}{' ' * (col_num - 1)}^ Error here")
        else:
            context_parts.append(f"{line_prefix}{lines[i]}")

    return '\n'.join(context_parts)


def extract_expected(exc: ParseException) -> List[str]:
    """Extract expected tokens from exception"""
    expected = []

    # Parse from message (pyparsing doesn't always have .expected attribute)
    msg = str(exc)
    if "Expected" in msg:
        # Extract patterns like "Expected {pattern}"
        expected_match = re.search(r"Expected\s+(.+?)(?:\s+\(|$)", msg)
        if expected_match:
            expected.append(expected_match.group(1))

    return expected if expected else ["valid syntax"]


def extract_got(source_text: str, line_num: int, col_num: int) -> str:
    """Extract what was actually found at the error location"""
    lines = source_text.split('\n')

    if line_num <= len(lines):
        error_line = lines[line_num - 1]
        if col_num <= len(error_line):
            # Get a few characters around the error position
            start = max(0, col_num - 1)
            end = min(len(error_line), col_num + 10)
            got_text = error_line[start:end].strip()
            if got_text:
                return f"'{got_text}'"
            else:
                return "end of line"
    return "unknown"


def generate_suggestions(exc: ParseException, got: str, expected: List[str]) -> List[str]:
    """Generate helpful suggestions based on the error"""
    suggestions = []

    # Common syntax suggestions
    if ";" in got:
        suggestions.append("REI1 doesn't use semicolons - try removing them")

    if "{" in got or "}" in got:
        suggestions.append("Use parentheses () instead of braces {} in REI1")

    if "Expected identifier" in str(expected):
        suggestions.append("Function and variable names must start with a letter")

    if "Expected '->'" in str(expected):
        suggestions.append("Function types need '->' between parameter and return types")

    if "where" in str(expected):
        suggestions.append("Contracts must be followed by 'where' keyword")

    if "case" in got.lower():
        suggestions.append("Case expressions need 'of' keyword after the expression")

    return suggestions


def enhance_parse_exception_dict(exc: ParseException, source_text: str) -> Dict:
    """Convert pyparsing exception to enhanced REI1 error dict"""
    line_num = exc.lineno
    col_num = exc.column

    # Get context around the error
    context = get_context_lines(source_text, line_num, col_num)

    # Extract what was expected
    expected = extract_expected(exc)

    # Extract what was actually found
    got = extract_got(source_text, line_num, col_num)

    # Generate suggestions
    suggestions = generate_suggestions(exc, got, expected)

    return make_parse_error(
        message=str(exc),
        location=exc.loc,
        line=line_num,
        column=col_num,
        expected=expected,
        got=got,
        context=context,
        suggestions=suggestions
    )


# ============================================================================
# COMPATIBILITY CLASSES (for existing code)
# ============================================================================

class REI1ParseError(Exception):
    """Exception class for compatibility with existing code"""
    def __init__(self, message: str, location: int = 0, line: int = 0, column: int = 0,
                 expected: Optional[List[str]] = None, got: Optional[str] = None,
                 context: Optional[str] = None, suggestions: Optional[List[str]] = None):
        self.message = message
        self.location = location
        self.line = line
        self.column = column
        self.expected = expected or []
        self.got = got
        self.context = context
        self.suggestions = suggestions or []
        super().__init__(message)

    def __str__(self) -> str:
        error_dict = make_parse_error(
            self.message, self.location, self.line, self.column,
            self.expected, self.got, self.context, self.suggestions
        )
        return format_parse_error(error_dict)


class REI1ErrorHandler:
    """Compatibility class - delegates to functional implementation"""
    def __init__(self, source_text: str, filename: str = "<input>"):
        self.source_text = source_text
        self.filename = filename
        self.lines = source_text.split('\n')

    def enhance_parse_exception(self, exc: ParseException) -> REI1ParseError:
        """Convert pyparsing exception to enhanced REI1 error"""
        error_dict = enhance_parse_exception_dict(exc, self.source_text)
        return REI1ParseError(
            message=error_dict['message'],
            location=error_dict['location'],
            line=error_dict['line'],
            column=error_dict['column'],
            expected=error_dict['expected'],
            got=error_dict['got'],
            context=error_dict['context'],
            suggestions=error_dict['suggestions']
        )

    def _get_context(self, line_num: int, col_num: int, context_lines: int = 2) -> str:
        return get_context_lines(self.source_text, line_num, col_num, context_lines)

    def _extract_expected(self, exc: ParseException) -> List[str]:
        return extract_expected(exc)

    def _extract_got(self, exc: ParseException, line_num: int, col_num: int) -> str:
        return extract_got(self.source_text, line_num, col_num)

    def _generate_suggestions(self, exc: ParseException, got: str, expected: List[str]) -> List[str]:
        return generate_suggestions(exc, got, expected)


def create_enhanced_parser_with_errors(parser_func, source_text: str, filename: str = "<input>"):
    """Wrapper to add enhanced error handling to any parser"""
    def enhanced_parse(*args, **kwargs):
        try:
            return parser_func(*args, **kwargs)
        except ParseException as e:
            error_dict = enhance_parse_exception_dict(e, source_text)
            raise REI1ParseError(
                message=error_dict['message'],
                location=error_dict['location'],
                line=error_dict['line'],
                column=error_dict['column'],
                expected=error_dict['expected'],
                got=error_dict['got'],
                context=error_dict['context'],
                suggestions=error_dict['suggestions']
            ) from e

    return enhanced_parse
