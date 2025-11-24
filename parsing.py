"""
REI1 Programming Language Parser
Comprehensive parser for REI1 language with CST preservation and source spans
"""

from typing import List, Dict, Any, Union, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from functools import partial, reduce
import re

# Import pyparsing with error handling
try:
    from pyparsing import (
        Word, alphas, alphanums, QuotedString, Optional as PyParsingOptional,
        ZeroOrMore, OneOrMore, Literal, Forward, Group, Keyword, ParseException,
        Regex, Suppress, LineEnd, nums, oneOf, ParseResults, ParserElement,
        LineStart, StringEnd, White, SkipTo, restOfLine, Combine, delimitedList,
        infixNotation, opAssoc
    )
    # Enable packrat parsing for performance
    ParserElement.enablePackrat()
except ImportError:
    raise ImportError("pyparsing library not found. Install with: pip install pyparsing")


@dataclass(frozen=True)
class SourceSpan:
    """Source location information for preserving CST"""
    filename: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    text: str = ""

    def __str__(self) -> str:
        if self.start_line == self.end_line:
            return f"{self.filename}:{self.start_line}:{self.start_col}-{self.end_col}"
        return f"{self.filename}:{self.start_line}:{self.start_col}-{self.end_line}:{self.end_col}"


@dataclass(frozen=True)
class Token:
    """REI1 token with source information"""
    type: str
    value: Any
    span: SourceSpan

    def __str__(self) -> str:
        return f"{self.type}({self.value})"


@dataclass(frozen=True)
class CSTNode:
    """Concrete Syntax Tree node preserving all source information"""
    type: str
    value: Any
    children: List['CSTNode'] = field(default_factory=list)
    span: Optional[SourceSpan] = None
    tokens: List[Token] = field(default_factory=list)

    def __str__(self) -> str:
        if self.children:
            children_str = ", ".join(str(child) for child in self.children)
            return f"{self.type}({self.value}, [{children_str}])"
        return f"{self.type}({self.value})"


# Import enhanced error handling
from error_handling import REI1ErrorHandler
# Keep local REI1ParseError for now to avoid signature conflicts


class REI1ParseError(Exception):
    """REI1 parsing error with detailed context"""
    def __init__(self, message: str, span: Optional[SourceSpan] = None, context: str = ""):
        self.message = message
        self.span = span
        self.context = context
        super().__init__(self._format_error())

    def _format_error(self) -> str:
        if self.span:
            result = f"Parse error at {self.span}: {self.message}"
            if self.context:
                result += f"\n  Context: {self.context}"
            return result
        return f"Parse error: {self.message}"


class REI1TokenizerError(Exception):
    """REI1 tokenization error"""
    pass


class REI1Tokenizer:
    """REI1 tokenizer with comprehensive token support"""

    def __init__(self, filename: str = "<input>"):
        self.filename = filename
        self._setup_token_patterns()

    def _setup_token_patterns(self):
        """Setup all token patterns for REI1"""

        # Comments (start with # or //, go to end of line)
        self.comment_pattern = re.compile(r'(#|//).*$', re.MULTILINE)

        # String literals with escape sequences
        self.string_pattern = re.compile(r'"(?:[^"\\]|\\.)*"')

        # Character literals with backticks
        self.char_pattern = re.compile(r'`(?:[^`\\]|\\.)')

        # Numbers (integers and decimals, no leading/trailing dots)
        self.number_pattern = re.compile(r'-?(?:0|[1-9]\d*)(?:\.\d+)?')

        # Identifiers (alphanumeric + unicode + symbols, no leading digits)
        # Supports namespaced identifiers like MyNamespace.function
        self.identifier_pattern = re.compile(r'[a-zA-Z_α-ωΑ-Ω∀-∿][a-zA-Z0-9_α-ωΑ-Ω∀-∿\'-]*(?:\.[a-zA-Z_α-ωΑ-Ω∀-∿][a-zA-Z0-9_α-ωΑ-Ω∀-∿\'-]*)*')

        # Operators and special symbols
        self.operators = {
            # Core operators
            '=', '$', '|>', '_', ':', '|', '->', '=>', '>>', '??',
            # Math operators (can be used as function names)
            '+', '-', '*', '/', '%', '<', '>', '<=', '>=', '==', '/=', '!=',
            # List constructor
            '::',
            # Conditional operators
            '?',
            # Other symbols
            ';', '\\', 'λ', '~=', '++', '--',
        }

        # Keywords (removed Type, Sig, unsafe - now stdlib functions; removed True, False - now constructors)
        self.keywords = {
            'case', 'of', 'where', 'otherwise',
            'do', 'lambda', 'lam', 'if', 'then', 'else',
            'match', 'with', 'try', 'forever', 'module', 'import', 'export',
            'pre', 'post', 'par-map', 'par-fold', 'par-filter', 'get', 'spawn',
            'send', 'recv', 'recv_nowait', 'send_to_actor', 'IO', 'Proc'
        }


        # Delimiters
        self.delimiters = {'(', ')', '[', ']', '{', '}', ',', '.'}

        # Create operator pattern (sorted by length to match longest first)
        operators_sorted = sorted(self.operators, key=len, reverse=True)
        operator_escaped = [re.escape(op) for op in operators_sorted]
        self.operator_pattern = re.compile('|'.join(operator_escaped))

        # Delimiter pattern
        delim_escaped = [re.escape(d) for d in self.delimiters]
        self.delimiter_pattern = re.compile('|'.join(delim_escaped))

        # Whitespace pattern
        self.whitespace_pattern = re.compile(r'\s+')

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize REI1 source code using a principled priority-based approach"""
        tokens = []
        lines = text.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Remove comments first (# or //)
            if '#' in line:
                line = line[:line.index('#')]
            elif '//' in line:
                line = line[:line.index('//')]

            pos = 0
            while pos < len(line):
                # Skip whitespace
                if line[pos].isspace():
                    pos += 1
                    continue

                # Apply priority-based token matching
                token = self._match_token_at_position(line, pos, line_num)
                if token:
                    tokens.append(token)
                    pos += len(token.span.text)
                else:
                    # Unknown character
                    char = line[pos]
                    span = SourceSpan(
                        self.filename, line_num, pos + 1, line_num, pos + 2, char
                    )
                    raise REI1TokenizerError(f"Unknown character '{char}' at {span}")

        return tokens

    def _match_token_at_position(self, line: str, pos: int, line_num: int) -> Optional[Token]:
        """Match a token at a specific position using priority order"""

        # Priority 1: String literals (highest priority, can contain any characters)
        if line[pos] == '"':
            string_match = self.string_pattern.match(line, pos)
            if string_match:
                value = string_match.group(0)
                processed_value = self._process_string_escapes(value[1:-1])
                span = SourceSpan(
                    self.filename, line_num, pos + 1, line_num, pos + len(value) + 1, value
                )
                return Token("STRING", processed_value, span)

        # Priority 2: Character literals
        if line[pos] == '`':
            char_match = self.char_pattern.match(line, pos)
            if char_match:
                value = char_match.group(0)
                processed_value = self._process_char_escapes(value[1:])
                span = SourceSpan(
                    self.filename, line_num, pos + 1, line_num, pos + len(value) + 1, value
                )
                return Token("CHAR", processed_value, span)

        # Priority 3: Numbers (including negative numbers)
        if line[pos].isdigit() or (line[pos] == '-' and pos + 1 < len(line) and line[pos+1].isdigit()):
            num_match = self.number_pattern.match(line, pos)
            if num_match:
                value = num_match.group(0)
                processed_value = float(value) if '.' in value else int(value)
                span = SourceSpan(
                    self.filename, line_num, pos + 1, line_num, pos + len(value) + 1, value
                )
                return Token("NUMBER", processed_value, span)

        # Priority 4: Multi-character operators (longest match first)
        op_match = self.operator_pattern.match(line, pos)
        if op_match:
            value = op_match.group(0)
            span = SourceSpan(
                self.filename, line_num, pos + 1, line_num, pos + len(value) + 1, value
            )
            return Token("OPERATOR", value, span)

        # Priority 5: Delimiters (single characters)
        if line[pos] in self.delimiters:
            value = line[pos]
            span = SourceSpan(
                self.filename, line_num, pos + 1, line_num, pos + 2, value
            )
            return Token("DELIMITER", value, span)

        # Priority 6: Identifiers and keywords (lowest priority)
        id_match = self.identifier_pattern.match(line, pos)
        if id_match:
            value = id_match.group(0)
            span = SourceSpan(
                self.filename, line_num, pos + 1, line_num, pos + len(value) + 1, value
            )
            if value in self.keywords:
                return Token("KEYWORD", value, span)
            else:
                return Token("IDENTIFIER", value, span)

        return None

    def _process_string_escapes(self, s: str) -> str:
        """Process escape sequences in strings"""
        escape_map = {
            'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"',
            '0': '\0', 'a': '\a', 'b': '\b', 'f': '\f', 'v': '\v'
        }

        result = []
        i = 0
        while i < len(s):
            if s[i] == '\\' and i + 1 < len(s):
                next_char = s[i + 1]
                if next_char in escape_map:
                    result.append(escape_map[next_char])
                    i += 2
                else:
                    # Unknown escape, keep as-is
                    result.append(s[i])
                    i += 1
            else:
                result.append(s[i])
                i += 1

        return ''.join(result)

    def _process_char_escapes(self, s: str) -> str:
        """Process escape sequences in character literals"""
        if len(s) == 1:
            return s
        elif len(s) == 2 and s[0] == '\\':
            escape_map = {
                'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '`': '`',
                '0': '\0', 'a': '\a', 'b': '\b', 'f': '\f', 'v': '\v'
            }
            return escape_map.get(s[1], s[1])
        else:
            # Invalid character literal, but let parser handle the error
            return s


class REI1Grammar:
    """REI1 grammar definition using pyparsing"""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._setup_grammar()

    def _setup_grammar(self):
        """Setup the clean REI1 grammar focusing on structure, not tokenization"""

        # Forward declarations for recursive structures
        expression = Forward()
        pattern = Forward()
        type_expr = Forward()
        contract = Forward()

        # Basic tokens (use exact string matching, not regex)
        # Keywords
        case_kw = Keyword("case")
        of_kw = Keyword("of")
        lambda_kw = Keyword("lambda") | Keyword("lam") | Keyword("\\") | Literal("λ")
        where_kw = Keyword("where")
        pre_kw = Keyword("pre")
        post_kw = Keyword("post")
        get_kw = Keyword("get")

        # Literals (simplified - let tokenizer handle the details)
        string_literal = QuotedString('"', escChar='\\').setParseAction(lambda t: ("STRING", t[0]))
        char_literal = Regex(r'`(?:[^`\\]|\\.)').setParseAction(lambda t: ("CHAR", t[0][1:]))
        number = Regex(r'-?(?:0|[1-9]\d*)(?:\.\d+)?').setParseAction(
            lambda t: ("NUMBER", float(str(t[0])) if '.' in str(t[0]) else int(str(t[0])))
        )
        # Note: True and False are now constructors, not boolean literals

        # Identifiers (simplified) - base pattern
        identifier_base = Regex(r'[a-zA-Z_][a-zA-Z0-9_\'-]*(?:\.[a-zA-Z_][a-zA-Z0-9_\'-]*)*')

        # Constructor identifier (starts with uppercase)
        constructor_identifier = Regex(r'[A-Z][a-zA-Z0-9_\'-]*(?:\.[A-Z][a-zA-Z0-9_\'-]*)*')

        # Keywords to exclude from identifiers
        excluded_keywords = (case_kw | of_kw | lambda_kw | where_kw | pre_kw | post_kw | get_kw)

        # Context-specific identifiers with keyword exclusion
        value_identifier = (~excluded_keywords + identifier_base.copy()).setParseAction(lambda t: ("IDENTIFIER", t[0]))
        binding_identifier = ~excluded_keywords + identifier_base.copy()  # Raw string for binding names
        pattern_identifier = ~excluded_keywords + identifier_base.copy()  # Raw string for pattern matching
        pattern_constructor_name = ~excluded_keywords + constructor_identifier.copy()  # Constructor names in patterns

        # Operators (exact matches) - order matters! Longer operators first
        pipeline_op = Literal("|>")
        dollar_op = Literal("$")
        arrow_op = Literal("->")
        double_arrow_op = Literal("=>")
        cons_op = Literal("::")
        append_op = Literal("++")
        double_equals_op = Literal("==")
        not_equal_op = Literal("/=") | Literal("!=")
        approx_equal_op = Literal("~=")
        less_equal_op = Literal("<=")
        greater_equal_op = Literal(">=")
        equals_op = Literal("=")
        colon_op = Literal(":")
        pipe_op = Literal("|")

        # Math operators - multi-character operators defined separately above
        # Single character operators can use oneOf
        math_ops = (
            cons_op | append_op | double_equals_op | not_equal_op | approx_equal_op |
            less_equal_op | greater_equal_op |
            oneOf("+ - * / % < > = ??")
        )        # Special tokens
        placeholder = Literal("_")
        not_implemented = Literal("??")

        # Basic atoms
        atom = (
            not_implemented.setParseAction(lambda t: ("NOT_IMPLEMENTED", "??")) |
            placeholder.setParseAction(lambda t: ("PARTIAL_PLACEHOLDER", "_")) |
            string_literal |
            char_literal |
            number |
            value_identifier
        )

        # Parenthesized expressions
        parenthesized = (Suppress("(") + expression + Suppress(")")).setParseAction(lambda t: ("PARENTHESIZED", t[0]))

        # Tuple expressions (comma-separated values in parentheses)
        tuple_expr = (
            Suppress("(") +
            delimitedList(expression, ",") +
            Suppress(")")
        ).setParseAction(lambda t: ("TUPLE", list(t)) if len(t) > 1 else ("PARENTHESIZED", t[0]))

        # List literals
        list_literal = (
            Suppress("[") +
            PyParsingOptional(delimitedList(expression, ",")) +
            Suppress("]")
        ).setParseAction(lambda t: ("LIST", list(t)))

        # Record literals { key = value, key = value }
        def make_record_field(tokens):
            # tokens[0] should be the identifier string, tokens[1] should be the expression
            key = tokens[0]
            # If key is still a tuple for some reason, extract it
            if isinstance(key, tuple):
                key = key[1] if len(key) > 1 and key[0] == 'IDENTIFIER' else str(key)
            return ("RECORD_FIELD", {"key": key, "value": tokens[1]})

        record_field = (
            binding_identifier + Suppress("=") + expression
        ).setParseAction(make_record_field)

        record_literal = (
            Suppress("{") +
            PyParsingOptional(delimitedList(record_field, ",")) +
            Suppress("}")
        ).setParseAction(lambda t: ("RECORD", list(t)))        # Primary expressions - try tuple first, then parenthesized (to handle both cases)
        primary_expr = record_literal | tuple_expr | list_literal | atom

        # Patterns (simplified)
        pattern_var = pattern_identifier.setParseAction(lambda t: ("PATTERN_VAR", t[0]))
        pattern_wildcard = placeholder.setParseAction(lambda t: ("PATTERN_WILDCARD", "_"))
        pattern_literal = (string_literal | char_literal | number).setParseAction(lambda t: ("PATTERN_LITERAL", t[0]))

        # List patterns
        pattern_empty_list = (
            Suppress("[") + Suppress("]")
        ).setParseAction(lambda t: ("PATTERN_LIST", []))

        # Constructor pattern (parenthesized to avoid ambiguity)
        pattern_constructor = (
            Suppress("(") + pattern_constructor_name + ZeroOrMore(pattern) + Suppress(")")
        ).setParseAction(lambda t: ("PATTERN_CONSTRUCTOR", {"name": t[0], "args": list(t[1:])}))

        # Unparenthesized constructor pattern (for case expressions)
        # This pattern has a capital first letter (constructor name) followed by patterns
        # Only atomic patterns (vars, wildcards, literals) can follow to avoid ambiguity
        pattern_atomic = pattern_literal | pattern_wildcard | pattern_var
        pattern_constructor_unparens = (
            pattern_constructor_name + OneOrMore(pattern_atomic)
        ).setParseAction(lambda t: ("PATTERN_CONSTRUCTOR", {"name": t[0], "args": list(t[1:])}))

        pattern_paren = Suppress("(") + pattern + Suppress(")")

        # List cons pattern (defined after pattern forward reference)
        pattern_list_cons = (
            Suppress("(") + pattern + Suppress("::") + pattern + Suppress(")")
        ).setParseAction(lambda t: ("PATTERN_CONS", {"head": t[0], "tail": t[1]}))

        pattern <<= (
            pattern_empty_list |
            pattern_list_cons |
            pattern_literal |
            pattern_wildcard |
            pattern_paren |
            pattern_constructor_unparens |
            pattern_constructor |
            pattern_var
        )        # Function calls
        function_name = (math_ops | binding_identifier).setParseAction(lambda t: ("FUNCTION_NAME", t[0]))
        function_call = Group(
            function_name + OneOrMore(primary_expr)
        ).setParseAction(lambda t: ("FUNCTION_CALL", list(t[0])))

        # Lambda expressions - now properly ordered before function_call
        lambda_expr = (
            lambda_kw + OneOrMore(pattern_identifier) + Suppress(double_arrow_op) + expression
        ).setParseAction(lambda t: ("LAMBDA", {"params": [("PATTERN_VAR", p) for p in t[1:-1]], "body": t[-1]}))

        # Unsafe expressions - removed, unsafe is now a stdlib function
        # unsafe $ expr is just function application: unsafe(expr)

        # Async expressions
        # REMOVED: no more async/await
        # async_kw = Keyword("async")
        # async_expr = (
        #     async_kw + Suppress(dollar_op) + expression
        # ).setParseAction(lambda t: ("ASYNC", t[1]))

        # Get expressions
        get_expr = (
            get_kw + primary_expr
        ).setParseAction(lambda t: ("GET", t[1]))

        # Built-in IO and Proc functions
        io_read = (
            Keyword("IO") + Suppress(".") + Keyword("read")
        ).setParseAction(lambda t: ("IDENTIFIER", "IO.read"))

        io_write = (
            Keyword("IO") + Suppress(".") + Keyword("write")
        ).setParseAction(lambda t: ("IDENTIFIER", "IO.write"))

        proc_spawn = (
            Keyword("Proc") + Suppress(".") + Keyword("spawn")
        ).setParseAction(lambda t: ("IDENTIFIER", "Proc.spawn"))

        proc_send = (
            Keyword("Proc") + Suppress(".") + Keyword("send")
        ).setParseAction(lambda t: ("IDENTIFIER", "Proc.send"))

        proc_recv = (
            Keyword("Proc") + Suppress(".") + Keyword("recv")
        ).setParseAction(lambda t: ("IDENTIFIER", "Proc.recv"))

        proc_recv_nowait = (
            Keyword("Proc") + Suppress(".") + Keyword("recv_nowait")
        ).setParseAction(lambda t: ("IDENTIFIER", "Proc.recv_nowait"))

        proc_send_to_actor = (
            Keyword("Proc") + Suppress(".") + Keyword("send_to_actor")
        ).setParseAction(lambda t: ("IDENTIFIER", "Proc.send_to_actor"))

        # Parallel operations
        par_map = Keyword("par-map").setParseAction(lambda t: ("IDENTIFIER", "par-map"))
        par_fold = Keyword("par-fold").setParseAction(lambda t: ("IDENTIFIER", "par-fold"))
        par_filter = Keyword("par-filter").setParseAction(lambda t: ("IDENTIFIER", "par-filter"))

        # Update atom to include new built-ins
        atom = (
            not_implemented.setParseAction(lambda t: ("NOT_IMPLEMENTED", "??")) |
            placeholder.setParseAction(lambda t: ("PARTIAL_PLACEHOLDER", "_")) |
            io_read | io_write | proc_spawn | proc_send | proc_recv |
            proc_recv_nowait | proc_send_to_actor |
            par_map | par_fold | par_filter |
            string_literal |
            char_literal |
            number |
            value_identifier
        )

        # Case expressions
        case_arm = (
            pattern + Suppress("=>") + expression
        ).setParseAction(lambda t: ("CASE_ARM", {"pattern": t[0], "body": t[1]}))

        case_expr = (
            case_kw + primary_expr + Suppress(of_kw) + delimitedList(case_arm, ",")
        ).setParseAction(lambda t: ("CASE", {"expr": t[1], "arms": list(t[2:])}))

        # Application expressions (function calls, case expressions, lambdas, get, and atoms)
        # Order matters: lambda_expr before function_call to avoid conflicts
        app_expr = get_expr | case_expr | lambda_expr | function_call | primary_expr

        # Infix binding expressions (for let bindings: x = value)
        def make_infix_binding(tokens):
            items = list(tokens)
            if len(items) == 1:
                return items[0]
            elif len(items) == 3 and items[1] == "=":
                # Convert x = value to a function call: = x value
                return ("FUNCTION_CALL", [("FUNCTION_NAME", "="), items[0], items[2]])
            else:
                raise ValueError(f"Invalid infix binding: {items}")

        infix_binding_expr = (
            app_expr + PyParsingOptional(Literal("=") + app_expr)
        ).setParseAction(make_infix_binding)

        # Dollar application (right-associative)
        dollar_expr = Forward()
        def make_dollar(tokens):
            items = list(tokens)
            if len(items) == 1:
                return items[0]
            return ("DOLLAR_APP", {"function": items[0], "argument": items[1]})

        dollar_expr <<= (infix_binding_expr + PyParsingOptional(Suppress(dollar_op) + dollar_expr)).setParseAction(make_dollar)

        # Semicolon expressions (let expressions) - between dollar and pipeline
        def parse_semicolon_expr(tokens):
            """Parse semicolon expressions into nested let bindings"""
            tokens = list(tokens)

            if len(tokens) == 1:
                return tokens[0]

            # Build nested let expressions from right to left
            result = tokens[-1]  # Last expression after final semicolon

            # Work backwards through the expressions
            for i in range(len(tokens) - 2, -1, -1):
                expr = tokens[i]

                # Check if this is a binding (x = value), now parsed as function call from infix syntax
                if (isinstance(expr, tuple) and len(expr) == 2 and
                    expr[0] == "FUNCTION_CALL" and isinstance(expr[1], list) and
                    len(expr[1]) >= 3 and
                    isinstance(expr[1][0], tuple) and expr[1][0] == ("FUNCTION_NAME", "=") and
                    isinstance(expr[1][1], tuple) and expr[1][1][0] == "IDENTIFIER"):

                    # This is a binding: x = value (from infix syntax)
                    var_name = expr[1][1][1]  # Extract variable name from ("IDENTIFIER", "x")
                    var_value = expr[1][2] if len(expr[1]) > 2 else ("ERROR", "missing_value")

                    result = ("LET_BINDING", {
                        "var": var_name,
                        "value": var_value,
                        "body": result
                    })
                else:
                    # Regular expression (let _ = expr in next)
                    result = ("LET_EXPR", {
                        "expr": expr,
                        "body": result
                    })

            return result

        # Semicolon expression - allow semicolons to separate dollar expressions
        semicolon_expr = (
            dollar_expr + ZeroOrMore(Suppress(";") + dollar_expr)
        ).setParseAction(parse_semicolon_expr)

        # Pipeline expression (left-associative)
        pipeline_expr = semicolon_expr + ZeroOrMore(Suppress(pipeline_op) + semicolon_expr)
        def make_pipeline(tokens):
            if len(tokens) == 1:
                return tokens[0]
            result = tokens[0]
            for func in tokens[1:]:
                result = ("PIPELINE", {"input": result, "function": func})
            return result
        pipeline_expr.setParseAction(make_pipeline)

        # Top-level expression (no semicolons at this level)
        expression <<= pipeline_expr        # Contract expressions - use a simplified expression grammar to avoid circular dependency
        # Contract expressions can be basic comparisons, function calls, and simple operations
        contract_identifier = binding_identifier.setParseAction(lambda t: ("IDENTIFIER", t[0]))

        # Allow operators as function names in contracts (for prefix notation like (/= x y))
        contract_operator = oneOf("/= != == <= >= < > + - * / % mod").setParseAction(lambda t: ("FUNCTION_NAME", t[0]))

        contract_primary = (
            string_literal |
            char_literal |
            number |
            contract_identifier |
            Suppress("(") + Forward() + Suppress(")")  # Forward declaration for nested expressions
        )

        # Forward declare contract expression for recursion
        contract_expr = Forward()
        contract_primary_paren = (
            Suppress("(") + contract_expr + Suppress(")")
        ).setParseAction(lambda t: ("PARENTHESIZED", t[0]))

        # Update contract primary to include parenthesized expressions
        contract_primary = (
            string_literal |
            char_literal |
            number |
            contract_identifier |
            contract_primary_paren
        )

        # Contract function calls can use identifiers OR operators as function names
        contract_function_name = contract_operator | contract_identifier
        contract_function_call = (
            contract_function_name + OneOrMore(contract_primary)
        ).setParseAction(lambda t: ("FUNCTION_CALL", {"name": t[0], "args": list(t[1:])}))

        # Contract operations can work on function calls or primary expressions
        contract_operand = contract_function_call | contract_primary

        contract_operation = (
            contract_operand + oneOf("/= != == <= >= < > + - * /") + contract_operand
        ).setParseAction(lambda t: ("OPERATION", {"left": t[0], "op": t[1], "right": t[2]}))

        # Contract expression is operation or operand
        contract_expr <<= (contract_operation | contract_operand)

        # Contract definition parser
        def parse_contract(tokens):
            """Parse contract with pre and post conditions"""
            result = {"pre": None, "post": None}

            i = 0
            while i < len(tokens):
                if isinstance(tokens[i], str) and tokens[i] == "pre":
                    i += 1  # Skip 'pre'
                    if i < len(tokens):
                        result["pre"] = tokens[i]
                        i += 1
                elif isinstance(tokens[i], str) and tokens[i] == "post":
                    i += 1  # Skip 'post'
                    if i < len(tokens):
                        result["post"] = tokens[i]
                        i += 1
                else:
                    i += 1

            return ("CONTRACT", result)

        contract_def = (
            where_kw + Suppress("{") +
            PyParsingOptional(pre_kw + Suppress(":") + contract_expr + PyParsingOptional(Suppress(","))) +
            PyParsingOptional(post_kw + Suppress(":") + contract_expr) +
            Suppress("}")
        ).setParseAction(parse_contract)
        contract <<= contract_def

        # Type expressions (enhanced to handle parameterized types and function types)
        # Type atom should not match keywords to avoid consuming them inappropriately
        type_identifier = Regex(r'[a-zA-Z_][a-zA-Z0-9_\'-]*(?:\.[a-zA-Z_][a-zA-Z0-9_\'-]*)*')
        type_var_pattern = Regex(r'[a-z][a-zA-Z0-9_]*')

        # Reuse the same excluded_keywords from above
        type_atom = (~excluded_keywords + type_identifier).setParseAction(lambda t: ("TYPE_ATOM", t[0]))
        type_var = (~excluded_keywords + type_var_pattern).setParseAction(lambda t: ("TYPE_VAR", t[0]))

        # Forward declare for recursion
        type_expr_inner = Forward()

        # Parenthesized type expressions - need to distinguish between:
        # (a, b, c) = tuple (comma-separated)
        # (List a) = type application (space-separated)
        # (a -> b) = function type (contains ->)
        # (a) = simple parenthesized type

        def parse_parenthesized_content(s, loc, tokens):
            """Custom parser for parenthesized content to handle both comma and space separation"""
            # This will be called with the raw content inside parentheses
            content = tokens[0]

            # Check if it contains commas (tuple) or arrows (function)
            if "," in str(content):
                # This is comma-separated, should be a tuple
                elements = content.split(",")
                return ("TYPE_TUPLE", [elem.strip() for elem in elements])
            elif "->" in str(content):
                # This contains arrows, should be parsed as function type
                return ("TYPE_PAREN", content)  # Let function parser handle it
            else:
                # This is space-separated or single element
                # Try to determine if it's a type application
                parts = str(content).strip().split()
                if len(parts) == 1:
                    return ("TYPE_PAREN", parts[0])
                elif len(parts) >= 2:
                    # Multiple space-separated parts = type application
                    return ("TYPE_APP", {"constructor": parts[0], "args": parts[1:]})
                else:
                    return ("TYPE_PAREN", content)

        # For now, let's use a simpler approach with manual token handling
        type_paren_simple = (
            Suppress("(") + type_expr_inner + Suppress(")")
        ).setParseAction(lambda t: ("TYPE_PAREN", t[0]))

        type_paren_tuple = (
            Suppress("(") + delimitedList(type_expr_inner, ",") + Suppress(")")
        ).setParseAction(lambda t: ("TYPE_TUPLE", list(t)) if len(t) > 1 else ("TYPE_PAREN", t[0]))

        type_paren_app = (
            Suppress("(") + type_atom + OneOrMore(type_atom | type_var) + Suppress(")")
        ).setParseAction(lambda t: ("TYPE_APP", {"constructor": t[0], "args": list(t[1:])}))

        # Try type application first, then tuple, then simple
        type_paren = type_paren_app | type_paren_tuple | type_paren_simple

        # List type expressions [Type] or [Type, Type, ...] for tuple-like lists
        type_list = (
            Suppress("[") + delimitedList(type_expr_inner, ",") + Suppress("]")
        ).setParseAction(lambda t: ("TYPE_LIST", list(t)) if len(t) > 1 else ("TYPE_LIST", t[0]))

        # Basic type elements (no automatic type application here)
        basic_type_element = type_list | type_paren | type_atom | type_var

        # Type application (explicit - only when we know we want it)
        # This is for cases like List a, Tree (List String), etc.
        type_application = (
            basic_type_element + OneOrMore(basic_type_element)
        ).setParseAction(lambda t: ("TYPE_APP", {"constructor": t[0], "args": list(t[1:])}))

        # Function types - this is the key change
        # In function contexts, multiple types before -> are parameters (tuple style)
        # Type applications only happen in explicit contexts

        def parse_function_signature_type(tokens):
            """Parse type expressions specifically in function signature context

            Supports both:
            - Tuple style: Num Num -> Num (multiple params = tuple)
            - Curried style: Num -> (Num -> Num) (explicit parentheses for currying)
            """
            tokens = list(tokens)

            # Find first -> position at top level (not inside parentheses)
            arrow_pos = None
            paren_depth = 0

            for i, token in enumerate(tokens):
                if token == "(":
                    paren_depth += 1
                elif token == ")":
                    paren_depth -= 1
                elif token == "->" and paren_depth == 0:
                    arrow_pos = i
                    break

            if arrow_pos is None:
                # No arrow - this is a non-function type
                if len(tokens) == 1:
                    return tokens[0]
                else:
                    # Multiple tokens without arrow in function signature context
                    # This should NOT be a type application anymore - that requires parentheses
                    # This might be an error or we should treat it differently
                    # For now, let's assume this shouldn't happen in well-formed signatures
                    return ("TYPE_SEQUENCE", tokens)  # Mark as error/invalid

            # Has arrow - function type
            param_tokens = tokens[:arrow_pos]
            result_tokens = tokens[arrow_pos + 1:]

            # Parse parameter side
            if len(param_tokens) == 0:
                param_type = ("TYPE_ATOM", "Unit")
            elif len(param_tokens) == 1:
                param_type = param_tokens[0]
            else:
                # Multiple parameters = tuple style (A B -> C)
                param_type = ("TYPE_TUPLE", param_tokens)

            # Parse result side recursively (to handle nested function types)
            if len(result_tokens) == 1:
                result_type = result_tokens[0]
            else:
                # Multiple tokens on result side - could be more arrows or invalid
                # Check if there are more arrows in the result
                has_arrow = any(token == "->" for token in result_tokens)
                if has_arrow:
                    # Recursive function type parsing
                    result_type = parse_function_signature_type(result_tokens)
                else:
                    # Multiple tokens without arrows on result side
                    # In the new design, this should be an error - type applications need parentheses
                    result_type = ("TYPE_SEQUENCE", result_tokens)  # Mark as error/invalid

            return ("TYPE_FUNC", {"from": param_type, "to": result_type})

        # Function signature type parser (for use in module signatures)
        function_signature_type = (
            OneOrMore(basic_type_element | arrow_op)
        ).setParseAction(parse_function_signature_type)

        # Regular type expression (for other contexts like return types, type definitions)
        regular_type_expr = type_application | basic_type_element

        # Complete type expression - use function signature type in sig contexts, regular elsewhere
        complete_type_expr = function_signature_type

        # Set the forward reference
        type_expr_inner <<= complete_type_expr
        type_expr <<= complete_type_expr

        # Type definitions (simplified approach)
        type_param = Regex(r'[a-z][a-zA-Z0-9_]*')

        # Type constructor in definition (use individual types, not applications)
        constructor_arg_type = type_list | type_paren | type_atom | type_var

        type_constructor = Group(
            binding_identifier + ZeroOrMore(constructor_arg_type)
        ).setParseAction(lambda t: ("TYPE_CONSTRUCTOR", {"name": t[0][0], "args": list(t[0][1:])}))

        def parse_type_def(tokens):
            """Parse type definition with optional parameters"""
            tokens = list(tokens)
            # Structure: [name, param1, param2, ..., constructor1, constructor2, ...]
            # Find the first TYPE_CONSTRUCTOR tuple to know where constructors start
            name = tokens[0]
            constructor_start = 1

            # Find first constructor (tuple starting with "TYPE_CONSTRUCTOR")
            for i in range(1, len(tokens)):
                if isinstance(tokens[i], tuple) and tokens[i][0] == "TYPE_CONSTRUCTOR":
                    constructor_start = i
                    break

            params = tokens[1:constructor_start]
            constructors = tokens[constructor_start:]

            return ("TYPE_DEF", {
                "name": name,
                "params": params,
                "constructors": constructors
            })

        type_definition = (
            binding_identifier + Suppress(equals_op) + expression
        ).setParseAction(lambda t: ("VALUE_BINDING", {"name": ("IDENTIFIER", t[0]), "value": t[1]}))

        # Parameter patterns for function definitions
        # Typed parameter: (name: Type)
        typed_param = (
            Suppress("(") + pattern_identifier + Suppress(colon_op) + type_expr + Suppress(")")
        ).setParseAction(lambda t: ("TYPED_PARAM", {"name": t[0], "type": t[1]}))

        # Untyped parameter: name or pattern (use a copy to avoid modifying the main pattern parser)
        untyped_param = pattern.copy().setParseAction(lambda t: ("UNTYPED_PARAM", t[0]))

        # Function parameter (typed or untyped)
        function_param = typed_param | untyped_param

        # Return type annotation: -> (name: Type) or -> Type
        return_type_named = (
            Suppress("->") + Suppress("(") + pattern_identifier + Suppress(colon_op) + type_expr + Suppress(")")
        ).setParseAction(lambda t: ("RETURN_TYPE", {"name": t[0], "type": t[1]}))

        return_type_simple = (
            Suppress("->") + type_expr
        ).setParseAction(lambda t: ("RETURN_TYPE", {"name": None, "type": t[0]}))

        return_type = return_type_named | return_type_simple

        # Guard clause: | expression
        # Use contract_expr for the guard expression (simpler than full expression)
        guard_clause = (
            Suppress("|") + contract_expr
        ).setParseAction(lambda t: ("GUARD", t[0]))

        # Function definition with inline types and contracts
        # Syntax: name param1 param2 [| guard] [-> returnType] [where {...}] = body
        def parse_function_def(tokens):
            tokens = list(tokens)
            name = tokens[0]
            params = []
            guard = None
            return_type = None
            contract = None
            body = tokens[-1]

            # Extract components from the middle tokens
            for token in tokens[1:-1]:
                if isinstance(token, tuple):
                    if token[0] in ("TYPED_PARAM", "UNTYPED_PARAM"):
                        params.append(token)
                    elif token[0] == "GUARD":
                        guard = token
                    elif token[0] == "RETURN_TYPE":
                        return_type = token
                    elif token[0] == "CONTRACT":
                        contract = token

            return ("FUNCTION_DEF", {
                "name": name,
                "params": params,
                "guard": guard,
                "return_type": return_type,
                "contract": contract,
                "body": body
            })

        function_definition = (
            binding_identifier +
            ZeroOrMore(function_param) +
            PyParsingOptional(return_type) +
            PyParsingOptional(contract) +
            Suppress(equals_op) +
            expression
        ).setParseAction(parse_function_def)

        # Combined function/value definition parser
        def parse_function_or_value(tokens):
            tokens = list(tokens)
            name = tokens[0]

            # Check if this has parameters (anything between name and = that's not a return type or contract)
            has_params = False
            params = []
            guard = None
            return_type = None
            contract = None
            body = tokens[-1]

            # Scan through tokens to find parameters vs modifiers
            for token in tokens[1:-1]:
                if isinstance(token, tuple):
                    if token[0] in ("TYPED_PARAM", "UNTYPED_PARAM"):
                        has_params = True
                        params.append(token)
                    elif token[0] == "GUARD":
                        guard = token
                    elif token[0] == "RETURN_TYPE":
                        return_type = token
                    elif token[0] == "CONTRACT":
                        contract = token

            if has_params or guard or return_type or contract:
                # This is a function definition
                return ("FUNCTION_DEF", {
                    "name": name,
                    "params": params,
                    "guard": guard,
                    "return_type": return_type,
                    "contract": contract,
                    "body": body
                })
            else:
                # This is a value binding
                return ("VALUE_BINDING", {"name": name, "value": body})

        # Create specific parsers for different function definition forms
        # 1. Function with params, guard, return type, and contract
        function_def_full = (
            binding_identifier +
            OneOrMore(function_param) +
            PyParsingOptional(guard_clause) +
            return_type +
            contract +
            Suppress(equals_op) +
            expression
        ).setParseAction(parse_function_or_value)

        # 2. Function with params, guard, and return type (no contract)
        function_def_typed = (
            binding_identifier +
            OneOrMore(function_param) +
            PyParsingOptional(guard_clause) +
            return_type +
            Suppress(equals_op) +
            expression
        ).setParseAction(parse_function_or_value)

        # 3. Function with params, guard, and contract (no explicit return type)
        function_def_contract = (
            binding_identifier +
            OneOrMore(function_param) +
            PyParsingOptional(guard_clause) +
            contract +
            Suppress(equals_op) +
            expression
        ).setParseAction(parse_function_or_value)

        # 4. Function with just params and optional guard (no return type, no contract)
        function_def_simple = (
            binding_identifier +
            OneOrMore(function_param) +
            PyParsingOptional(guard_clause) +
            Suppress(equals_op) +
            expression
        ).setParseAction(parse_function_or_value)

        # 5. Value binding (no params)
        value_binding = (
            binding_identifier +
            Suppress(equals_op) +
            expression
        ).setParseAction(parse_function_or_value)

        # Try the most specific patterns first
        function_or_value_definition = (
            function_def_full |
            function_def_typed |
            function_def_contract |
            function_def_simple |
            value_binding
        )

        # Module signatures and implementations
        # Module signature - REMOVED, Sig is now a stdlib function if needed
        # sig_member = Group(
        #     binding_identifier + Suppress(colon_op) + type_expr
        # ).setParseAction(lambda t: ("SIG_MEMBER", {"name": t[0][0], "type": t[0][1]}))
        #
        # module_signature = (
        #     binding_identifier + Suppress(equals_op) + Suppress(sig_kw) +
        #     Suppress("(") + PyParsingOptional(delimitedList(sig_member, ",")) + Suppress(")")
        # ).setParseAction(lambda t: ("MODULE_SIG", {"name": t[0], "members": list(t[1:]) if len(t) > 1 else []}))

        # Module implementation (minimal form: ModuleName = SignatureName)
        # This should be more specific to avoid matching value bindings
        # Only match if the right side is a known module/signature identifier pattern
        # For now, make it more restrictive by requiring capitalized identifiers
        module_name_pattern = Regex(r'[A-Z][a-zA-Z0-9_]*(?:\.[A-Z][a-zA-Z0-9_]*)*')
        module_implementation = (
            module_name_pattern + Suppress(equals_op) + module_name_pattern
        ).setParseAction(lambda t: ("MODULE_IMPL", {"name": t[0], "signature": t[1]}))

        # Inline evaluation (> expr.) - evaluate and print in script files
        inline_eval = (
            Suppress(">") + expression
        ).setParseAction(lambda t: ("INLINE_EVAL", t[0]))

        # Top-level bindings (order matters - try most specific first)
        # Functions and values should be tried before module patterns
        binding = (
            function_or_value_definition |
            module_implementation
        )

        # Statements can be bindings or inline evaluations
        statement = (inline_eval | binding) + Suppress(".")
        program = ZeroOrMore(statement) + StringEnd()

        # Store the main parsers
        self.program = program
        self.statement = statement
        self.expression = expression
        self.contract = contract
        self.function_param = function_param
        self.return_type = return_type
        self.function_or_value_definition = function_or_value_definition
        self.binding_identifier = binding_identifier
        self.pattern = pattern
        self.primary_expr = primary_expr

        # Additional parsers for testing
        self.type_expr = type_expr
        # self.module_sig = module_signature  # Removed - Sig is now a stdlib function
        self.function_def = function_or_value_definition  # Alias for compatibility

    def parse_program(self, text: str, filename: str = "<input>") -> List[CSTNode]:
        """Parse a complete REI1 program"""
        try:
            # Preprocess text to remove comments and empty lines
            preprocessed_text = self._preprocess_text(text)

            # Handle empty programs (e.g., files with only comments)
            if not preprocessed_text.strip():
                return []

            result = self.program.parseString(preprocessed_text, parseAll=True)
            return self._convert_to_cst(result, filename)
        except ParseException as e:
            line_num = getattr(e, 'lineno', 1)
            col_num = getattr(e, 'col', 1)
            span = SourceSpan(filename, line_num, col_num, line_num, col_num + 1, "")
            context_line = text.split('\n')[line_num - 1] if line_num <= len(text.split('\n')) else ""
            raise REI1ParseError(str(e), span, context_line)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text to remove comments and clean up whitespace"""
        lines = text.split('\n')
        processed_lines = []

        for line in lines:
            # Remove comments (# or //)
            if '#' in line:
                line = line[:line.index('#')]
            elif '//' in line:
                line = line[:line.index('//')]

            # Strip whitespace
            line = line.strip()

            # Keep non-empty lines
            if line:
                processed_lines.append(line)

        # If no content remains after preprocessing, return empty program
        # This handles files that contain only comments
        result = '\n'.join(processed_lines)
        return result if result else ""

    def parse_expression(self, text: str, filename: str = "<input>") -> CSTNode:
        """Parse a single REI1 expression"""
        try:
            result = self.expression.parseString(text, parseAll=True)
            cst_nodes = self._convert_to_cst(result, filename)
            return cst_nodes[0] if cst_nodes else CSTNode("EMPTY", None)
        except ParseException as e:
            line_num = getattr(e, 'lineno', 1)
            col_num = getattr(e, 'col', 1)
            span = SourceSpan(filename, line_num, col_num, line_num, col_num + 1, "")
            context_line = text.split('\n')[line_num - 1] if line_num <= len(text.split('\n')) else ""
            raise REI1ParseError(str(e), span, context_line)

    def _convert_to_cst(self, parse_result: Any, filename: str) -> List[CSTNode]:
        """Convert pyparsing results to CST nodes"""
        if not parse_result:
            return []

        def convert_item(item, line: int = 1) -> CSTNode:
            if isinstance(item, tuple) and len(item) >= 2:
                node_type, value = item[0], item[1]

                # Create source span (approximate for now)
                span = SourceSpan(filename, line, 1, line, 1, str(item))

                if isinstance(value, dict):
                    # Handle structured nodes
                    children = []
                    for key, val in value.items():
                        if isinstance(val, list):
                            children.extend([convert_item(v, line) for v in val])
                        else:
                            children.append(convert_item(val, line))
                    return CSTNode(node_type, value, children, span)
                elif isinstance(value, list):
                    # Handle list values
                    children = [convert_item(v, line) for v in value]
                    return CSTNode(node_type, None, children, span)
                else:
                    # Handle simple values
                    return CSTNode(node_type, value, [], span)
            else:
                # Handle simple items
                span = SourceSpan(filename, line, 1, line, 1, str(item))
                return CSTNode("UNKNOWN", item, [], span)

        return [convert_item(item) for item in parse_result]


class REI1Parser:
    """Main REI1 parser combining tokenizer and grammar"""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.grammar = REI1Grammar(debug)

    def parse_file(self, filepath: str) -> List[CSTNode]:
        """Parse a REI1 source file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.grammar.parse_program(content, filepath)
        except FileNotFoundError:
            raise REI1ParseError(f"File not found: {filepath}")
        except UnicodeDecodeError as e:
            raise REI1ParseError(f"Cannot decode file {filepath}: {e}")

    def parse_string(self, text: str, filename: str = "<input>") -> List[CSTNode]:
        """Parse REI1 source code from string"""
        return self.grammar.parse_program(text, filename)

    def parse_expression(self, text: str, filename: str = "<input>") -> CSTNode:
        """Parse a single REI1 expression"""
        return self.grammar.parse_expression(text, filename)

    def tokenize(self, text: str, filename: str = "<input>") -> List[Token]:
        """Tokenize REI1 source code"""
        tokenizer = REI1Tokenizer(filename)
        return tokenizer.tokenize(text)


# Factory functions for creating parsers
def create_parser(debug: bool = False) -> REI1Parser:
    """Create a REI1 parser"""
    return REI1Parser(debug=debug)


def create_debug_parser() -> REI1Parser:
    """Create a REI1 parser with debug enabled"""
    return REI1Parser(debug=True)


# Utility functions for working with CST
def find_nodes_by_type(cst: CSTNode, node_type: str) -> List[CSTNode]:
    """Find all nodes of a specific type in CST"""
    result = []

    def search(node: CSTNode):
        if node.type == node_type:
            result.append(node)
        for child in node.children:
            search(child)

    search(cst)
    return result


def pretty_print_cst(cst: CSTNode, indent: int = 0) -> str:
    """Pretty print a CST node for debugging"""
    result = "  " * indent + f"{cst.type}"
    if cst.value is not None:
        result += f"({repr(cst.value)})"
    result += "\n"

    for child in cst.children:
        result += pretty_print_cst(child, indent + 1)

    return result


def cst_to_dict(cst: CSTNode) -> Dict[str, Any]:
    """Convert CST to dictionary representation"""
    result = {
        "type": cst.type,
        "value": cst.value,
        "span": {
            "filename": cst.span.filename,
            "start_line": cst.span.start_line,
            "start_col": cst.span.start_col,
            "end_line": cst.span.end_line,
            "end_col": cst.span.end_col,
        } if cst.span else None,
        "children": [cst_to_dict(child) for child in cst.children]
    }
    return result


if __name__ == "__main__":
    # Example usage and testing
    parser = create_debug_parser()

    # Test simple expression
    try:
        test_expr = "add 1 2"
        result = parser.parse_expression(test_expr)
        print("Expression parse result:")
        print(pretty_print_cst(result))
    except REI1ParseError as e:
        print(f"Parse error: {e}")

    # Test simple program
    try:
        test_program = """
        # Simple REI1 program
        add x y = + x y.
        result = add 3 4.
        """
        results = parser.parse_string(test_program)
        print("\nProgram parse result:")
        for result in results:
            print(pretty_print_cst(result))
    except REI1ParseError as e:
        print(f"Parse error: {e}")
