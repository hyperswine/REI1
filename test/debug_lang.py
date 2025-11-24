"""
Generalized debugger for REI1 language parsing
Pure functional style - functions and immutable data
"""

import sys
import os
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from parsing import REI1Grammar
from error_handling import REI1ErrorHandler, REI1ParseError, enhance_parse_exception_dict
from pyparsing import ParseException


# ============================================================================
# DATA STRUCTURES (Immutable Dictionaries)
# ============================================================================

def make_debug_session(
    test_file: str,
    grammar_component: Optional[str] = None,
    start_pos: int = 0,
    end_pos: Optional[int] = None,
    show_tokens: bool = True,
    show_ast: bool = True,
    show_context: bool = True,
    step_by_step: bool = False
) -> Dict:
    """Create an immutable debug session configuration"""
    return {
        'test_file': test_file,
        'grammar_component': grammar_component,
        'start_pos': start_pos,
        'end_pos': end_pos,
        'show_tokens': show_tokens,
        'show_ast': show_ast,
        'show_context': show_context,
        'step_by_step': step_by_step
    }


# ============================================================================
# PURE DEBUGGING FUNCTIONS
# ============================================================================

def read_file_content(filepath: str) -> Optional[str]:
    """Read file content, return None if not found"""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return None


def extract_debug_content(content: str, start_pos: int, end_pos: Optional[int]) -> str:
    """Extract portion of content to debug"""
    if end_pos is not None:
        return content[start_pos:end_pos]
    else:
        return content[start_pos:]


def debug_tokenization(content: str, verbose: bool = True) -> Dict[str, Any]:
    """Debug tokenization process - pure function"""
    if verbose:
        print("\n=== TOKENIZATION DEBUG ===")

    try:
        # Basic character analysis
        if verbose:
            print(f"Characters around positions:")
            for i in range(min(len(content), 200)):
                char = content[i]
                if char in ' \t\n\r':
                    char_repr = repr(char)
                else:
                    char_repr = char
                print(f"  {i:3d}: {char_repr}")

        return {"status": "success", "length": len(content)}

    except Exception as e:
        if verbose:
            print(f"Tokenization error: {e}")
        return {"status": "error", "error": str(e)}


def debug_component(grammar: REI1Grammar, content: str, component_name: str, verbose: bool = True) -> Dict[str, Any]:
    """Debug a specific grammar component - pure function"""
    if verbose:
        print(f"\n=== COMPONENT DEBUG: {component_name} ===")

    try:
        # Get the grammar component
        if hasattr(grammar, component_name):
            component = getattr(grammar, component_name)
            if verbose:
                print(f"Testing component: {component}")

            # Try to parse with the component
            result = component.parseString(content, parseAll=True)
            if verbose:
                print(f"SUCCESS: {result}")
            return {"status": "success", "result": result.asList()}

        else:
            if verbose:
                print(f"Component '{component_name}' not found in grammar")
                available_components = [attr for attr in dir(grammar)
                                      if not attr.startswith('_') and callable(getattr(grammar, attr))]
                print(f"Available components: {available_components}")
            return {"status": "error", "error": "component_not_found",
                   "available": [attr for attr in dir(grammar)
                               if not attr.startswith('_') and callable(getattr(grammar, attr))]}

    except ParseException as e:
        error_dict = enhance_parse_exception_dict(e, content)
        if verbose:
            from error_handling import format_parse_error
            print(f"PARSE ERROR:\n{format_parse_error(error_dict)}")
        return {"status": "error", "error": "parse_error", "details": error_dict}
    except Exception as e:
        if verbose:
            print(f"UNEXPECTED ERROR: {e}")
        return {"status": "error", "error": "unexpected", "details": str(e)}


def find_parse_progress(grammar: REI1Grammar, content: str, verbose: bool = True) -> Dict[str, Any]:
    """Find how far parsing progressed before failing - pure function"""
    if verbose:
        print("\n--- Finding parse progress ---")

    # Try parsing progressively smaller chunks
    max_length = len(content)
    working_length = 0

    for length in range(1, max_length + 1):
        try:
            chunk = content[:length]
            grammar.program.parseString(chunk, parseAll=True)
            working_length = length
        except:
            break

    if verbose:
        print(f"Successfully parsed up to character {working_length}")
        if working_length < max_length:
            problem_start = max(0, working_length - 20)
            problem_end = min(max_length, working_length + 20)
            problem_area = content[problem_start:problem_end]
            print(f"Problem area around char {working_length}:")
            print(f"'{problem_area}'")

    return {"working_length": working_length, "total_length": max_length}


def debug_full_parse(grammar: REI1Grammar, content: str, filename: str, verbose: bool = True) -> Dict[str, Any]:
    """Debug full file parsing - pure function"""
    if verbose:
        print(f"\n=== FULL PARSE DEBUG ===")

    try:
        # Use the proper parse_program method which includes preprocessing
        result = grammar.parse_program(content, filename)
        if verbose:
            print(f"SUCCESS: Parsed {len(result)} top-level items")
        return {"status": "success", "items": len(result)}

    except Exception as e:
        # Handle both ParseException and REI1ParseError
        if hasattr(e, 'lineno'):
            # This is a ParseException
            if isinstance(e, ParseException):
                error_dict = enhance_parse_exception_dict(e, content)
                if verbose:
                    from error_handling import format_parse_error
                    print(f"PARSE ERROR:\n{format_parse_error(error_dict)}")

                # Try progressive parsing to find how far we got
                progress = find_parse_progress(grammar, content, verbose=False)
                return {
                    "status": "error",
                    "error": "parse_error",
                    "details": error_dict,
                    "progress": progress
                }

        # Handle other errors (like REI1ParseError)
        if verbose:
            print(f"PARSE ERROR: {e}")

        # Try progressive parsing to find how far we got
        progress = find_parse_progress(grammar, content, verbose=False)
        return {
            "status": "error",
            "error": "parse_error",
            "details": str(e),
            "progress": progress
        }


def debug_file_session(session: Dict, verbose: bool = True) -> Dict[str, Any]:
    """Debug a complete file or portion of it - pure function"""
    # Read the test file
    content = read_file_content(session['test_file'])
    if content is None:
        if verbose:
            print(f"Error: File {session['test_file']} not found")
        return {"error": "file_not_found"}

    # Extract the portion to debug
    debug_content = extract_debug_content(content, session['start_pos'], session['end_pos'])

    if verbose:
        print(f"=== Debugging {session['test_file']} ===")
        print(f"Content length: {len(content)} characters")
        print(f"Debug range: {session['start_pos']} to {session['end_pos'] or 'end'}")
        print(f"Debug content ({len(debug_content)} chars):")
        print("-" * 50)
        print(debug_content)
        print("-" * 50)

    results = {}
    grammar = REI1Grammar()

    # Test tokenization first
    if session['show_tokens']:
        results["tokenization"] = debug_tokenization(debug_content, verbose)

    # Test specific grammar component if specified
    if session['grammar_component']:
        results["component"] = debug_component(grammar, debug_content, session['grammar_component'], verbose)

    # Test full parsing
    results["full_parse"] = debug_full_parse(grammar, debug_content, session['test_file'], verbose)

    return results


# ============================================================================
# COMPATIBILITY CLASSES (for existing code)
# ============================================================================

class DebugSession:
    """Compatibility wrapper for debug session configuration"""
    def __init__(self, test_file: str, grammar_component: Optional[str] = None,
                 start_pos: int = 0, end_pos: Optional[int] = None,
                 show_tokens: bool = True, show_ast: bool = True,
                 show_context: bool = True, step_by_step: bool = False):
        self.test_file = test_file
        self.grammar_component = grammar_component
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.show_tokens = show_tokens
        self.show_ast = show_ast
        self.show_context = show_context
        self.step_by_step = step_by_step


class REI1Debugger:
    """Compatibility wrapper - delegates to functional implementation"""
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.grammar = REI1Grammar()

    def debug_file(self, session: DebugSession) -> Dict[str, Any]:
        """Debug a complete file or portion of it"""
        session_dict = make_debug_session(
            test_file=session.test_file,
            grammar_component=session.grammar_component,
            start_pos=session.start_pos,
            end_pos=session.end_pos,
            show_tokens=session.show_tokens,
            show_ast=session.show_ast,
            show_context=session.show_context,
            step_by_step=session.step_by_step
        )
        return debug_file_session(session_dict, self.verbose)

    def debug_component(self, text: str, component_name: str) -> Dict[str, Any]:
        """Debug a specific grammar component"""
        return debug_component(self.grammar, text, component_name, self.verbose)

    def _debug_tokenization(self, content: str) -> Dict[str, Any]:
        return debug_tokenization(content, self.verbose)

    def _debug_component(self, content: str, component_name: str) -> Dict[str, Any]:
        return debug_component(self.grammar, content, component_name, self.verbose)

    def _debug_full_parse(self, content: str, filename: str) -> Dict[str, Any]:
        return debug_full_parse(self.grammar, content, filename, self.verbose)

    def _find_parse_progress(self, content: str) -> Dict[str, Any]:
        return find_parse_progress(self.grammar, content, self.verbose)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the debugger"""
    import argparse

    parser = argparse.ArgumentParser(description="REI1 Language Debugger")
    parser.add_argument("input", help="File to debug or string content (use --string)")
    parser.add_argument("--component", "-c", help="Specific grammar component to test")
    parser.add_argument("--string", "-t", action="store_true", help="Treat input as string content instead of file")
    parser.add_argument("--start", "-s", type=int, default=0, help="Start position")
    parser.add_argument("--end", "-e", type=int, help="End position")
    parser.add_argument("--no-tokens", action="store_true", help="Skip tokenization debug")
    parser.add_argument("--no-ast", action="store_true", help="Skip AST debug")
    parser.add_argument("--step", action="store_true", help="Step-by-step debugging")

    args = parser.parse_args()

    grammar = REI1Grammar()

    if args.string:
        # Debug string content directly
        content = args.input
        print(f"=== Debugging string content ===")
        print(f"Content: '{content}'")
        print("-" * 50)

        if args.component:
            results = {"component": debug_component(grammar, content, args.component, verbose=True)}
        else:
            results = {"full_parse": debug_full_parse(grammar, content, "<string>", verbose=True)}
    else:
        # Debug file
        session = make_debug_session(
            test_file=args.input,
            grammar_component=args.component,
            start_pos=args.start,
            end_pos=args.end,
            show_tokens=not args.no_tokens,
            show_ast=not args.no_ast,
            step_by_step=args.step
        )

        results = debug_file_session(session, verbose=True)

    print(f"\n=== DEBUGGING COMPLETE ===")
    print(f"Results summary: {results}")


if __name__ == "__main__":
    main()

