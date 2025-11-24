"""
REI1 Programming Language - Main Entry Point
A pure functional language with ML modules, actors, and multiple dispatch
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import os

# Readline support for history and auto-completion
try:
  import readline
  READLINE_AVAILABLE = True
except ImportError:
  READLINE_AVAILABLE = False

from parsing import create_parser, create_debug_parser, REI1ParseError, pretty_print_cst
from semantics import create_analyzer, create_debug_analyzer, REI1SemanticsError
from interpreter import create_interpreter, create_debug_interpreter, REI1RuntimeError


def create_arg_parser() -> argparse.ArgumentParser:
  """Create command line argument parser"""
  parser = argparse.ArgumentParser(
      description='REI1 Programming Language - Pure functional with actors and modules',
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog="""
Examples:
  %(prog)s script.rei1            # Run a REI1 script
  %(prog)s -i                     # Interactive mode
  %(prog)s --parse script.rei1    # Parse and show CST
  %(prog)s --analyze script.rei1  # Parse, analyze and show AST
  %(prog)s --debug script.rei1    # Run with debug output
  %(prog)s -i --debug             # Interactive mode with debug
  timeout 5 %(prog)s script.rei1  # Run with 5 second timeout (Unix/macOS)
        """
  )

  parser.add_argument(
      'script',
      nargs='?',
      help='REI1 script file to execute'
  )

  parser.add_argument(
      '-i', '--interactive',
      action='store_true',
      help='Start interactive mode'
  )

  parser.add_argument(
      '--parse',
      action='store_true',
      help='Parse file and show CST (for debugging)'
  )

  parser.add_argument(
      '--analyze',
      action='store_true',
      help='Parse and analyze file, show AST (for debugging)'
  )

  parser.add_argument(
      '--debug',
      action='store_true',
      help='Enable debug output for all stages'
  )

  parser.add_argument(
      '--version',
      action='version',
      version='REI1 v0.2.0 (Full Interpreter)'
  )

  return parser


def parse_file(script_path: str, debug: bool = False) -> None:
  """Parse a REI1 script file and show the CST"""
  try:
    # Create parser
    parser = create_debug_parser() if debug else create_parser()

    # Parse the file
    print(f"Parsing {script_path}...")
    cst_nodes = parser.parse_file(script_path)

    print(f"\nParsed {len(cst_nodes)} top-level bindings:")
    print("=" * 50)

    for i, node in enumerate(cst_nodes, 1):
      print(f"\nBinding {i}:")
      print(pretty_print_cst(node))

  except FileNotFoundError:
    print(f"Error: Script file '{script_path}' not found")
    print(f"  Hint: Check the file path and make sure the file exists")
    sys.exit(1)
  except PermissionError:
    print(f"Error: Permission denied reading '{script_path}'")
    print(f"  Hint: Make sure you have read permissions for this file")
    sys.exit(1)
  except UnicodeDecodeError as e:
    print(f"Error: Cannot decode file '{script_path}': {e}")
    print(f"  Hint: Make sure the file is a text file with UTF-8 encoding")
    sys.exit(1)
  except REI1ParseError as e:
    print(f"Parse error in '{script_path}': {e}")
    sys.exit(1)
  except Exception as e:
    print(f"Unexpected error while processing '{script_path}': {e}")
    if debug:
      import traceback
      traceback.print_exc()
    sys.exit(1)


def analyze_file(script_path: str, debug: bool = False) -> None:
  """Parse and analyze a REI1 script file and show the AST"""
  try:
    # Create parser and analyzer
    parser = create_debug_parser() if debug else create_parser()
    analyzer = create_debug_analyzer() if debug else create_analyzer()

    # Parse the file
    print(f"Parsing and analyzing {script_path}...")
    cst_nodes = parser.parse_file(script_path)

    print(f"\nParsed {len(cst_nodes)} top-level bindings:")
    print("=" * 50)

    for i, cst_node in enumerate(cst_nodes, 1):
      print(f"\nBinding {i} - CST:")
      print(pretty_print_cst(cst_node))

      try:
        ast_node = analyzer.analyze_module(cst_node)
        print(f"\nBinding {i} - AST:")
        print(f"Type: {ast_node.type}")
        print(f"Type Info: {ast_node.type_info}")
        if hasattr(ast_node, 'value') and ast_node.value:
          print(f"Value: {ast_node.value}")
      except REI1SemanticsError as e:
        print(f"Semantic analysis error: {e}")

  except FileNotFoundError:
    print(f"Error: Script file '{script_path}' not found")
    sys.exit(1)
  except REI1ParseError as e:
    print(f"Parse error in '{script_path}': {e}")
    sys.exit(1)
  except Exception as e:
    print(f"Unexpected error while processing '{script_path}': {e}")
    if debug:
      import traceback
      traceback.print_exc()
    sys.exit(1)


def run_script_file(script_path: str, debug: bool = False) -> None:
  """Run a REI1 script file with full interpretation"""
  try:
    # Create parser, analyzer, and interpreter
    parser = create_debug_parser() if debug else create_parser()
    analyzer = create_debug_analyzer() if debug else create_analyzer()
    interpreter = create_debug_interpreter() if debug else create_interpreter()

    # Parse the file
    if debug:
      print(f"Parsing {script_path}...")
    cst_nodes = parser.parse_file(script_path)
    if debug:
      print(f"Parsed {len(cst_nodes)} statements")

    # Analyze each statement
    ast_nodes = []
    for cst_node in cst_nodes:
      ast_node = analyzer.analyze_module(cst_node)
      ast_nodes.append(ast_node)

    if debug:
      print(f"Analyzed {len(ast_nodes)} statements")

    # Interpret the program
    final_env = interpreter.interpret_program(ast_nodes)

    # Show results
    print(f"Program executed successfully!")
    if debug or len(final_env) > 0:
      print(f"\nFinal environment ({len(final_env)} bindings):")
      for name, value in final_env.items():
        if not name.startswith('_') and name not in ['+', '-', '*', '/', '==', '!=', '<', '<=', '>', '>=', 'True', 'False']:
          print(f"  {name} = {value}")

  except FileNotFoundError:
    print(f"Error: Script file '{script_path}' not found")
    print(f"  Hint: Check the file path and make sure the file exists")
    sys.exit(1)
  except PermissionError:
    print(f"Error: Permission denied reading '{script_path}'")
    print(f"  Hint: Make sure you have read permissions for this file")
    sys.exit(1)
  except UnicodeDecodeError as e:
    print(f"Error: Cannot decode file '{script_path}': {e}")
    print(f"  Hint: Make sure the file is a text file with UTF-8 encoding")
    sys.exit(1)
  except REI1ParseError as e:
    print(f"Parse error in '{script_path}': {e}")
    sys.exit(1)
  except REI1SemanticsError as e:
    print(f"Semantic analysis error in '{script_path}': {e}")
    sys.exit(1)
  except REI1RuntimeError as e:
    print(f"\n{'='*70}")
    print(f"Runtime Error in '{script_path}'")
    print(f"{'='*70}")
    print(f"\nError: {e.message}")

    # Show source location if available
    if hasattr(e, 'span') and e.span:
      print(f"\nLocation: {e.span}")

    # Show source line if available
    if hasattr(e, 'source_line') and e.source_line:
      print(f"\nSource:")
      print(f"  {e.source_line}")
      print(f"  {'~' * len(e.source_line)}")

    # Show environment snapshot if available
    if hasattr(e, 'env_snapshot') and e.env_snapshot and debug:
      print(f"\nEnvironment at error:")
      bindings = e.env_snapshot.get('bindings', {})
      if bindings:
        user_bindings = {k: v for k, v in bindings.items()
                         if not k.startswith('_') and k not in ['+', '-', '*', '/', '==', '!=', '<', '<=', '>', '>=']}
        if user_bindings:
          for name, value in list(user_bindings.items())[:10]:  # Show first 10
            val_str = str(value).replace('\n', ' ')[:60]
            print(f"  {name} = {val_str}")
          if len(user_bindings) > 10:
            print(f"  ... and {len(user_bindings) - 10} more bindings")

    print(f"\n{'='*70}\n")
    sys.exit(1)
  except Exception as e:
    print(f"Unexpected error while executing '{script_path}': {e}")
    if debug:
      import traceback
      traceback.print_exc()
    sys.exit(1)


def setup_readline():
  """Setup readline with history and auto-completion"""
  if not READLINE_AVAILABLE:
    return

  # Setup history file
  history_file = os.path.expanduser("~/.rei1_history")
  try:
    readline.read_history_file(history_file)
  except (FileNotFoundError, PermissionError, OSError):
    pass  # First time, no history yet, or permission denied

  # Set history length
  try:
    readline.set_history_length(1000)
  except:
    pass

  # Setup auto-completion
  # Common REI1 keywords and built-ins
  completions = [
      # Keywords
      "case", "of", "where", "module", "sig", "type", "contract",
      "pre", "post", "unsafe", "Type",
      # Built-in functions
      "map", "filter", "fold", "head", "tail", "length", "concat",
      "cons", "print", "println", "reverse", "take", "drop",
      "par-map", "par-filter", "par-fold",
      # IO operations
      "IO.read", "IO.write",
      # REPL commands
      ":parse", ":analyze", ":env", ":help", "exit."
  ]

  def completer(text, state):
    options = [cmd for cmd in completions if cmd.startswith(text)]
    if state < len(options):
      return options[state]
    return None

  try:
    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")
  except:
    pass

  # Save history on exit
  import atexit
  try:
    atexit.register(lambda: readline.write_history_file(history_file))
  except:
    pass


def run_interactive_mode(debug: bool = False) -> None:
  """Run REI1 in interactive mode with full interpretation"""
  print("REI1 v0.2.0 (Full Interpreter) - Interactive Mode")
  print("Type 'exit.' to quit, ':help' for commands")
  if READLINE_AVAILABLE:
    print("Readline enabled: Use ↑/↓ for history, Tab for completion")
  if debug:
    print("Debug mode enabled")
  print()

  # Setup readline
  setup_readline()

  # Create parser, analyzer, and interpreter
  parser = create_debug_parser() if debug else create_parser()
  analyzer = create_debug_analyzer() if debug else create_analyzer()
  interpreter = create_debug_interpreter() if debug else create_interpreter()

  # Shared environment for interactive session - this must be threaded through
  from interpreter import create_builtin_runtime_env
  from semantics import create_builtin_env, env_bind, make_type_info

  session_env = create_builtin_runtime_env()  # Runtime environment
  semantic_env = create_builtin_env()  # Semantic/type environment

  while True:
    try:
      code = input("rei1> ")

      if code.strip() == "exit.":
        break

      if not code.strip():
        continue

      # Special commands
      if code.startswith(":parse "):
        expr_text = code[7:]  # Remove ":parse "
        try:
          cst = parser.parse_expression(expr_text)
          print("Expression CST:")
          print(pretty_print_cst(cst))
        except REI1ParseError as e:
          print(f"Parse error: {e}")
        continue

      if code.startswith(":analyze "):
        expr_text = code[9:]  # Remove ":analyze "
        try:
          cst = parser.parse_expression(expr_text)
          ast = analyzer._analyze_node(cst, analyzer.global_env)
          print("Expression AST:")
          print(f"Type: {ast.type}")
          print(f"Type Info: {ast.type_info}")
          if hasattr(ast, 'value') and ast.value:
            print(f"Value: {ast.value}")
        except (REI1ParseError, REI1SemanticsError) as e:
          print(f"Error: {e}")
        continue

      if code.strip() == ":env":
        print("Current environment:")
        # Get bindings from runtime environment
        user_bindings = {k: v for k, v in session_env['bindings'].items()
                         if not k.startswith('_') and k not in ['+', '-', '*', '/', '==', '!=', '<', '<=', '>', '>=', 'True', 'False']}
        if user_bindings:
          for name, value in user_bindings.items():
            val_str = str(value)
            if len(val_str) > 60:
              val_str = val_str[:57] + "..."
            print(f"  {name} = {val_str}")
        else:
          print("  (no user-defined bindings)")
        continue

      if code.strip() == ":help":
        print("REPL Commands:")
        print("  :parse <expr>     - Show parsed CST")
        print("  :analyze <expr>   - Show analyzed AST")
        print("  :env              - Show current environment")
        print("  :help             - Show this help")
        print("  exit.             - Exit REPL")
        print()
        print("Language features:")
        print("  x = 5.                    - Value binding")
        print("  double x = * x 2.         - Function definition")
        print("  double 4                  - Function call (no dot needed)")
        print("  10 |> double              - Pipeline")
        print("  User = Type $ User String - Create type")
        print("  unsafe $ IO.read \"file\"  - Unsafe IO")
        continue

      # Try to execute the code
      try:
        if code.endswith('.'):
          # Parse as complete statement/binding
          from semantics import analyze_cst_node
          from interpreter import eval_ast

          cst_nodes = parser.parse_string(code)
          for cst_node in cst_nodes:
            # Analyze in current semantic environment
            ast_node = analyze_cst_node(cst_node, semantic_env, debug)

            # Evaluate in runtime environment
            result_val, session_env = eval_ast(ast_node, session_env, debug)

            # Update semantic environment with new bindings
            if ast_node['type'] == "FUNCTION_DEF" and isinstance(ast_node.get('value'), dict):
              func_name = ast_node['value'].get('name', 'unknown')
              # Add function to semantic env with Function type
              semantic_env = env_bind(
                  semantic_env, func_name, make_type_info("Function"))
              print(f"Defined function: {func_name}")
            elif ast_node['type'] == "VALUE_BINDING" and isinstance(ast_node.get('value'), dict):
              var_name = ast_node['value'].get('name', 'unknown')
              # Add value to semantic env with its type
              var_type = ast_node.get('type_info', make_type_info("Unknown"))
              semantic_env = env_bind(semantic_env, var_name, var_type)
              # Show the bound value
              val_str = str(result_val.get('value', result_val))
              if len(val_str) > 60:
                val_str = val_str[:57] + "..."
              print(f"Bound: {var_name} = {val_str}")
            else:
              # For other statements, show result if any
              val_str = str(result_val.get('value', result_val))
              if len(val_str) > 60:
                val_str = val_str[:57] + "..."
              print(f"=> {val_str}")
        else:
          # Parse and evaluate as expression (no trailing dot)
          from semantics import analyze_cst_node
          from interpreter import eval_ast

          cst = parser.parse_expression(code)

          # Analyze in current semantic environment
          ast = analyze_cst_node(cst, semantic_env, debug)

          # Evaluate with session environment
          result_val, updated_env = eval_ast(ast, session_env, debug)

          # Don't update session env for expressions (they shouldn't have side effects)
          # But do show the result
          val_str = str(result_val.get('value', result_val))
          type_str = result_val.get('type', 'Unknown')
          print(f"=> {val_str} : {type_str}")
      except REI1ParseError as e:
        print(f"Parse error: {e}")
      except REI1SemanticsError as e:
        print(f"Semantic error: {e}")
      except REI1RuntimeError as e:
        print(f"\nRuntime Error:")
        print(f"  {e.message}")
        if hasattr(e, 'span') and e.span:
          print(f"  Location: {e.span}")
        if hasattr(e, 'source_line') and e.source_line:
          print(f"  Source: {e.source_line}")
        print()

    except KeyboardInterrupt:
      print("\nGoodbye!")
      break
    except EOFError:
      print("\nGoodbye!")
      break
    except Exception as e:
      print(f"Unexpected error: {e}")
      if debug:
        import traceback
        traceback.print_exc()
      print("  Hint: If this keeps happening, try restarting or use --debug for more details")


def show_language_info() -> None:
  """Show REI1 language information"""
  print("REI1 Programming Language")
  print("=" * 50)
  print("A pure functional language with:")
  print("• ML-style modules and signatures")
  print("• Actor-based concurrency")
  print("• Multiple dispatch")
  print("• Algebraic data types")
  print("• Design by contract")
  print()
  print("Features: Parsing, semantic analysis, and interpretation")
  print()


def main() -> None:
  """Main entry point for REI1"""
  arg_parser = create_arg_parser()
  args = arg_parser.parse_args()

  # Handle special cases
  if len(sys.argv) == 1:
    # No arguments - show info and start interactive mode
    show_language_info()
    print("Starting interactive mode...")
    print("Use 'rei1 --help' for command line options")
    print()
    run_interactive_mode(debug=False)
    return

  # Handle script operations
  if args.script:
    if not Path(args.script).exists():
      print(f"Error: Script file '{args.script}' does not exist")
      sys.exit(1)

    if args.parse:
      parse_file(args.script, debug=args.debug)
    elif args.analyze:
      analyze_file(args.script, debug=args.debug)
    else:
      run_script_file(args.script, debug=args.debug)

  # Handle interactive mode
  elif args.interactive:
    run_interactive_mode(debug=args.debug)

  else:
    # Show help and language info
    arg_parser.print_help()
    print()
    show_language_info()


if __name__ == "__main__":
  main()
