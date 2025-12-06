"""
REI1 Standard Library
Built-in functions and modules for REI1
Pure functional style using immutable dictionaries
"""

from typing import Dict, Callable, Any, List
import operator
from utilities import (
  binary_comparison_op,
  binary_arithmetic_op,
  validate_function_args,
  type_mismatch_error,
  is_value_dict,
  REI1RuntimeError
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def make_value(value: Any, type_name: str = "Unknown") -> Dict:
  """Create an immutable runtime value"""
  return {
      'value': value,
      'type': type_name
  }


# ============================================================================
# PRINT FUNCTIONS
# ============================================================================

def rei1_print(value: Dict) -> Dict:
  """Print a value to stdout"""
  if value['type'] == "String":
    print(value['value'], end='')
  elif value['type'] == "Num":
    print(value['value'], end='')
  elif value['type'] == "Bool":
    print("True" if value['value'] else "False", end='')
  elif value['type'] == "Char":
    print(value['value'], end='')
  elif value['type'] == "List":
    # Print list elements
    print("[", end='')
    for i, elem in enumerate(value['value']):
      if i > 0:
        print(", ", end='')
      rei1_print(elem)
    print("]", end='')
  else:
    print(f"<{value['type']}>", end='')

  return make_value(None, "Unit")


def rei1_println(value: Dict) -> Dict:
  """Print a value with newline"""
  rei1_print(value)
  print()  # Add newline
  return make_value(None, "Unit")


def rei1_show(value: Dict) -> Dict:
  """Convert value to string representation"""
  if value['type'] == "Num":
    return make_value(str(value['value']), "String")
  elif value['type'] == "String":
    return make_value(f'"{value["value"]}"', "String")
  elif value['type'] == "Char":
    return make_value(f"`{value['value']}", "String")
  elif value['type'] == "Bool":
    return make_value("True" if value['value'] else "False", "String")
  elif value['type'] == "List":
    elements = []
    for elem in value['value']:
      elem_str = rei1_show(elem)
      elements.append(elem_str['value'])
    return make_value(f"[{', '.join(elements)}]", "String")
  else:
    return make_value(f"<{value['type']}>", "String")


# ============================================================================
# LIST FUNCTIONS
# ============================================================================

def rei1_length(lst: Dict) -> Dict:
  """Get length of a list or string"""
  if lst['type'] == "List":
    return make_value(len(lst['value']), "Num")
  elif lst['type'] == "String":
    return make_value(len(lst['value']), "Num")
  else:
    raise REI1RuntimeError(f"Cannot get length of {lst['type']}")


def rei1_head(lst: Dict) -> Dict:
  """Get first element of a list"""
  if lst['type'] == "List":
    if not lst['value']:
      raise REI1RuntimeError("Cannot get head of empty list")
    return lst['value'][0]
  else:
    raise REI1RuntimeError(f"Cannot get head of {lst['type']}")


def rei1_tail(lst: Dict) -> Dict:
  """Get tail (all but first element) of a list"""
  if lst['type'] == "List":
    if not lst['value']:
      raise REI1RuntimeError("Cannot get tail of empty list")
    return make_value(lst['value'][1:], "List")
  else:
    raise REI1RuntimeError(f"Cannot get tail of {lst['type']}")


def rei1_cons(elem: Dict, lst: Dict) -> Dict:
  """Prepend element to list (:: operator)"""
  if lst['type'] == "List":
    return make_value([elem] + lst['value'], "List")
  else:
    raise REI1RuntimeError(f"Cannot cons to {lst['type']}")


def rei1_append(lst1: Dict, lst2: Dict) -> Dict:
  """Append two lists or concatenate two strings"""
  if lst1['type'] == "List" and lst2['type'] == "List":
    return make_value(lst1['value'] + lst2['value'], "List")
  elif lst1['type'] == "String" and lst2['type'] == "String":
    return make_value(lst1['value'] + lst2['value'], "String")
  else:
    raise REI1RuntimeError(f"Cannot append {lst1['type']} and {lst2['type']}")


def rei1_reverse(lst: Dict) -> Dict:
  """Reverse a list"""
  if lst['type'] != 'List':
    raise REI1RuntimeError(f"reverse requires list, got {lst['type']}")
  return make_value(list(reversed(lst['value'])), "List")


def rei1_take(n: Dict, lst: Dict) -> Dict:
  """Take first n elements from list"""
  validate_function_args("take", [n, lst], ["Num", "List"])
  return make_value(lst['value'][:int(n['value'])], "List")


def rei1_drop(n: Dict, lst: Dict) -> Dict:
  """Drop first n elements from list"""
  validate_function_args("drop", [n, lst], ["Num", "List"])
  return make_value(lst['value'][int(n['value']):], "List")


def rei1_elem(val: Dict, lst: Dict) -> Dict:
  """Check if element is in list"""
  if lst['type'] != 'List':
    raise REI1RuntimeError(f"elem requires list, got {lst['type']}")
  # Compare by value
  for item in lst['value']:
    if isinstance(item, dict) and 'value' in item:
      if item['value'] == val['value']:
        return make_value(True, "Bool")
    elif item == val['value']:
      return make_value(True, "Bool")
  return make_value(False, "Bool")


def rei1_range(start: Dict, end: Dict) -> Dict:
  """Create a range of numbers"""
  if start['type'] != 'Num' or end['type'] != 'Num':
    raise REI1RuntimeError(f"range requires numbers")
  result = [make_value(i, "Num")
            for i in range(int(start['value']), int(end['value']))]
  return make_value(result, "List")


def rei1_zip(lst1: Dict, lst2: Dict) -> Dict:
  """Zip two lists into list of pairs"""
  if lst1['type'] != 'List' or lst2['type'] != 'List':
    raise REI1RuntimeError(f"zip requires two lists")
  pairs = []
  for a, b in zip(lst1['value'], lst2['value']):
    pairs.append(make_value([a, b], "List"))
  return make_value(pairs, "List")


# Note: map, filter, fold are handled specially in interpreter.py
# because they need access to eval_ast for function application


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def rei1_eq(x: Dict, y: Dict) -> Dict:
  """Equality comparison"""
  if x['type'] != y['type']:
    return make_value(False, "Bool")
  return make_value(x['value'] == y['value'], "Bool")


def rei1_ne(x: Dict, y: Dict) -> Dict:
  """Not equal comparison"""
  result = rei1_eq(x, y)
  return make_value(not result['value'], "Bool")


# Use factory functions for comparison operations
_rei1_lt_impl = binary_comparison_op(operator.lt, "less than")
_rei1_gt_impl = binary_comparison_op(operator.gt, "greater than")
_rei1_le_impl = binary_comparison_op(operator.le, "less than or equal")
_rei1_ge_impl = binary_comparison_op(operator.ge, "greater than or equal")


def rei1_lt(x: Dict, y: Dict) -> Dict:
  """Less than comparison"""
  return _rei1_lt_impl(x, y, make_value)


def rei1_gt(x: Dict, y: Dict) -> Dict:
  """Greater than comparison"""
  return _rei1_gt_impl(x, y, make_value)


def rei1_le(x: Dict, y: Dict) -> Dict:
  """Less than or equal comparison"""
  return _rei1_le_impl(x, y, make_value)


def rei1_ge(x: Dict, y: Dict) -> Dict:
  """Greater than or equal comparison"""
  return _rei1_ge_impl(x, y, make_value)


# ============================================================================
# ARITHMETIC FUNCTIONS
# ============================================================================

def rei1_add(x: Dict, y: Dict) -> Dict:
  """Addition for numbers and strings"""
  if x['type'] == "Num" and y['type'] == "Num":
    return make_value(x['value'] + y['value'], "Num")
  elif x['type'] == "String" and y['type'] == "String":
    return make_value(x['value'] + y['value'], "String")
  elif x['type'] == "List" and y['type'] == "List":
    return make_value(x['value'] + y['value'], "List")
  else:
    raise REI1RuntimeError(f"Cannot add {x['type']} and {y['type']}")


# Use factory functions for arithmetic operations
_rei1_sub_impl = binary_arithmetic_op(operator.sub, "subtract")
_rei1_mul_impl = binary_arithmetic_op(operator.mul, "multiply")


def rei1_sub(x: Dict, y: Dict) -> Dict:
  """Subtraction"""
  return _rei1_sub_impl(x, y, make_value)


def rei1_mul(x: Dict, y: Dict) -> Dict:
  """Multiplication"""
  return _rei1_mul_impl(x, y, make_value)


def rei1_div(x: Dict, y: Dict) -> Dict:
  """Division"""
  if x['type'] != "Num" or y['type'] != "Num":
    raise REI1RuntimeError(f"Cannot divide {x['type']} and {y['type']}")
  if y['value'] == 0:
    raise REI1RuntimeError("Division by zero")
  return make_value(x['value'] / y['value'], "Num")


def rei1_mod(x: Dict, y: Dict) -> Dict:
  """Modulo"""
  if x['type'] != "Num" or y['type'] != "Num":
    raise REI1RuntimeError(f"Cannot compute modulo of {x['type']} and {y['type']}")
  if y['value'] == 0:
    raise REI1RuntimeError("Modulo by zero")
  return make_value(x['value'] % y['value'], "Num")


# ============================================================================
# SPECIAL FUNCTIONS
# ============================================================================

def rei1_Type(constructor_expr: Dict) -> Dict:
  """Type constructor function - creates ADT constructors at runtime

  Usage: User = Type $ User String Num String

  This is a runtime function that takes a constructor expression and returns
  a constructor function. The $ applies Type to the constructor expression.

  The constructor_expr is the AST of something like "User String Num String"
  which gets parsed as a function call: User(String, Num, String)

  We need to:
  1. Extract the constructor name (e.g., "User")
  2. Count the arity (e.g., 3 for String, Num, String)
  3. Return a constructor function that creates tagged instances
  """
  # Handle different possible structures
  # After desugaring and evaluation, this might be a value dict

  if isinstance(constructor_expr, dict):
    # Check if it's a FUNCTION_CALL AST node (not yet evaluated)
    if constructor_expr.get('type') == 'FUNCTION_CALL':
      # Extract function name and args from AST
      func_node = constructor_expr.get('value', {}).get('function', {})
      args = constructor_expr.get('value', {}).get('args', [])

      # Get constructor name
      if func_node.get('type') == 'FUNCTION_NAME':
        ctor_name = func_node.get('value', 'Unknown')
      elif func_node.get('type') == 'IDENTIFIER':
        ctor_name = func_node.get('value', 'Unknown')
      else:
        ctor_name = 'Unknown'

      arity = len(args)

    # Or it might be an already-evaluated value
    elif constructor_expr.get('type') == 'constructor_instance':
      # Already a constructor instance, extract info
      ctor_name = constructor_expr.get('constructor', 'Unknown')
      arity = constructor_expr.get('arity', 0)
    else:
      # Fallback: assume it's meant to be a nullary constructor
      ctor_name = str(constructor_expr.get('value', 'Unknown'))
      arity = 0
  else:
    # Unexpected format
    raise REI1RuntimeError(
        f"Type: unexpected constructor format: {type(constructor_expr)}")

  # Create the constructor function
  # This is a function that takes N arguments and returns a tagged value
  def constructor_function(*args):
    """Dynamically created constructor function"""
    arg_values = []
    for arg in args:
      # Extract value from wrapped dicts
      if isinstance(arg, dict) and 'value' in arg:
        arg_values.append(arg['value'])
      else:
        arg_values.append(arg)

    if len(arg_values) != arity:
      raise REI1RuntimeError(
          f"Constructor {ctor_name} expects {arity} arguments, got {len(arg_values)}"
      )

    # Return a tagged constructor instance
    return make_value({
        'constructor': ctor_name,
        'fields': arg_values,
        'arity': arity
    }, ctor_name)  # Type name is the constructor name

  # Return the constructor as a callable function value
  # We use the 'constructor' type to mark it as a constructor
  return {
      'type': 'constructor',
      'name': ctor_name,
      'arity': arity,
      'func': constructor_function
  }


def rei1_User(name: str, age: int, email: str) -> Dict:
  """User constructor - creates a User instance
  Usage: User "Alice" 30 "alice@example.com"
  """
  return {
      'type': 'constructor_instance',
      'constructor': 'User',
      'fields': [name, age, email],
      'arity': 3
  }


def rei1_Post(title: str, content: str, author: Dict) -> Dict:
  """Post constructor - creates a Post instance
  Usage: Post "Title" "Content" user_instance
  """
  return {
      'type': 'constructor_instance',
      'constructor': 'Post',
      'fields': [title, content, author],
      'arity': 3
  }


def rei1_unsafe(expr: Dict) -> Dict:
  """Unsafe function - marks expressions as unsafe for IO operations
  Usage: unsafe $ IO.read "file.txt"
  This is a marker function that passes through its argument but signals
  that unsafe operations (IO, side effects) are permitted in this context
  """
  # Just pass through the value - the marker is mainly for type checking
  return expr


def rei1_Sig(members: Dict) -> Dict:
  """Sig function - creates a callable module signature at runtime
  Usage: MySig = Sig { add: \"Num -> Num -> Num\", sub: \"Num -> Num -> Num\" }
  Then: MyMod = MySig { add: λ x y => + x y, sub: λ x y => - x y }

  Takes a record/dictionary of function names to type signatures.
  Returns a callable signature value that when called with an implementation,
  creates a module.
  """
  # Create a callable signature value
  # The signature value itself is callable and creates modules when invoked
  return make_value({
      'members': members,
      'sig_type': 'Signature',
      'callable': True  # Mark as callable to create modules
  }, "Signature")


# ============================================================================
# BUILT-IN FUNCTION REGISTRY
# ============================================================================

def make_builtin_function(name: str, func: Callable, type_signature: str = "") -> Dict:
  """Create a built-in function value"""
  return {
      'type': 'builtin_function',
      'name': name,
      'func': func,
      'type_signature': type_signature
  }


# Built-in function registry
# Note: map, filter, fold are handled in interpreter.py since they need eval_ast
BUILTIN_FUNCTIONS: Dict[str, Dict] = {
    # I/O functions
    "print": make_builtin_function("print", rei1_print, "a -> Unit"),
    "println": make_builtin_function("println", rei1_println, "a -> Unit"),
    "show": make_builtin_function("show", rei1_show, "a -> String"),

    # List functions
    "length": make_builtin_function("length", rei1_length, "List a -> Num"),
    "head": make_builtin_function("head", rei1_head, "List a -> a"),
    "tail": make_builtin_function("tail", rei1_tail, "List a -> List a"),
    "::": make_builtin_function("::", rei1_cons, "a -> List a -> List a"),
    "append": make_builtin_function("append", rei1_append, "List a -> List a -> List a"),
    "reverse": make_builtin_function("reverse", rei1_reverse, "List a -> List a"),
    "take": make_builtin_function("take", rei1_take, "Num -> List a -> List a"),
    "drop": make_builtin_function("drop", rei1_drop, "Num -> List a -> List a"),
    "elem": make_builtin_function("elem", rei1_elem, "a -> List a -> Bool"),
    "range": make_builtin_function("range", rei1_range, "Num -> Num -> List Num"),
    "zip": make_builtin_function("zip", rei1_zip, "List a -> List b -> List (a, b)"),

    # Comparison functions
    "==": make_builtin_function("==", rei1_eq, "a -> a -> Bool"),
    "!=": make_builtin_function("!=", rei1_ne, "a -> a -> Bool"),
    "<": make_builtin_function("<", rei1_lt, "a -> a -> Bool"),
    ">": make_builtin_function(">", rei1_gt, "a -> a -> Bool"),
    "<=": make_builtin_function("<=", rei1_le, "a -> a -> Bool"),
    ">=": make_builtin_function(">=", rei1_ge, "a -> a -> Bool"),

    # Arithmetic functions
    "+": make_builtin_function("+", rei1_add, "Num -> Num -> Num"),
    "-": make_builtin_function("-", rei1_sub, "Num -> Num -> Num"),
    "*": make_builtin_function("*", rei1_mul, "Num -> Num -> Num"),
    "/": make_builtin_function("/", rei1_div, "Num -> Num -> Num"),
    "%": make_builtin_function("%", rei1_mod, "Num -> Num -> Num"),

    # Special functions
    "Type": make_builtin_function("Type", rei1_Type, "Constructor -> Type"),
    "Sig": make_builtin_function("Sig", rei1_Sig, "Record -> Signature"),
    "unsafe": make_builtin_function("unsafe", rei1_unsafe, "a -> a"),

    # Data constructors
    "User": make_builtin_function("User", rei1_User, "String -> Num -> String -> User"),
    "Post": make_builtin_function("Post", rei1_Post, "String -> String -> User -> Post"),
}


def get_builtin_function(name: str) -> Dict:
  """Get a built-in function by name"""
  if name in BUILTIN_FUNCTIONS:
    return BUILTIN_FUNCTIONS[name]
  else:
    raise REI1RuntimeError(f"Unknown built-in function: {name}")


def list_builtin_functions() -> List[str]:
  """List all available built-in functions"""
  return list(BUILTIN_FUNCTIONS.keys())


if __name__ == "__main__":
  # Test built-in functions
  print("REI1 Standard Library")
  print("=" * 30)
  print(f"Available functions: {len(BUILTIN_FUNCTIONS)}")
  for name, func in BUILTIN_FUNCTIONS.items():
    print(f"  {name}: {func['type_signature']}")

  # Test some functions
  print("\nTesting functions:")

  # Create test values
  num1 = make_value(42, "Num")
  num2 = make_value(3, "Num")
  str1 = make_value("Hello", "String")
  list1 = make_value([num1, num2], "List")

  # Test show function
  show_result = rei1_show(num1)
  print(f"show(42) = {show_result['value']}")

  # Test length function
  len_result = rei1_length(list1)
  print(f"length([42, 3]) = {len_result['value']}")

  # Test cons function
  cons_result = rei1_cons(str1, make_value([], "List"))
  print(f"cons('Hello', []) = {rei1_show(cons_result)['value']}")
