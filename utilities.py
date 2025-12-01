"""
Utilities module for REI1 compiler/interpreter
Contains common helper functions to reduce code duplication
"""

from typing import Any, Dict, List, Optional, Callable, Tuple, TypeVar
import operator


# Custom exception classes (imported from stdlib)
from stdlib import REI1RuntimeError


T = TypeVar('T')


# ==================== VALUE EXTRACTION UTILITIES ====================

def extract_from_wrapper(
  data: Any,
  wrapper_type: Optional[str] = None,
  default: Any = None
) -> Any:
  """
  Generic extraction from tuple/dict wrappers

  Args:
    data: Input data (can be str, tuple, dict, or raw value)
    wrapper_type: Expected wrapper type (e.g., "IDENTIFIER", "PATTERN_VAR")
    default: Default value if extraction fails

  Returns:
    Extracted value or default

  Examples:
    extract_from_wrapper("foo") -> "foo"
    extract_from_wrapper(("IDENTIFIER", "foo")) -> "foo"
    extract_from_wrapper(("IDENTIFIER", "foo"), "IDENTIFIER") -> "foo"
    extract_from_wrapper({"value": 42}) -> 42
  """
  if isinstance(data, str):
    return data
  elif isinstance(data, tuple) and len(data) >= 2:
    if wrapper_type is None or data[0] == wrapper_type:
      return data[1]
  elif isinstance(data, dict) and 'value' in data:
    return data['value']
  return default


def unpack_typed_tuple(
  item: Any,
  expected_type: Optional[str] = None,
  min_length: int = 2
) -> Optional[Tuple[str, Any, List[Any]]]:
  """
  Safely unpack typed tuple (TYPE, value, *rest)

  Args:
    item: Input item
    expected_type: Expected type tag (or None for any)
    min_length: Minimum tuple length

  Returns:
    (type_tag, value, rest) or None if invalid

  Examples:
    unpack_typed_tuple(("TYPED_PARAM", data)) -> ("TYPED_PARAM", data, [])
    unpack_typed_tuple(("TYPED_PARAM", data, extra)) -> ("TYPED_PARAM", data, [extra])
  """
  if not isinstance(item, tuple) or len(item) < min_length:
    return None

  type_tag = item[0]
  if expected_type and type_tag != expected_type:
    return None

  value = item[1] if len(item) > 1 else None
  rest = list(item[2:]) if len(item) > 2 else []

  return (type_tag, value, rest)


# ==================== TYPE CHECKING UTILITIES ====================

def is_value_dict(val: Any) -> bool:
  """
  Check if value is a wrapped value dict

  Args:
    val: Value to check

  Returns:
    True if val is a dict with 'type' and 'value' keys
  """
  return isinstance(val, dict) and 'type' in val and 'value' in val


def is_tuple_with_type(item: Any, expected_type: Optional[str] = None) -> bool:
  """
  Check if item is tuple with type in first position

  Args:
    item: Item to check
    expected_type: Expected type tag (optional)

  Returns:
    True if item is a valid typed tuple
  """
  if not (isinstance(item, tuple) and len(item) >= 2):
    return False
  return item[0] == expected_type if expected_type else True


def get_dict_type(val: Dict) -> Optional[str]:
  """
  Safely get type from dict

  Args:
    val: Value dict

  Returns:
    Type string or None
  """
  return val.get('type') if isinstance(val, dict) else None


def safe_dict_get(
  data: Dict,
  *keys: str,
  default: Any = None,
  required: bool = False
) -> Any:
  """
  Safely navigate nested dict structure

  Args:
    data: Source dictionary
    *keys: Path of keys to navigate
    default: Default value if path not found
    required: Raise error if path not found

  Returns:
    Value at key path or default

  Examples:
    safe_dict_get(node, 'value', 'name') -> node['value']['name']
    safe_dict_get(node, 'value', 'name', required=True) -> raises if not found
  """
  current = data
  for key in keys:
    if isinstance(current, dict):
      current = current.get(key)
      if current is None:
        if required:
          raise ValueError(f"Required key path {keys} not found")
        return default
    else:
      if required:
        raise ValueError(f"Expected dict at {key}")
      return default
  return current if current is not None else default


# ==================== ERROR MESSAGE BUILDERS ====================

def type_mismatch_error(
  func_name: str,
  param_name: str,
  expected: str,
  actual: Dict
) -> REI1RuntimeError:
  """
  Generate type mismatch error

  Args:
    func_name: Function name
    param_name: Parameter name
    expected: Expected type
    actual: Actual value dict

  Returns:
    REI1RuntimeError with formatted message
  """
  actual_type = actual.get('type', 'Unknown')
  return REI1RuntimeError(
    f"{func_name} requires {expected} for {param_name}, got {actual_type}"
  )


def arity_error(func_name: str, expected: int, got: int) -> REI1RuntimeError:
  """
  Generate arity mismatch error

  Args:
    func_name: Function name
    expected: Expected number of arguments
    got: Actual number of arguments

  Returns:
    REI1RuntimeError with formatted message
  """
  return REI1RuntimeError(
    f"{func_name} requires {expected} arguments, got {got}"
  )


def operation_error(
  op: str,
  left_type: str,
  right_type: str
) -> REI1RuntimeError:
  """
  Generate operation error

  Args:
    op: Operation name
    left_type: Left operand type
    right_type: Right operand type

  Returns:
    REI1RuntimeError with formatted message
  """
  return REI1RuntimeError(
    f"Cannot {op} {left_type} and {right_type}"
  )


# ==================== VALIDATION UTILITIES ====================

def validate_function_args(
  func_name: str,
  args: List[Dict],
  expected_types: List[str]
) -> None:
  """
  Validate function arguments match expected types

  Args:
    func_name: Function name for error messages
    args: List of argument values
    expected_types: List of expected type names

  Raises:
    REI1RuntimeError if validation fails
  """
  if len(args) != len(expected_types):
    raise arity_error(func_name, len(expected_types), len(args))

  for i, (arg, expected) in enumerate(zip(args, expected_types)):
    actual = arg.get('type', 'Unknown')
    if actual != expected:
      raise type_mismatch_error(
        func_name,
        f"argument {i+1}",
        expected,
        arg
      )


def dispatch_by_type(
  value: Dict,
  handlers: Dict[str, Callable],
  default_handler: Optional[Callable] = None
) -> Any:
  """
  Generic type-based dispatch

  Args:
    value: Value dict with 'type' field
    handlers: Map of type names to handler functions
    default_handler: Fallback handler

  Returns:
    Result of calling the appropriate handler

  Raises:
    ValueError if no handler found and no default

  Examples:
    dispatch_by_type(
      {"type": "Num", "value": 42},
      {"Num": lambda v: v['value'] * 2}
    ) -> 84
  """
  value_type = value.get('type', 'Unknown')
  handler = handlers.get(value_type, default_handler)
  if handler is None:
    raise ValueError(f"No handler for type: {value_type}")
  return handler(value)


# ==================== BINARY OPERATION FACTORIES ====================

def binary_comparison_op(
  op: Callable[[Any, Any], bool],
  op_name: str,
  allowed_types: Optional[List[str]] = None
) -> Callable[[Dict, Dict, Callable], Dict]:
  """
  Factory for binary comparison operations

  Args:
    op: Python operator function (e.g., operator.lt)
    op_name: Name for error messages
    allowed_types: Types that support this operation

  Returns:
    Function that performs the comparison

  Examples:
    rei1_lt = binary_comparison_op(operator.lt, "less than")
    result = rei1_lt({"type": "Num", "value": 1}, {"type": "Num", "value": 2}, make_value)
  """
  if allowed_types is None:
    allowed_types = ["Num", "String"]

  def comparison(x: Dict, y: Dict, make_value: Callable) -> Dict:
    if x['type'] != y['type']:
      raise operation_error(op_name, x['type'], y['type'])
    if x['type'] not in allowed_types:
      raise operation_error(op_name, x['type'], y['type'])
    return make_value(op(x['value'], y['value']), "Bool")

  return comparison


def binary_arithmetic_op(
  op: Callable[[Any, Any], Any],
  op_name: str,
  allowed_types: Optional[List[str]] = None
) -> Callable[[Dict, Dict, Callable], Dict]:
  """
  Factory for binary arithmetic operations

  Args:
    op: Python operator function (e.g., operator.add)
    op_name: Name for error messages
    allowed_types: Types that support this operation

  Returns:
    Function that performs the arithmetic operation

  Examples:
    rei1_add = binary_arithmetic_op(operator.add, "add")
    result = rei1_add({"type": "Num", "value": 1}, {"type": "Num", "value": 2}, make_value)
  """
  if allowed_types is None:
    allowed_types = ["Num"]

  def arithmetic(x: Dict, y: Dict, make_value: Callable) -> Dict:
    if x['type'] != y['type']:
      raise operation_error(op_name, x['type'], y['type'])
    if x['type'] not in allowed_types:
      raise operation_error(op_name, x['type'], y['type'])
    return make_value(op(x['value'], y['value']), x['type'])

  return arithmetic
