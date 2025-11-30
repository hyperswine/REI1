"""
REI1 Interpreter - Pure Functional Style
No classes, only pure functions and immutable data structures
Side effects (I/O, actors) handled at boundaries
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
import operator
import pykka
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import queue
import uuid
from utilities import (
  is_value_dict,
  validate_function_args,
  type_mismatch_error,
  arity_error,
  extract_from_wrapper
)

# Import stdlib functions
from stdlib import (
    make_value,
    REI1RuntimeError,
    # I/O functions
    rei1_print as stdlib_print_impl,
    rei1_println as stdlib_println_impl,
    rei1_show as stdlib_show_impl,
    # List functions
    rei1_head as stdlib_head_impl,
    rei1_tail as stdlib_tail_impl,
    rei1_length as stdlib_length_impl,
    rei1_cons as stdlib_cons_impl,
    rei1_append as stdlib_concat_impl,
    rei1_reverse as stdlib_reverse_impl,
    rei1_take as stdlib_take_impl,
    rei1_drop as stdlib_drop_impl,
    rei1_elem as stdlib_elem_impl,
    rei1_range as stdlib_range_impl,
    rei1_zip as stdlib_zip_impl,
    # Arithmetic
    rei1_add as stdlib_add_impl,
    rei1_sub as stdlib_sub_impl,
    rei1_mul as stdlib_mul_impl,
    rei1_div as stdlib_div_impl,
    rei1_mod as stdlib_mod_impl,
    # Comparison
    rei1_eq as stdlib_eq_impl,
    rei1_ne as stdlib_ne_impl,
    rei1_lt as stdlib_lt_impl,
    rei1_gt as stdlib_gt_impl,
    rei1_le as stdlib_le_impl,
    rei1_ge as stdlib_ge_impl,
    # Module system
    rei1_Sig as stdlib_sig_impl,
)


# ============================================================================
# DATA STRUCTURES (Immutable Dictionaries)
# ============================================================================

def make_value(value: Any, type_name: str = "Unknown") -> Dict:
  """Create an immutable runtime value"""
  return {
      'value': value,
      'type': type_name
  }


def make_runtime_env(parent: Optional[Dict] = None, bindings: Optional[Dict] = None) -> Dict:
  """Create an immutable runtime environment"""
  return {
      'parent': parent,
      'bindings': bindings or {}
  }


def make_function(params: List[str], body: Dict, closure_env: Dict, contract: Optional[Dict] = None) -> Dict:
  """Create a function value with closure and optional contract"""
  return {
      'type': 'function',
      'params': params,
      'body': body,
      'closure_env': closure_env,
      'contract': contract  # Store contract AST for runtime validation
  }


def make_function_clause(patterns: List[Dict], body: Dict, guard: Optional[Dict] = None, contract: Optional[Dict] = None) -> Dict:
  """Create a single function clause with patterns and optional guard"""
  return {
      'patterns': patterns,
      'guard': guard,
      'body': body,
      'contract': contract
  }


def make_multi_clause_function(name: str, clauses: List[Dict], closure_env: Dict) -> Dict:
  """Create a multi-clause function (for pattern matching dispatch)"""
  return {
      'type': 'multi_clause_function',
      'name': name,
      'clauses': clauses,
      'closure_env': closure_env
  }


def try_match_clause(args: List[Dict], clause: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Optional[Dict]:
  """Try to match arguments against a clause's patterns and check guard

  Args:
      args: List of evaluated argument values
      clause: Function clause with patterns, guard, body, and contract
      env: Environment for pattern matching
      debug: Debug flag
      context: Execution context

  Returns:
      Dictionary of variable bindings if match succeeds and guard passes, None otherwise
  """
  patterns = clause['patterns']

  # Check arity
  if len(args) != len(patterns):
    return None

  # Try to match each argument against its pattern
  bindings = {}
  for i, (arg_val, pattern) in enumerate(zip(args, patterns)):
    # Check if pattern has type constraint (for multiple dispatch)
    # pattern may have 'type_constraint' field added during function definition
    type_constraint = pattern.get('type_constraint')

    if debug:
      print(f"  Pattern {i}: {pattern}")
      print(
          f"  Arg {i}: type={arg_val.get('type')}, value={arg_val.get('value')}")
      print(f"  Type constraint: {type_constraint}")

    if type_constraint:
      # This is a typed pattern - check if arg_val's type matches
      arg_type = arg_val.get('type', 'Unknown')
      expected_type = type_constraint.get('name', type_constraint) if isinstance(
          type_constraint, dict) else type_constraint

      if debug:
        print(f"  Checking type: {arg_type} vs {expected_type}")

      # Type matching
      if expected_type == 'List':
        # For List type, accept any List regardless of element type
        if arg_type != 'List':
          if debug:
            print(f"  Type mismatch: List required but got {arg_type}")
          return None
      elif arg_type != expected_type:
        # Type mismatch - this clause doesn't match
        if debug:
          print(
              f"  Type mismatch: {expected_type} required but got {arg_type}")
        return None

    pattern_type = pattern['type']

    if pattern_type == 'PATTERN_VAR':
      # Variable pattern always matches and binds the value
      var_name = pattern['value']
      bindings[var_name] = arg_val

    elif pattern_type == 'PATTERN_LIST':
      # List pattern (e.g., []) must match an empty list
      pattern_list = pattern['value']
      arg_list = arg_val.get('value', [])

      # For now, only support empty list pattern []
      if isinstance(pattern_list, list) and len(pattern_list) == 0:
        # Empty list pattern - match only empty lists
        if not isinstance(arg_list, list) or len(arg_list) != 0:
          return None
      else:
        # TODO: Support non-empty list patterns
        return None

    elif pattern_type == 'PATTERN_CONS':
      # List cons pattern (x::xs) must match and bind head and tail
      match_result = matches_pattern(arg_val, pattern, env)
      if match_result is None:
        return None
      # Add any extracted bindings
      bindings.update(match_result)

    elif pattern_type == 'PATTERN_LITERAL':
      # Literal pattern must match exactly
      match_result = matches_pattern(arg_val, pattern, env)
      if match_result is None:
        return None

    elif pattern_type == 'PATTERN_CONSTRUCTOR':
      # Constructor pattern must match structure and extract bindings
      match_result = matches_pattern(arg_val, pattern, env)
      if match_result is None:
        return None
      # Add any extracted bindings
      bindings.update(match_result)

    elif pattern_type == 'PARTIAL_PLACEHOLDER':
      # Wildcard matches anything, no binding
      pass

    else:
      # Unknown pattern type - fail to match
      return None

  # Patterns matched! Now check the guard (if any)
  guard = clause.get('guard')
  if guard:
    # Create environment with pattern bindings
    guard_env = env
    for var_name, var_value in bindings.items():
      guard_env = env_bind_value(guard_env, var_name, var_value)

    # Check for 'otherwise' keyword (always true)
    if guard['type'] == 'IDENTIFIER' and guard['value'] == 'otherwise':
      # otherwise is always true
      return bindings

    # Evaluate the guard expression
    try:
      if context is None:
        context = make_execution_context()
      guard_result, _ = eval_ast(guard, guard_env, debug, context)

      # Guard must evaluate to true
      if guard_result.get('type') == 'Bool':
        if guard_result.get('value') == True:
          return bindings
        else:
          return None  # Guard failed
      else:
        # Non-boolean guard - treat as false
        return None
    except Exception:
      # Guard evaluation failed - treat as false
      return None

  # No guard or guard passed - return bindings
  return bindings


def make_execution_context(unsafe_allowed: bool = False) -> Dict:
  """Create an execution context for tracking unsafe operations"""
  return {
      'unsafe_allowed': unsafe_allowed
  }


# ============================================================================
# ENVIRONMENT OPERATIONS
# ============================================================================

def env_bind_value(env: Dict, name: str, value: Dict) -> Dict:
  """Return new environment with name bound to value"""
  return {
      **env,
      'bindings': {**env['bindings'], name: value}
  }


def env_lookup_value(env: Dict, name: str) -> Optional[Dict]:
  """Look up a value in the environment chain"""
  if name in env['bindings']:
    return env['bindings'][name]
  elif env['parent']:
    return env_lookup_value(env['parent'], name)
  return None


def unwrap_value(val: Any) -> Any:
  """Recursively unwrap value dicts to extract raw Python values"""
  if isinstance(val, dict) and 'value' in val and 'type' in val:
    # It's a wrapped value - extract the value and recurse
    inner = val['value']
    if isinstance(inner, list):
      # Unwrap list elements
      return [unwrap_value(elem) for elem in inner]
    elif isinstance(inner, dict) and 'value' in inner:
      # Nested wrapped value
      return unwrap_value(inner)
    else:
      return inner
  elif isinstance(val, list):
    # Plain list - unwrap elements
    return [unwrap_value(elem) for elem in val]
  else:
    return val


# ============================================================================
# BUILT-IN OPERATIONS
# ============================================================================

BUILTIN_OPERATORS = {
    '+': stdlib_add_impl,
    '-': stdlib_sub_impl,
    '*': stdlib_mul_impl,
    '/': stdlib_div_impl,
    '%': stdlib_mod_impl,
    '==': stdlib_eq_impl,
    '!=': stdlib_ne_impl,
    '/=': stdlib_ne_impl,  # Haskell-style not-equal
    '<': stdlib_lt_impl,
    '>': stdlib_gt_impl,
    '<=': stdlib_le_impl,
    '>=': stdlib_ge_impl,
    '++': stdlib_add_impl,  # String/list concatenation (same as +)
}


# ============================================================================
# UNSAFE I/O PRIMITIVES
# ============================================================================

def io_read(filename: Dict, context: Dict) -> str:
  """Read file content - UNSAFE"""
  if not context.get('unsafe_allowed', False):
    raise RuntimeError("IO.read must be used within an 'unsafe' block")
  try:
    fname = filename['value'] if isinstance(
        filename, dict) and 'value' in filename else filename
    with open(fname, 'r') as f:
      return f.read()
  except Exception as e:
    raise RuntimeError(f"IO.read error: {e}")


def io_write(filename: Dict, content: Dict, context: Dict) -> None:
  """Write content to file - UNSAFE"""
  if not context.get('unsafe_allowed', False):
    raise RuntimeError("IO.write must be used within an 'unsafe' block")
  try:
    fname = filename['value'] if isinstance(
        filename, dict) and 'value' in filename else filename
    cont = content['value'] if isinstance(
        content, dict) and 'value' in content else content
    with open(fname, 'w') as f:
      f.write(cont)
  except Exception as e:
    raise RuntimeError(f"IO.write error: {e}")


# ============================================================================
# ACTOR SYSTEM (Using Pykka)
# ============================================================================

class ActorRegistry:
  """Registry for managing actors"""

  def __init__(self):
    self.actors: Dict[str, pykka.ActorRef] = {}

  def register(self, actor_id: str, actor_ref: pykka.ActorRef):
    """Register an actor"""
    self.actors[actor_id] = actor_ref

  def get_actor(self, actor_id: str) -> Optional[pykka.ActorRef]:
    """Get actor by ID"""
    return self.actors.get(actor_id)

  def terminate_all(self):
    """Terminate all actors"""
    for actor_ref in self.actors.values():
      try:
        actor_ref.stop()
      except Exception:
        pass
    self.actors.clear()


# Global actor registry
_actor_registry = ActorRegistry()


class REI1Actor(pykka.ThreadingActor):
  """Actor that runs REI1 functions"""

  def __init__(self, actor_id: str, handler_func: Dict, env: Dict):
    super().__init__()
    self.actor_id = actor_id
    self.handler_func = handler_func
    self.env = env
    self.mailbox = queue.Queue()

  def on_receive(self, message):
    """Handle incoming message by calling the handler function"""
    try:
      # Call the handler function with the message
      if self.handler_func['type'] == 'function':
        # Wrap message as appropriate type based on Python type
        if isinstance(message, int) or isinstance(message, float):
          msg_val = make_value(message, "Num")
        elif isinstance(message, str):
          msg_val = make_value(message, "String")
        elif isinstance(message, list):
          msg_val = make_value(message, "List")
        elif isinstance(message, bool):
          msg_val = make_value(message, "Bool")
        else:
          msg_val = make_value(message, "Unknown")

        # Bind message to function parameter
        params = self.handler_func['params']
        if len(params) >= 1:
          call_env = env_bind_value(
              self.handler_func['closure_env'], params[0], msg_val)
          # Evaluate function body
          result_val, _ = eval_ast(self.handler_func['body'], call_env, False)
          # Store the result (not the input) in mailbox for Proc.recv
          self.mailbox.put(result_val)
          return result_val
      return None
    except Exception as e:
      print(f"Actor {self.actor_id} error: {e}")
      return None


def proc_spawn(handler_func: Dict, env: Dict) -> Dict:
  """Spawn a new actor with given handler function - UNSAFE"""
  actor_id = str(uuid.uuid4())
  actor = REI1Actor.start(actor_id, handler_func, env)
  _actor_registry.register(actor_id, actor)
  return make_value(actor_id, "ActorID")


def proc_send(actor_id: Dict, message: Dict) -> None:
  """Send message to actor - UNSAFE"""
  # Unwrap the actor ID and message
  aid = unwrap_value(actor_id)
  msg = unwrap_value(message)

  actor_ref = _actor_registry.get_actor(aid)
  if actor_ref:
    actor_ref.tell(msg)
  else:
    raise RuntimeError(f"Actor not found: {aid}")


def proc_recv(actor_id: Dict, timeout: Optional[float] = None) -> Any:
  """Receive message from actor's mailbox - UNSAFE"""
  # Unwrap the actor ID
  aid = unwrap_value(actor_id)

  actor_ref = _actor_registry.get_actor(aid)
  if actor_ref:
    actor = actor_ref.proxy()
    try:
      # Get the mailbox from the actor
      mailbox = actor.mailbox.get()
      if mailbox:
        try:
          # The mailbox contains the result value dict from the actor's handler
          result = mailbox.get(timeout=timeout if timeout else 5.0)
          # Result is already a value dict, return it directly
          return result
        except queue.Empty:
          return make_value(None, "Unit")
      return make_value(None, "Unit")
    except Exception as e:
      raise RuntimeError(f"Error receiving from actor: {e}")
  else:
    raise RuntimeError(f"Actor not found: {aid}")


# ============================================================================
# PARALLEL PROCESSING
# ============================================================================

def par_map_impl(func: Dict, lst: Dict, env: Dict) -> Dict:
  """Parallel map implementation"""
  validate_function_args("par-map", [func, lst], ["function", "List"])

  # For now, use sequential execution (full parallel requires picklable closures)
  # TODO: Implement proper parallel execution with process pool
  results = []
  for elem in lst['value']:
    # Ensure elem is wrapped as a value dict
    if not is_value_dict(elem):
      elem = make_value(elem, type(elem).__name__)
    # Apply function to each element
    if len(func['params']) >= 1:
      call_env = env_bind_value(func['closure_env'], func['params'][0], elem)
      result_val, _ = eval_ast(func['body'], call_env, False)
      results.append(result_val)
  return make_value(results, "List")


def par_filter_impl(pred: Dict, lst: Dict, env: Dict) -> Dict:
  """Parallel filter implementation"""
  validate_function_args("par-filter", [pred, lst], ["function", "List"])

  results = []
  for elem in lst['value']:
    # Ensure elem is wrapped as a value dict
    if not is_value_dict(elem):
      elem = make_value(elem, type(elem).__name__)
    # Apply predicate to each element
    if len(pred['params']) >= 1:
      call_env = env_bind_value(pred['closure_env'], pred['params'][0], elem)
      result_val, _ = eval_ast(pred['body'], call_env, False)
      if result_val['type'] == 'Bool' and result_val['value']:
        results.append(elem)
  return make_value(results, "List")


def par_fold_impl(func: Dict, init: Dict, lst: Dict, env: Dict) -> Dict:
  """Parallel fold implementation (sequential for now)"""
  validate_function_args("par-fold", [func, lst], ["function", "List"])

  acc = init
  for elem in lst['value']:
    # Ensure elem is wrapped as a value dict
    if not is_value_dict(elem):
      elem = make_value(elem, type(elem).__name__)
    # Apply function to accumulator and element
    if len(func['params']) >= 2:
      call_env = env_bind_value(func['closure_env'], func['params'][0], elem)
      call_env = env_bind_value(call_env, func['params'][1], acc)
      acc, _ = eval_ast(func['body'], call_env, False)
  return acc


# ============================================================================
# CONTRACT VALIDATION
# ============================================================================

def validate_contract(contract_ast: Dict, param_names: List[str], arg_values: List[Dict],
                      result_value: Optional[Dict], env: Dict, is_precondition: bool, debug: bool = False) -> None:
  """Validate a contract condition (pre or post)

  Args:
      contract_ast: The contract AST node with 'pre' and 'post' conditions
      param_names: List of parameter names
      arg_values: List of argument values
      result_value: The result value (for postconditions, None for preconditions)
      env: The environment to use for evaluation
      is_precondition: True if checking precondition, False for postcondition
      debug: Enable debug output
  """
  if not contract_ast or contract_ast.get('type') != 'CONTRACT':
    return

  contract_value = contract_ast.get('value', {})
  condition_ast = contract_value.get('pre' if is_precondition else 'post')

  if not condition_ast:
    return

  # Build environment with parameter bindings
  check_env = env

  # Add operators to environment so they can be called by name in contracts
  for op_name, op_func in BUILTIN_OPERATORS.items():
    # Wrap operator as a builtin function value
    op_value = make_value({
        'type': 'builtin_function',
        'name': op_name,
        'func': op_func,
        'arity': 2,  # All operators are binary
        'needs_context': False
    }, "BuiltinFunction")
    check_env = env_bind_value(check_env, op_name, op_value)

  for param_name, arg_value in zip(param_names, arg_values):
    check_env = env_bind_value(check_env, param_name, arg_value)

  # For postconditions, also bind 'result'
  if not is_precondition and result_value is not None:
    check_env = env_bind_value(check_env, 'result', result_value)

  if debug:
    condition_type = "precondition" if is_precondition else "postcondition"
    print(f"Checking {condition_type}: {condition_ast}")

  # Evaluate the condition
  try:
    condition_result, _ = eval_ast(condition_ast, check_env, debug)

    if condition_result['type'] != 'Bool':
      raise REI1RuntimeError(
          f"Contract condition must be boolean, got {condition_result['type']}")

    if not condition_result['value']:
      condition_type = "Precondition" if is_precondition else "Postcondition"
      # Build helpful error message
      if is_precondition:
        args_str = ', '.join(
            f"{name}={val['value']}" for name, val in zip(param_names, arg_values))
        raise REI1RuntimeError(
            f"{condition_type} violated with arguments: {args_str}")
      else:
        result_str = result_value['value'] if result_value else 'None'
        raise REI1RuntimeError(
            f"{condition_type} violated: result={result_str}")
  except REI1RuntimeError:
    raise
  except Exception as e:
    raise REI1RuntimeError(f"Error evaluating contract: {e}")


# ============================================================================
# STANDARD LIBRARY FUNCTIONS (Higher-order functions that need eval_ast)
# ============================================================================

def stdlib_map(func: Dict, lst: Dict, env: Dict) -> Dict:
  """Map function over list"""
  validate_function_args("map", [func, lst], ["function", "List"])

  results = []
  for elem in lst['value']:
    if len(func['params']) >= 1:
      call_env = env_bind_value(func['closure_env'], func['params'][0], elem)
      result_val, _ = eval_ast(func['body'], call_env, False)
      results.append(result_val)
  return make_value(results, "List")


def stdlib_filter(pred: Dict, lst: Dict, env: Dict) -> Dict:
  """Filter list with predicate"""
  validate_function_args("filter", [pred, lst], ["function", "List"])

  results = []
  for elem in lst['value']:
    if len(pred['params']) >= 1:
      call_env = env_bind_value(pred['closure_env'], pred['params'][0], elem)
      result_val, _ = eval_ast(pred['body'], call_env, False)
      if result_val['type'] == 'Bool' and result_val['value']:
        results.append(elem)
  return make_value(results, "List")


def stdlib_fold(func: Dict, init: Dict, lst: Dict, env: Dict) -> Dict:
  """Fold list with function and initial value"""
  validate_function_args("fold", [func, lst], ["function", "List"])

  acc = init
  for elem in lst['value']:
    if len(func['params']) >= 2:
      call_env = env_bind_value(func['closure_env'], func['params'][0], elem)
      call_env = env_bind_value(call_env, func['params'][1], acc)
      acc, _ = eval_ast(func['body'], call_env, False)
  return acc


# Wrappers for stdlib functions to match interface expected by environment
def stdlib_print(value: Dict) -> Dict:
  """Print value without newline"""
  return stdlib_print_impl(value)


def stdlib_println(value: Dict) -> Dict:
  """Print value with newline"""
  return stdlib_println_impl(value)


def stdlib_show(value: Dict) -> Dict:
  """Convert value to string representation"""
  return stdlib_show_impl(value)


def stdlib_head(lst: Dict) -> Dict:
  """Get first element of list"""
  return stdlib_head_impl(lst)


def stdlib_tail(lst: Dict) -> Dict:
  """Get tail of list"""
  return stdlib_tail_impl(lst)


def stdlib_length(lst: Dict) -> Dict:
  """Get length of list"""
  return stdlib_length_impl(lst)


def stdlib_concat(lst1: Dict, lst2: Dict) -> Dict:
  """Concatenate two lists or strings"""
  return stdlib_concat_impl(lst1, lst2)


def stdlib_cons(elem: Dict, lst: Dict) -> Dict:
  """Prepend element to list"""
  return stdlib_cons_impl(elem, lst)


def stdlib_reverse(lst: Dict) -> Dict:
  """Reverse a list"""
  return stdlib_reverse_impl(lst)


def stdlib_take(n: Dict, lst: Dict) -> Dict:
  """Take first n elements from list"""
  return stdlib_take_impl(n, lst)


def stdlib_drop(n: Dict, lst: Dict) -> Dict:
  """Drop first n elements from list"""
  return stdlib_drop_impl(n, lst)


def stdlib_elem(val: Dict, lst: Dict) -> Dict:
  """Check if element is in list"""
  return stdlib_elem_impl(val, lst)


def stdlib_range(start: Dict, end: Dict) -> Dict:
  """Create a range of numbers"""
  return stdlib_range_impl(start, end)


def stdlib_zip(lst1: Dict, lst2: Dict) -> Dict:
  """Zip two lists into list of pairs"""
  return stdlib_zip_impl(lst1, lst2)


def rei1_Type(constructor_ast: Dict) -> Dict:
  """Type constructor function - creates ADT constructors at runtime

  Usage: User = Type $ User String Num String

  This runtime function receives an AST node (FUNCTION_CALL) and creates
  a constructor function. The AST represents something like "User String Num String"
  parsed as User(String, Num, String).

  Returns a constructor function that creates tagged instances.
  """
  if not isinstance(constructor_ast, dict):
    raise RuntimeError(f"Type: expected AST dict, got {type(constructor_ast)}")

  # The AST should be a FUNCTION_CALL node
  if constructor_ast.get('type') != 'FUNCTION_CALL':
    raise RuntimeError(
        f"Type: expected FUNCTION_CALL, got {constructor_ast.get('type')}")

  # Extract function name and args count from AST
  func_node = constructor_ast.get('value', {}).get('function', {})
  args = constructor_ast.get('value', {}).get('args', [])

  # Get constructor name from the function node
  if func_node.get('type') == 'FUNCTION_NAME':
    ctor_name = func_node.get('value', 'Unknown')
  elif func_node.get('type') == 'IDENTIFIER':
    ctor_name = func_node.get('value', 'Unknown')
  else:
    ctor_name = 'Unknown'

  arity = len(args)

  # Return a constructor value that can be called
  return {
      'type': 'constructor',
      'ctor_name': ctor_name,
      'type_name': ctor_name,  # Type name same as constructor for simple ADTs
      'arity': arity
  }


def rei1_unsafe(expr_ast: Dict, env: Dict, context: Dict) -> Dict:
  """Unsafe function - enables unsafe context for IO operations
  Usage: unsafe $ IO.read "file.txt"

  This function receives the unevaluated AST and evaluates it with
  unsafe_allowed=True in the context, allowing IO operations.
  """
  # Create a new context with unsafe enabled
  unsafe_context = make_execution_context(unsafe_allowed=True)

  # Evaluate the expression with the unsafe context
  result, _ = eval_ast(expr_ast, env, context=unsafe_context)
  return result


def create_builtin_runtime_env() -> Dict:
  """Create runtime environment with built-in operations"""
  env = make_runtime_env()

  # Bind built-in operators as special functions
  for op_name in BUILTIN_OPERATORS.keys():
    env = env_bind_value(env, op_name, make_value(op_name, "BuiltinOperator"))

  # Bind IO functions
  io_read_func = make_value({
      'type': 'builtin_function',
      'name': 'IO.read',
      'func': io_read,
      'arity': 1,
      'needs_context': True  # IO operations need context to check unsafe
  }, "BuiltinFunction")
  env = env_bind_value(env, "IO.read", io_read_func)

  io_write_func = make_value({
      'type': 'builtin_function',
      'name': 'IO.write',
      'func': io_write,
      'arity': 2,
      'needs_context': True  # IO operations need context to check unsafe
  }, "BuiltinFunction")
  env = env_bind_value(env, "IO.write", io_write_func)

  # Bind Proc (actor) functions
  proc_spawn_func = make_value({
      'type': 'builtin_function',
      'name': 'Proc.spawn',
      'func': proc_spawn,
      'arity': 1,
      'needs_env': True
  }, "BuiltinFunction")
  env = env_bind_value(env, "Proc.spawn", proc_spawn_func)

  proc_send_func = make_value({
      'type': 'builtin_function',
      'name': 'Proc.send',
      'func': proc_send,
      'arity': 2
  }, "BuiltinFunction")
  env = env_bind_value(env, "Proc.send", proc_send_func)

  proc_recv_func = make_value({
      'type': 'builtin_function',
      'name': 'Proc.recv',
      'func': proc_recv,
      'arity': 1
  }, "BuiltinFunction")
  env = env_bind_value(env, "Proc.recv", proc_recv_func)

  # Bind parallel processing functions
  par_map_func = make_value({
      'type': 'builtin_function',
      'name': 'par-map',
      'func': par_map_impl,
      'arity': 2,
      'needs_env': True
  }, "BuiltinFunction")
  env = env_bind_value(env, "par-map", par_map_func)

  par_filter_func = make_value({
      'type': 'builtin_function',
      'name': 'par-filter',
      'func': par_filter_impl,
      'arity': 2,
      'needs_env': True
  }, "BuiltinFunction")
  env = env_bind_value(env, "par-filter", par_filter_func)

  par_fold_func = make_value({
      'type': 'builtin_function',
      'name': 'par-fold',
      'func': par_fold_impl,
      'arity': 3,
      'needs_env': True
  }, "BuiltinFunction")
  env = env_bind_value(env, "par-fold", par_fold_func)

  # Bind standard library functions
  map_func = make_value({
      'type': 'builtin_function',
      'name': 'map',
      'func': stdlib_map,
      'arity': 2,
      'needs_env': True
  }, "BuiltinFunction")
  env = env_bind_value(env, "map", map_func)

  filter_func = make_value({
      'type': 'builtin_function',
      'name': 'filter',
      'func': stdlib_filter,
      'arity': 2,
      'needs_env': True
  }, "BuiltinFunction")
  env = env_bind_value(env, "filter", filter_func)

  fold_func = make_value({
      'type': 'builtin_function',
      'name': 'fold',
      'func': stdlib_fold,
      'arity': 3,
      'needs_env': True
  }, "BuiltinFunction")
  env = env_bind_value(env, "fold", fold_func)

  head_func = make_value({
      'type': 'builtin_function',
      'name': 'head',
      'func': stdlib_head,
      'arity': 1
  }, "BuiltinFunction")
  env = env_bind_value(env, "head", head_func)

  tail_func = make_value({
      'type': 'builtin_function',
      'name': 'tail',
      'func': stdlib_tail,
      'arity': 1
  }, "BuiltinFunction")
  env = env_bind_value(env, "tail", tail_func)

  length_func = make_value({
      'type': 'builtin_function',
      'name': 'length',
      'func': stdlib_length,
      'arity': 1
  }, "BuiltinFunction")
  env = env_bind_value(env, "length", length_func)

  concat_func = make_value({
      'type': 'builtin_function',
      'name': 'concat',
      'func': stdlib_concat,
      'arity': 2
  }, "BuiltinFunction")
  env = env_bind_value(env, "concat", concat_func)

  cons_func = make_value({
      'type': 'builtin_function',
      'name': 'cons',
      'func': stdlib_cons,
      'arity': 2
  }, "BuiltinFunction")
  env = env_bind_value(env, "cons", cons_func)

  print_func = make_value({
      'type': 'builtin_function',
      'name': 'print',
      'func': stdlib_print,
      'arity': 1
  }, "BuiltinFunction")
  env = env_bind_value(env, "print", print_func)

  println_func = make_value({
      'type': 'builtin_function',
      'name': 'println',
      'func': stdlib_println,
      'arity': 1
  }, "BuiltinFunction")
  env = env_bind_value(env, "println", println_func)

  show_func = make_value({
      'type': 'builtin_function',
      'name': 'show',
      'func': stdlib_show,
      'arity': 1
  }, "BuiltinFunction")
  env = env_bind_value(env, "show", show_func)

  # Additional list functions
  reverse_func = make_value({
      'type': 'builtin_function',
      'name': 'reverse',
      'func': stdlib_reverse,
      'arity': 1
  }, "BuiltinFunction")
  env = env_bind_value(env, "reverse", reverse_func)

  take_func = make_value({
      'type': 'builtin_function',
      'name': 'take',
      'func': stdlib_take,
      'arity': 2
  }, "BuiltinFunction")
  env = env_bind_value(env, "take", take_func)

  drop_func = make_value({
      'type': 'builtin_function',
      'name': 'drop',
      'func': stdlib_drop,
      'arity': 2
  }, "BuiltinFunction")
  env = env_bind_value(env, "drop", drop_func)

  elem_func = make_value({
      'type': 'builtin_function',
      'name': 'elem',
      'func': stdlib_elem,
      'arity': 2
  }, "BuiltinFunction")
  env = env_bind_value(env, "elem", elem_func)

  range_func = make_value({
      'type': 'builtin_function',
      'name': 'range',
      'func': stdlib_range,
      'arity': 2
  }, "BuiltinFunction")
  env = env_bind_value(env, "range", range_func)

  zip_func = make_value({
      'type': 'builtin_function',
      'name': 'zip',
      'func': stdlib_zip,
      'arity': 2
  }, "BuiltinFunction")
  env = env_bind_value(env, "zip", zip_func)

  # Special runtime functions
  # Type - receives AST, creates constructors at runtime
  type_func = make_value({
      'type': 'builtin_function',
      'name': 'Type',
      'func': rei1_Type,
      'arity': 1,
      'receives_ast': True  # Special flag: receives unevaluated AST
  }, "BuiltinFunction")
  env = env_bind_value(env, "Type", type_func)

  # unsafe - enables unsafe context and evaluates expression
  unsafe_func = make_value({
      'type': 'builtin_function',
      'name': 'unsafe',
      'func': rei1_unsafe,
      'arity': 1,
      'receives_ast': True,   # Receives unevaluated AST
      'needs_context': True   # Needs context and env to evaluate with unsafe enabled
  }, "BuiltinFunction")
  env = env_bind_value(env, "unsafe", unsafe_func)

  # Sig - creates a signature (module interface)
  sig_func = make_value({
      'type': 'builtin_function',
      'name': 'Sig',
      'func': stdlib_sig_impl,
      'arity': 1
  }, "BuiltinFunction")
  env = env_bind_value(env, "Sig", sig_func)

  # Bool values - constructor instances (not constructor functions)
  # These are actual Bool values, not functions that create Bool values
  true_val = make_value({
      'constructor': 'True',
      'type': 'Bool',
      'fields': []
  }, "Bool")
  env = env_bind_value(env, "True", true_val)

  false_val = make_value({
      'constructor': 'False',
      'type': 'Bool',
      'fields': []
  }, "Bool")
  env = env_bind_value(env, "False", false_val)

  return env


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def eval_ast(ast_node: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """
  Evaluate an AST node and return (result_value, updated_environment).
  This is a pure function that threads the environment through evaluation.
  """
  if context is None:
    context = make_execution_context()

  if debug:
    print(f"Evaluating: {ast_node['type']}")

  node_type = ast_node['type']

  if node_type == "NUMBER":
    return eval_number(ast_node, env, debug, context)
  elif node_type == "STRING":
    return eval_string(ast_node, env, debug, context)
  elif node_type == "IDENTIFIER":
    return eval_identifier(ast_node, env, debug, context)
  elif node_type == "FUNCTION_CALL":
    return eval_function_call(ast_node, env, debug, context)
  elif node_type == "FUNCTION_NAME":
    return eval_function_name(ast_node, env, debug, context)
  elif node_type == "FUNCTION_DEF":
    return eval_function_def(ast_node, env, debug, context)
  elif node_type == "VALUE_BINDING":
    return eval_value_binding(ast_node, env, debug, context)
  elif node_type == "LIST":
    return eval_list(ast_node, env, debug, context)
  elif node_type == "RECORD":
    return eval_record(ast_node, env, debug, context)
  elif node_type == "LAMBDA":
    return eval_lambda(ast_node, env, debug, context)
  elif node_type == "CASE":
    return eval_case(ast_node, env, debug, context)
  elif node_type == "OPERATION":
    return eval_operation(ast_node, env, debug, context)
  elif node_type == "TYPE_DEF":
    return eval_type_def(ast_node, env, debug, context)
  elif node_type == "INLINE_EVAL":
    return eval_inline_eval(ast_node, env, debug, context)
  elif node_type == "PARTIAL_PLACEHOLDER":
    # Placeholder for partial application
    return make_value("_placeholder_", "Placeholder"), env
  elif node_type == "NOT_IMPLEMENTED":
    # ?? operator - unimplemented placeholder
    # Get function name from context if available (this will be called when evaluating function body)
    func_name = context.get(
        'current_function', '<unknown>') if context else '<unknown>'
    raise REI1RuntimeError(
        f"Function '{func_name}' is not yet implemented (marked with ??)")
  # Note: PIPELINE is now desugared to FUNCTION_CALL in semantic phase
  else:
    if debug:
      print(f"Unknown node type: {node_type}")
    return make_value(None, "Unknown"), env


def eval_number(ast_node: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """Evaluate number literal"""
  value = ast_node['value']
  return make_value(value, "Num"), env


def eval_string(ast_node: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """Evaluate string literal"""
  value = ast_node['value']
  return make_value(value, "String"), env


def eval_identifier(ast_node: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """Evaluate identifier by looking up in environment"""
  name = ast_node['value']
  value = env_lookup_value(env, name)

  if value is None:
    raise RuntimeError(f"Unbound identifier: {name}")

  return value, env


def eval_function_name(ast_node: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """Evaluate function name (similar to identifier)"""
  name = ast_node['value']
  value = env_lookup_value(env, name)

  if value is None:
    # Check if it's a built-in operator
    if name in BUILTIN_OPERATORS:
      return make_value(name, "BuiltinOperator"), env
    raise RuntimeError(f"Unbound function: {name}")

  return value, env


def eval_function_call(ast_node: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """Evaluate function application"""
  if context is None:
    context = make_execution_context()

  value_dict = ast_node['value']

  # Evaluate function
  func_ast = value_dict['function']
  func_val, env = eval_ast(func_ast, env, debug, context)

  # Check if this builtin receives AST instead of evaluated args
  receives_ast = False
  needs_context = False
  if func_val['type'] == 'BuiltinFunction':
    func_data = func_val['value']
    receives_ast = func_data.get('receives_ast', False)
    needs_context = func_data.get('needs_context', False)

  # Evaluate arguments (unless function receives AST)
  args = []
  args_ast = []  # Keep AST for functions that receive it
  has_placeholders = False
  placeholder_positions = []

  for i, arg_ast in enumerate(value_dict['args']):
    args_ast.append(arg_ast)

    if not receives_ast:
      arg_val, env = eval_ast(arg_ast, env, debug, context)
      args.append(arg_val)

      # Check if this is a placeholder
      if arg_val['type'] == 'Placeholder' and arg_val['value'] == '_placeholder_':
        has_placeholders = True
        placeholder_positions.append(i)

  # If there are placeholders, create a partially applied function
  if has_placeholders:
    # Create a new function that fills in the non-placeholder arguments
    def make_partial_func(original_func, args_with_placeholders, positions):
      # Return a function that takes the missing arguments
      param_names = [f"_arg{i}" for i in positions]

      # Create a lambda-like function body
      # For now, we'll create a closure
      partial_func = {
          'type': 'partial_function',
          'original_func': original_func,
          'fixed_args': args_with_placeholders,
          'placeholder_positions': positions,
          'params': param_names
      }
      return partial_func

    partial = make_partial_func(func_val, args, placeholder_positions)
    return make_value(partial, "PartialFunction"), env

  # Regular function application (no placeholders)

  # Apply function
  if func_val['type'] == 'BuiltinOperator':
    # Built-in operator (now uses stdlib functions that expect Dict values)
    op_name = func_val['value']
    if op_name in BUILTIN_OPERATORS:
      op_func = BUILTIN_OPERATORS[op_name]
      if len(args) >= 2:
        # Operators now expect wrapped Dict values, not raw values
        result = op_func(args[0], args[1])
        return result, env
      else:
        raise RuntimeError(
            f"Operator {op_name} requires 2 arguments, got {len(args)}")
    else:
      raise RuntimeError(f"Unknown operator: {op_name}")
  elif func_val['type'] == 'BuiltinFunction':
    # Built-in function (like IO.read, IO.write, map, filter, etc.)
    func_data = func_val['value']
    func_impl = func_data['func']
    arity = func_data['arity']
    needs_env = func_data.get('needs_env', False)
    needs_context_flag = func_data.get('needs_context', False)
    receives_ast_flag = func_data.get('receives_ast', False)

    # For functions that receive AST, pass AST nodes instead of evaluated values
    if receives_ast_flag:
      if len(args_ast) != arity:
        raise RuntimeError(
            f"Function {func_data['name']} expects {arity} arguments, got {len(args_ast)}")

      try:
        if needs_context_flag:
          # Function needs both AST and context (like unsafe)
          if arity == 1:
            result = func_impl(args_ast[0], env, context)
          else:
            raise RuntimeError(
                f"Unsupported arity for receives_ast+needs_context: {arity}")
        else:
          # Function only needs AST (like Type)
          if arity == 1:
            result = func_impl(args_ast[0])
          else:
            raise RuntimeError(f"Unsupported arity for receives_ast: {arity}")

        # Wrap result if needed
        if isinstance(result, dict) and 'type' in result:
          return result, env
        else:
          return make_value(result, "Unknown"), env
      except Exception as e:
        raise RuntimeError(f"Error calling {func_data['name']}: {e}")

    # Regular evaluation path
    # Support partial application: if fewer args provided than expected, create partial function
    if len(args) < arity:
      # Create a partially applied builtin function
      partial_data = {
          'type': 'partial_builtin',
          'original_func': func_val,
          'fixed_args': args,
          'remaining_arity': arity - len(args)
      }
      return make_value(partial_data, "PartialFunction"), env
    elif len(args) > arity:
      raise RuntimeError(
          f"Function {func_data['name']} expects {arity} arguments, got {len(args)}")

    # Call the built-in function with argument values
    try:
      if needs_context_flag:
        # Functions that need context (IO operations)
        if arity == 1:
          result = func_impl(args[0], context)
        elif arity == 2:
          result = func_impl(args[0], args[1], context)
        elif arity == 3:
          result = func_impl(args[0], args[1], args[2], context)
        else:
          raise RuntimeError(f"Unsupported arity: {arity}")
      elif needs_env:
        # Functions that need environment (map, filter, fold, etc.)
        if arity == 1:
          result = func_impl(args[0], env)
        elif arity == 2:
          result = func_impl(args[0], args[1], env)
        elif arity == 3:
          result = func_impl(args[0], args[1], args[2], env)
        else:
          raise RuntimeError(f"Unsupported arity: {arity}")
      else:
        # Regular built-in functions
        if arity == 1:
          # Pass the wrapped Dict directly - stdlib functions expect it
          result = func_impl(args[0])
        elif arity == 2:
          result = func_impl(args[0], args[1])
        elif arity == 3:
          result = func_impl(args[0], args[1], args[2])
        else:
          raise RuntimeError(f"Unsupported arity: {arity}")

      # Wrap result if it's not already a dict with type
      if isinstance(result, dict) and 'type' in result:
        return result, env
      else:
        return make_value(result, "Unknown"), env
    except Exception as e:
      raise RuntimeError(f"Error calling {func_data['name']}: {e}")
  elif func_val['type'] == 'constructor':
    # Type constructor - create a data structure
    ctor_name = func_val['ctor_name']
    type_name = func_val['type_name']
    arity = func_val['arity']

    if len(args) != arity:
      raise RuntimeError(
          f"Constructor {ctor_name} expects {arity} arguments, got {len(args)}")

    # Create a tagged value (type_name, ctor_name, arg_values...)
    ctor_data = {
        'constructor': ctor_name,
        'type': type_name,
        'fields': [arg['value'] for arg in args]
    }
    return make_value(ctor_data, type_name), env
  elif func_val['type'] == 'Signature':
    # Calling a Signature value creates a Module
    # MySig { add: λ x y => + x y, ... } creates a module
    if len(args) != 1:
      raise RuntimeError(
          f"Signature expects 1 argument (implementation record), got {len(args)}")

    implementation = args[0]

    # Create a module value
    module_data = {
        'signature': func_val['value'],  # The signature data
        'implementation': implementation,
        'module_type': 'Module'
    }
    return make_value(module_data, "Module"), env
  elif func_val['type'] == 'function':
    # User-defined function
    func_params = func_val['params']
    func_body = func_val['body']
    func_env = func_val['closure_env']
    func_contract = func_val.get('contract')

    # Support partial application: if fewer args provided than expected, create partial function
    if len(args) < len(func_params):
      # Create a partially applied user function
      partial_data = {
          'type': 'partial_user_func',
          'original_func': func_val,
          'fixed_args': args,
          'remaining_params': func_params[len(args):]
      }
      return make_value(partial_data, "PartialFunction"), env
    elif len(args) > len(func_params):
      raise RuntimeError(
          f"Function expects {len(func_params)} arguments, got {len(args)}")

    # Check precondition if contract exists
    if func_contract:
      validate_contract(func_contract, func_params, args,
                        None, func_env, True, debug)

    # Bind arguments to parameters in function environment
    call_env = func_env
    for param_name, arg_val in zip(func_params, args):
      call_env = env_bind_value(call_env, param_name, arg_val)

    # Evaluate function body in call environment, passing context through
    result_val, _ = eval_ast(func_body, call_env, debug, context)

    # Check postcondition if contract exists
    if func_contract:
      validate_contract(func_contract, func_params, args,
                        result_val, func_env, False, debug)

    return result_val, env
  elif func_val['type'] == 'multi_clause_function':
    # Multi-clause function - try each clause in order until one matches
    func_name = func_val['name']
    clauses = func_val['clauses']
    func_env = func_val['closure_env']

    if debug:
      print(
          f"DEBUG: multi_clause_function '{func_name}' has {len(clauses)} clauses")
      for i, clause in enumerate(clauses):
        print(f"DEBUG:   Clause {i} patterns: {clause.get('patterns')}")

    # Try each clause in order
    for i, clause in enumerate(clauses):
      if debug:
        print(f"DEBUG: Trying clause {i}")
      # Try to match this clause's patterns and check guard
      bindings = try_match_clause(args, clause, func_env, debug, context)

      if bindings is not None:
        # This clause matches! Execute its body with the bindings
        clause_body = clause['body']
        clause_contract = clause.get('contract')

        # Check precondition if contract exists
        if clause_contract:
          # For multi-clause functions, we use the pattern variable names as params
          param_names = list(bindings.keys())
          param_values = [bindings[name] for name in param_names]
          validate_contract(clause_contract, param_names,
                            param_values, None, func_env, True, debug)

        # Bind pattern variables to their matched values
        call_env = func_env
        for var_name, var_value in bindings.items():
          call_env = env_bind_value(call_env, var_name, var_value)

        # Evaluate the clause body
        result_val, _ = eval_ast(clause_body, call_env, debug, context)

        # Check postcondition if contract exists
        if clause_contract:
          param_names = list(bindings.keys())
          param_values = [bindings[name] for name in param_names]
          validate_contract(clause_contract, param_names,
                            param_values, result_val, func_env, False, debug)

        return result_val, env

    # No clause matched - pattern match failure
    raise RuntimeError(
        f"No matching clause for function {func_name} with {len(args)} argument(s)")
  elif func_val['type'] == 'PartialFunction':
    # Partially applied function - fill in remaining arguments
    partial_data = func_val['value']

    # Handle different partial function types
    if partial_data.get('type') == 'partial_builtin':
      # Partial builtin function
      original_func = partial_data['original_func']
      fixed_args = partial_data['fixed_args']
      remaining_arity = partial_data['remaining_arity']

      # Combine fixed args with new args
      complete_args = fixed_args + args

      # Check if we have enough args now
      func_data = original_func['value']
      total_arity = func_data['arity']

      if len(complete_args) < total_arity:
        # Still not enough - create another partial function
        partial_data2 = {
            'type': 'partial_builtin',
            'original_func': original_func,
            'fixed_args': complete_args,
            'remaining_arity': total_arity - len(complete_args)
        }
        return make_value(partial_data2, "PartialFunction"), env
      elif len(complete_args) > total_arity:
        raise RuntimeError(
            f"Too many arguments for partial function: expected {total_arity}, got {len(complete_args)}")

      # Exact match - apply the function
      # Reconstruct the function call with complete args
      reconstructed_call = {
          'type': 'FUNCTION_CALL',
          'value': {
              'function': {'type': 'IDENTIFIER', 'value': func_data['name']},
              'args': []
          }
      }
      # Manually apply the builtin
      func_impl = func_data['func']
      needs_env = func_data.get('needs_env', False)
      needs_context_flag = func_data.get('needs_context', False)

      try:
        if needs_context_flag:
          if total_arity == 1:
            result = func_impl(complete_args[0], context)
          elif total_arity == 2:
            result = func_impl(complete_args[0], complete_args[1], context)
          elif total_arity == 3:
            result = func_impl(
                complete_args[0], complete_args[1], complete_args[2], context)
          else:
            raise RuntimeError(f"Unsupported arity: {total_arity}")
        elif needs_env:
          if total_arity == 1:
            result = func_impl(complete_args[0], env)
          elif total_arity == 2:
            result = func_impl(complete_args[0], complete_args[1], env)
          elif total_arity == 3:
            result = func_impl(
                complete_args[0], complete_args[1], complete_args[2], env)
          else:
            raise RuntimeError(f"Unsupported arity: {total_arity}")
        else:
          if total_arity == 1:
            result = func_impl(complete_args[0])
          elif total_arity == 2:
            result = func_impl(complete_args[0], complete_args[1])
          elif total_arity == 3:
            result = func_impl(
                complete_args[0], complete_args[1], complete_args[2])
          else:
            raise RuntimeError(f"Unsupported arity: {total_arity}")

        if isinstance(result, dict) and 'type' in result:
          return result, env
        else:
          return make_value(result, "Unknown"), env
      except Exception as e:
        raise RuntimeError(
            f"Error calling partial builtin {func_data['name']}: {e}")

    elif partial_data.get('type') == 'partial_user_func':
      # Partial user-defined function
      original_func = partial_data['original_func']
      fixed_args = partial_data['fixed_args']
      remaining_params = partial_data['remaining_params']

      # Combine fixed args with new args
      complete_args = fixed_args + args

      # Check if we have enough args now
      func_params = original_func['params']

      if len(complete_args) < len(func_params):
        # Still not enough - create another partial function
        partial_data2 = {
            'type': 'partial_user_func',
            'original_func': original_func,
            'fixed_args': complete_args,
            'remaining_params': func_params[len(complete_args):]
        }
        return make_value(partial_data2, "PartialFunction"), env
      elif len(complete_args) > len(func_params):
        raise RuntimeError(
            f"Too many arguments for partial function: expected {len(func_params)}, got {len(complete_args)}")

      # Exact match - apply the function
      func_body = original_func['body']
      func_env = original_func['closure_env']

      call_env = func_env
      for param_name, arg_val in zip(func_params, complete_args):
        call_env = env_bind_value(call_env, param_name, arg_val)

      result_val, _ = eval_ast(func_body, call_env, debug, context)
      return result_val, env

    elif 'placeholder_positions' in partial_data:
      # Placeholder-based partial functions
      original_func = partial_data['original_func']
      fixed_args = partial_data['fixed_args']
      placeholder_positions = partial_data['placeholder_positions']

      # Fill in the placeholder positions with the new arguments
      if len(args) != len(placeholder_positions):
        raise RuntimeError(
            f"Partial function expects {len(placeholder_positions)} arguments, got {len(args)}")

      # Create complete argument list
      complete_args = fixed_args.copy()
      for pos, arg in zip(placeholder_positions, args):
        complete_args[pos] = arg

      # Now apply the original function with complete arguments
      if original_func['type'] == 'BuiltinOperator':
        # Handle builtin operators (now using stdlib functions)
        op_name = original_func['value']
        if op_name in BUILTIN_OPERATORS:
          op_func = BUILTIN_OPERATORS[op_name]
          if len(complete_args) >= 2:
            # Operators now expect wrapped Dict values
            result = op_func(complete_args[0], complete_args[1])
            return result, env
          else:
            raise RuntimeError(
                f"Operator {op_name} requires 2 arguments, got {len(complete_args)}")
        else:
          raise RuntimeError(f"Unknown operator: {op_name}")
      elif original_func['type'] == 'BuiltinFunction':
        # Handle builtin functions
        func_data = original_func['value']
        func_impl = func_data['func']
        total_arity = func_data['arity']
        needs_env = func_data.get('needs_env', False)
        needs_context_flag = func_data.get('needs_context', False)

        if len(complete_args) != total_arity:
          raise RuntimeError(
              f"Function {func_data['name']} expects {total_arity} arguments, got {len(complete_args)}")

        try:
          if needs_context_flag:
            if total_arity == 1:
              result = func_impl(complete_args[0], context)
            elif total_arity == 2:
              result = func_impl(complete_args[0], complete_args[1], context)
            elif total_arity == 3:
              result = func_impl(
                  complete_args[0], complete_args[1], complete_args[2], context)
            else:
              raise RuntimeError(f"Unsupported arity: {total_arity}")
          elif needs_env:
            if total_arity == 1:
              result = func_impl(complete_args[0], env)
            elif total_arity == 2:
              result = func_impl(complete_args[0], complete_args[1], env)
            elif total_arity == 3:
              result = func_impl(
                  complete_args[0], complete_args[1], complete_args[2], env)
            else:
              raise RuntimeError(f"Unsupported arity: {total_arity}")
          else:
            if total_arity == 1:
              result = func_impl(complete_args[0])
            elif total_arity == 2:
              result = func_impl(complete_args[0], complete_args[1])
            elif total_arity == 3:
              result = func_impl(
                  complete_args[0], complete_args[1], complete_args[2])
            else:
              raise RuntimeError(f"Unsupported arity: {total_arity}")

          if isinstance(result, dict) and 'type' in result:
            return result, env
          else:
            return make_value(result, "Unknown"), env
        except Exception as e:
          raise RuntimeError(
              f"Error calling partial builtin {func_data['name']}: {e}")
      elif original_func['type'] == 'function':
        # Handle user-defined functions
        func_params = original_func['params']
        func_body = original_func['body']
        func_env = original_func['closure_env']

        if len(complete_args) != len(func_params):
          raise RuntimeError(
              f"Function expects {len(func_params)} arguments, got {len(complete_args)}")

        call_env = func_env
        for param_name, arg_val in zip(func_params, complete_args):
          call_env = env_bind_value(call_env, param_name, arg_val)

        result_val, _ = eval_ast(func_body, call_env, debug, context)
        return result_val, env
      else:
        raise RuntimeError(
            f"Cannot apply partial function to {original_func['type']}")
    else:
      raise RuntimeError(f"Unknown partial function format: {partial_data}")
  else:
    raise RuntimeError(f"Cannot call non-function value: {func_val}")


def eval_function_def(ast_node: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """Evaluate function definition and bind in environment with clause support"""
  value_dict = ast_node['value']
  func_name = value_dict['name']

  # Extract parameter patterns from raw_params if available, else from children
  param_patterns = []
  param_names = []  # For simple variable patterns

  # Try to use raw_params from semantics (preserves pattern information)
  raw_params = value_dict.get('raw_params')
  if raw_params:
    for param_tuple in raw_params:
      if isinstance(param_tuple, tuple) and len(param_tuple) >= 2:
        param_type, param_data = param_tuple[0], param_tuple[1]

        if param_type == "TYPED_PARAM" and isinstance(param_data, dict):
          # TYPED_PARAM: {'name': ('PATTERN_VAR', 'x'), 'type': ...}
          name_tuple = param_data.get('name')
          type_constraint = param_data.get('type')  # Extract type constraint

          if isinstance(name_tuple, tuple) and len(name_tuple) >= 2:
            pattern_type_str, pattern_value = name_tuple[0], name_tuple[1]

            # Extract type name from type constraint
            type_name = None
            if isinstance(type_constraint, tuple) and len(type_constraint) >= 2:
              if type_constraint[0] == 'TYPE_ATOM':
                type_name = type_constraint[1]
              elif type_constraint[0] == 'TYPE_SEQUENCE':
                # For List a, just use 'List'
                type_seq = type_constraint[1]
                if isinstance(type_seq, list) and len(type_seq) > 0:
                  first_type = type_seq[0]
                  if isinstance(first_type, tuple) and first_type[0] == 'TYPE_ATOM':
                    type_name = first_type[1]

            if debug:
              print(
                  f"DEBUG: Extracted type_name={type_name} from constraint={type_constraint}")

            pattern = {
                'type': pattern_type_str,
                'value': pattern_value
            }

            # Add type constraint for multiple dispatch
            if type_name:
              pattern['type_constraint'] = type_name

            if debug:
              print(f"DEBUG: Created pattern={pattern}")

            param_patterns.append(pattern)
            if pattern_type_str == 'PATTERN_VAR':
              param_names.append(pattern_value)
        elif param_type == "UNTYPED_PARAM":
          # UNTYPED_PARAM: param_data is the pattern directly
          # Could be ('PATTERN_VAR', 'x'), ('PATTERN_LIST', []), etc.
          if isinstance(param_data, tuple) and len(param_data) >= 2:
            pattern_type_str, pattern_value = param_data[0], param_data[1]
            param_patterns.append({
                'type': pattern_type_str,
                'value': pattern_value
            })
            if pattern_type_str == 'PATTERN_VAR':
              param_names.append(pattern_value)
  else:
    # Fallback: Extract from AST node children (old method)
    for child in ast_node.get('children', []):
      if child['type'] in ('TYPED_PARAM', 'UNTYPED_PARAM'):
        param_tuple = child.get('value')

        # Handle both tuple format (UNTYPED_PARAM) and dict format (TYPED_PARAM)
        if isinstance(param_tuple, dict):
          # TYPED_PARAM: {'name': ('PATTERN_VAR', 'x'), 'type': ...}
          name_tuple = param_tuple.get('name')
          if isinstance(name_tuple, tuple) and len(name_tuple) >= 2:
            pattern_type = name_tuple[0]
            pattern_value = name_tuple[1]

            if pattern_type == 'PATTERN_VAR':
              param_patterns.append({
                  'type': 'PATTERN_VAR',
                  'value': pattern_value
              })
              param_names.append(pattern_value)
            elif pattern_type == 'PATTERN_LITERAL':
              param_patterns.append({
                  'type': 'PATTERN_LITERAL',
                  'value': pattern_value
              })
            elif pattern_type == 'PATTERN_CONSTRUCTOR':
              param_patterns.append({
                  'type': 'PATTERN_CONSTRUCTOR',
                  'value': pattern_value
              })
            elif pattern_type == 'PARTIAL_PLACEHOLDER':
              param_patterns.append({
                  'type': 'PARTIAL_PLACEHOLDER',
                  'value': '_'
              })
        elif isinstance(param_tuple, tuple) and len(param_tuple) >= 2:
          # UNTYPED_PARAM: ('PATTERN_VAR', 'n') or ('PATTERN_LITERAL', ...)
          pattern_type = param_tuple[0]
          pattern_value = param_tuple[1]

          # Create a pattern AST node
          if pattern_type == 'PATTERN_VAR':
            # Variable pattern - binds to variable
            param_patterns.append({
                'type': 'PATTERN_VAR',
                'value': pattern_value
            })
            param_names.append(pattern_value)
          elif pattern_type == 'PATTERN_LITERAL':
            # Literal pattern - matches exact value
            param_patterns.append({
                'type': 'PATTERN_LITERAL',
                'value': pattern_value
            })
          elif pattern_type == 'PATTERN_CONSTRUCTOR':
            # Constructor pattern - matches constructor and extracts fields
            param_patterns.append({
                'type': 'PATTERN_CONSTRUCTOR',
                'value': pattern_value
            })
          elif pattern_type == 'PARTIAL_PLACEHOLDER':
            # Wildcard pattern
            param_patterns.append({
                'type': 'PARTIAL_PLACEHOLDER',
                'value': '_'
            })

  func_body = value_dict['body']
  guard_ast = value_dict.get('guard')
  contract_ast = value_dict.get('contract')

  # Check if function with this name already exists
  existing_func = env_lookup_value(env, func_name)

  if existing_func and param_patterns:  # Has patterns, so could be multi-clause
    # Check if existing function is single-clause or multi-clause
    if existing_func.get('type') == 'function':
      # Convert single function to multi-clause
      first_clause = make_function_clause(
          [{'type': 'PATTERN_VAR', 'value': p}
           for p in existing_func['params']],
          existing_func['body'],
          None,  # No guard in old-style function
          existing_func.get('contract')
      )
      new_clause = make_function_clause(
          param_patterns, func_body, guard_ast, contract_ast)
      func_value = make_multi_clause_function(
          func_name,
          [first_clause, new_clause],
          env
      )
    elif existing_func.get('type') == 'multi_clause_function':
      # Add new clause to existing multi-clause function
      new_clause = make_function_clause(
          param_patterns, func_body, guard_ast, contract_ast)
      existing_clauses = existing_func['clauses']

      func_value = make_multi_clause_function(
          func_name,
          existing_clauses + [new_clause],
          env
      )
    else:
      # Not a function, create new single-clause function
      if param_names:
        func_value = make_function(param_names, func_body, env, contract_ast)
      else:
        # Has only non-variable patterns, use multi-clause from start
        clause = make_function_clause(
            param_patterns, func_body, guard_ast, contract_ast)
        func_value = make_multi_clause_function(func_name, [clause], env)
  else:
    # No existing function, or no patterns - create simple function
    # Check if any pattern has type constraints (for multiple dispatch)
    has_type_constraints = any(p.get('type_constraint')
                               for p in param_patterns)

    if param_names and not any(p['type'] != 'PATTERN_VAR' for p in param_patterns) and not guard_ast and not has_type_constraints:
      # All simple variable patterns, no guard, no type constraints - use simple function
      func_value = make_function(param_names, func_body, env, contract_ast)
    elif param_patterns or guard_ast or has_type_constraints:
      # Has non-variable patterns, guard, or type constraints - use multi-clause
      clause = make_function_clause(
          param_patterns, func_body, guard_ast, contract_ast)
      func_value = make_multi_clause_function(func_name, [clause], env)
    else:
      # No patterns at all (0-arity function)
      func_value = make_function([], func_body, env, contract_ast)

  # Bind function in environment
  new_env = env_bind_value(env, func_name, func_value)

  # Update closure_env to point to the new environment (for recursion)
  # This allows recursive calls to see the updated function with all clauses
  if func_value.get('type') == 'multi_clause_function':
    func_value['closure_env'] = new_env
  elif func_value.get('type') == 'function':
    func_value['closure_env'] = new_env

  return make_value(func_name, "Function"), new_env


def eval_value_binding(ast_node: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """Evaluate value binding and bind in environment"""
  if context is None:
    context = make_execution_context()

  value_dict = ast_node['value']
  var_name = value_dict['name']
  value_ast = value_dict['value']

  # Evaluate the value
  value_result, env = eval_ast(value_ast, env, debug, context)

  # Bind in environment
  new_env = env_bind_value(env, var_name, value_result)

  return value_result, new_env


def eval_list(ast_node: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """Evaluate list expression"""
  if context is None:
    context = make_execution_context()

  elements = []
  for child in ast_node.get('children', []):
    elem_val, env = eval_ast(child, env, debug, context)
    elements.append(elem_val)  # Keep wrapped Dict, not raw value

  return make_value(elements, "List"), env


def eval_record(ast_node: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """Evaluate record expression
  Records are dictionaries: { key = value, key = value }
  """
  if context is None:
    context = make_execution_context()

  record_dict = {}
  fields = ast_node.get('value', {})

  # The fields should be a dict mapping keys to AST nodes
  for key, value_ast in fields.items():
    # Evaluate the value expression
    value_val, env = eval_ast(value_ast, env, debug, context)
    record_dict[key] = value_val

  return make_value(record_dict, "Record"), env


def eval_lambda(ast_node: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """Evaluate lambda expression"""
  value_dict = ast_node['value']
  params = value_dict.get('params', [])
  body = value_dict['body']

  # Extract parameter names
  param_names = []
  for param in params:
    # Handle nested tuples: ('PATTERN_VAR', ('PATTERN_VAR', 'x'))
    if isinstance(param, tuple) and param[0] == 'PATTERN_VAR':
      inner = param[1]
      if isinstance(inner, tuple) and inner[0] == 'PATTERN_VAR':
        param_names.append(inner[1])
      elif isinstance(inner, str):
        param_names.append(inner)
    elif isinstance(param, str):
      param_names.append(param)

  # Create function value with closure
  func_value = make_function(param_names, body, env)

  return func_value, env


def eval_case(ast_node: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """Evaluate case expression with pattern matching"""
  if context is None:
    context = make_execution_context()

  value_dict = ast_node['value']
  scrutinee_ast = value_dict['scrutinee']
  branches = value_dict.get('branches', [])

  # Evaluate scrutinee
  scrutinee_val, env = eval_ast(scrutinee_ast, env, debug, context)

  # Try each branch
  for branch in branches:
    if branch['type'] == 'CASE_BRANCH':
      pattern_ast = branch['value']['pattern']
      body_ast = branch['value']['body']

      # Try to match pattern and extract bindings
      match_result = matches_pattern(scrutinee_val, pattern_ast, env)
      if match_result is not None:
        # Pattern matched - create new environment with pattern bindings
        pattern_bindings = match_result if isinstance(
            match_result, dict) else {}
        branch_env = env
        for var_name, var_val in pattern_bindings.items():
          branch_env = env_bind_value(branch_env, var_name, var_val)

        # Evaluate branch body in extended environment
        result_val, _ = eval_ast(body_ast, branch_env, debug, context)
        return result_val, env

  raise RuntimeError(f"No matching case branch for value: {scrutinee_val}")


def matches_pattern(value: Dict, pattern_ast: Dict, env: Dict):
  """Pattern matching with constructor patterns and variable extraction

  Returns:
      None if pattern doesn't match
      {} (empty dict) if pattern matches but extracts no variables
      dict of {var_name: var_value} if pattern matches and extracts variables
  """
  if pattern_ast['type'] == 'PATTERN_LITERAL':
    # Literal pattern - exact match
    literal_val = pattern_ast['value']
    if isinstance(literal_val, tuple):
      literal_val = literal_val[1]
    return {} if value['value'] == literal_val else None

  elif pattern_ast['type'] == 'PARTIAL_PLACEHOLDER':
    # Wildcard pattern - always matches, no bindings
    return {}

  elif pattern_ast['type'] == 'PATTERN_VAR' or pattern_ast['type'] == 'IDENTIFIER':
    # Variable pattern - always matches and binds the variable
    var_name = pattern_ast['value']
    return {var_name: value}

  elif pattern_ast['type'] == 'PATTERN_CONS':
    # List cons pattern (x :: xs) - matches non-empty list
    pattern_data = pattern_ast['value']
    head_pattern = pattern_data['head']
    tail_pattern = pattern_data['tail']

    # Check if value is a list
    if not isinstance(value, dict) or value.get('type') != 'List':
      return None

    lst = value.get('value', [])
    if not isinstance(lst, list) or len(lst) == 0:
      return None

    # Extract head and tail
    head_val = lst[0]
    tail_val = make_value(lst[1:], "List")

    # Match patterns recursively
    all_bindings = {}

    # Convert tuple patterns to AST format for recursive matching
    def tuple_to_pattern(pat):
      if isinstance(pat, tuple):
        pat_type, pat_value = pat
        return {'type': pat_type, 'value': pat_value}
      return pat

    head_pattern_ast = tuple_to_pattern(head_pattern)
    tail_pattern_ast = tuple_to_pattern(tail_pattern)

    # Match head pattern
    head_result = matches_pattern(head_val, head_pattern_ast, env)
    if head_result is None:
      return None
    all_bindings.update(head_result)

    # Match tail pattern
    tail_result = matches_pattern(tail_val, tail_pattern_ast, env)
    if tail_result is None:
      return None
    all_bindings.update(tail_result)

    return all_bindings

  elif pattern_ast['type'] == 'PATTERN_LIST':
    # Empty list pattern [] - matches empty list only
    # Check if value is an empty list
    if not isinstance(value, dict) or value.get('type') != 'List':
      return None

    lst = value.get('value', [])
    if not isinstance(lst, list) or len(lst) != 0:
      return None

    return {}  # Empty list matches, no bindings

  elif pattern_ast['type'] == 'PATTERN_CONSTRUCTOR':
    # Constructor pattern - match constructor and extract fields
    pattern_data = pattern_ast['value']
    pattern_ctor_name = pattern_data['name']
    pattern_args = pattern_data.get('args', [])

    # Special case: cons pattern matches non-empty lists
    if pattern_ctor_name == 'cons':
      # cons x xs matches a non-empty list, binding head to x and tail to xs
      if not isinstance(value, dict) or value.get('type') != 'List':
        return None

      lst = value.get('value', [])
      if not isinstance(lst, list) or len(lst) == 0:
        return None

      # cons should have exactly 2 args: head and tail
      if len(pattern_args) != 2:
        return None

      head_pattern = pattern_args[0]
      tail_pattern = pattern_args[1]

      # Extract head and tail
      head_val = lst[0]
      tail_val = make_value(lst[1:], "List")

      # Match patterns recursively
      all_bindings = {}

      # Match head pattern
      if isinstance(head_pattern, tuple) and head_pattern[0] == 'PATTERN_VAR':
        var_name = head_pattern[1]
        all_bindings[var_name] = head_val
      elif isinstance(head_pattern, tuple) and head_pattern[0] == 'PARTIAL_PLACEHOLDER':
        # Wildcard - no binding
        pass
      else:
        # Try recursive match
        head_result = matches_pattern(head_val, {'type': head_pattern[0], 'value': head_pattern[1]} if isinstance(
            head_pattern, tuple) else head_pattern, env)
        if head_result is None:
          return None
        all_bindings.update(head_result)

      # Match tail pattern
      if isinstance(tail_pattern, tuple) and tail_pattern[0] == 'PATTERN_VAR':
        var_name = tail_pattern[1]
        all_bindings[var_name] = tail_val
      elif isinstance(tail_pattern, tuple) and tail_pattern[0] == 'PARTIAL_PLACEHOLDER':
        # Wildcard - no binding
        pass
      else:
        # Try recursive match
        tail_result = matches_pattern(tail_val, {'type': tail_pattern[0], 'value': tail_pattern[1]} if isinstance(
            tail_pattern, tuple) else tail_pattern, env)
        if tail_result is None:
          return None
        all_bindings.update(tail_result)

      return all_bindings

    # Check if value is a constructor instance
    # Constructor values are wrapped: {'value': {'constructor': ..., 'fields': ...}, 'type': ...}
    if not isinstance(value, dict):
      return None

    # Get the inner constructor data
    value_data = value.get('value', {})
    if not isinstance(value_data, dict) or 'constructor' not in value_data:
      return None

    value_ctor_name = value_data['constructor']
    value_fields = value_data.get('fields', [])

    # Check constructor names match
    if pattern_ctor_name != value_ctor_name:
      return None

    # Check arity matches
    if len(pattern_args) != len(value_fields):
      return None

    # Match each argument pattern and collect bindings
    all_bindings = {}
    for pattern_arg, field_value in zip(pattern_args, value_fields):
      if isinstance(pattern_arg, tuple) and pattern_arg[0] == 'PATTERN_VAR':
        # Simple variable pattern in constructor
        var_name = pattern_arg[1]
        # Wrap field value if it's not already wrapped
        if isinstance(field_value, dict) and 'type' in field_value:
          all_bindings[var_name] = field_value
        else:
          # Infer type from value
          if isinstance(field_value, str):
            all_bindings[var_name] = make_value(field_value, "String")
          elif isinstance(field_value, (int, float)):
            all_bindings[var_name] = make_value(field_value, "Num")
          elif isinstance(field_value, bool):
            all_bindings[var_name] = make_value(field_value, "Bool")
          elif isinstance(field_value, list):
            all_bindings[var_name] = make_value(field_value, "List")
          else:
            all_bindings[var_name] = make_value(field_value, "Unknown")
      elif isinstance(pattern_arg, tuple) and pattern_arg[0] == 'PARTIAL_PLACEHOLDER':
        # Wildcard in constructor - skip this field
        continue
      else:
        # Nested pattern - recursively match (not implemented yet)
        # For now, treat as wildcard
        continue

    return all_bindings

  else:
    # Unknown pattern type - default to exact value match
    return {} if value.get('value') == pattern_ast.get('value') else None


def eval_operation(ast_node: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """Evaluate binary operation"""
  if context is None:
    context = make_execution_context()

  value_dict = ast_node['value']
  left_ast = value_dict['left']
  right_ast = value_dict['right']
  op = value_dict['op']

  # Evaluate operands
  left_val, env = eval_ast(left_ast, env, debug, context)
  right_val, env = eval_ast(right_ast, env, debug, context)

  # Apply operation (now using stdlib functions that expect Dict values)
  if op in BUILTIN_OPERATORS:
    op_func = BUILTIN_OPERATORS[op]
    result = op_func(left_val, right_val)
    return result, env
  else:
    raise RuntimeError(f"Unknown operation: {op}")


def eval_pipeline(ast_node: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """Evaluate pipeline expression (input |> function)"""
  if context is None:
    context = make_execution_context()

  value_dict = ast_node['value']
  input_ast = value_dict['input']
  func_ast = value_dict['function']

  # Evaluate input
  input_val, env = eval_ast(input_ast, env, debug, context)

  # Evaluate function
  func_val, env = eval_ast(func_ast, env, debug, context)

  # Apply function to input
  if func_val['type'] == 'function':
    func_params = func_val['params']
    func_body = func_val['body']
    func_env = func_val['closure_env']

    # Bind input to first parameter
    if len(func_params) >= 1:
      call_env = env_bind_value(func_env, func_params[0], input_val)
      result_val, _ = eval_ast(func_body, call_env, debug, context)
      return result_val, env
    else:
      raise RuntimeError("Pipeline function requires at least 1 parameter")
  else:
    raise RuntimeError(f"Pipeline requires function, got: {func_val['type']}")


def eval_type_def(ast_node: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """Evaluate type definition and create constructor functions"""
  value_dict = ast_node['value']
  type_name = value_dict['name']
  constructors = value_dict.get('constructors', [])

  # For each constructor, create a function that builds the data structure
  new_env = env

  for constructor in constructors:
    if isinstance(constructor, tuple) and len(constructor) >= 2:
      ctor_type, ctor_data = constructor[0], constructor[1]
      if ctor_type == 'TYPE_CONSTRUCTOR' and isinstance(ctor_data, dict):
        ctor_name_data = ctor_data.get('name')
        ctor_args = ctor_data.get('args', [])

        # Extract constructor name
        if isinstance(ctor_name_data, tuple) and ctor_name_data[0] == 'IDENTIFIER':
          ctor_name = ctor_name_data[1]
        else:
          continue

        # Create a constructor function that returns a tagged tuple
        # Constructor takes N arguments and returns (TypeName, ConstructorName, args...)
        param_names = [f"arg{i}" for i in range(len(ctor_args))]

        # Create a special constructor function
        ctor_func = {
            'type': 'constructor',
            'type_name': type_name,
            'ctor_name': ctor_name,
            'arity': len(ctor_args)
        }

        new_env = env_bind_value(new_env, ctor_name, ctor_func)

  return make_value(None, "Unit"), new_env


def eval_inline_eval(ast_node: Dict, env: Dict, debug: bool = False, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
  """Evaluate inline evaluation statement (> expr.) and print the result"""
  # The value contains the expression AST
  expr_ast = ast_node['value']

  # Evaluate the expression
  result_val, _ = eval_ast(expr_ast, env, debug, context)

  # Print the result
  val_str = str(result_val.get('value', result_val))
  type_str = result_val.get('type', 'Unknown')
  print(f"{val_str}")

  # Return the result value and unchanged environment (inline eval doesn't bind)
  return result_val, env


# ============================================================================
# PROGRAM EVALUATION
# ============================================================================

def eval_program(ast_nodes: List[Dict], debug: bool = False) -> Tuple[Dict, List[Tuple[str, Any]]]:
  """
  Evaluate a program (list of AST nodes) and return final environment and bindings.
  Returns (final_env, list of (name, value) pairs)
  """
  env = create_builtin_runtime_env()
  results = []

  for ast_node in ast_nodes:
    try:
      value, env = eval_ast(ast_node, env, debug)

      # Track top-level bindings
      if ast_node['type'] == 'VALUE_BINDING':
        var_name = ast_node['value']['name']
        # Use unwrap_value to recursively unwrap nested structures
        unwrapped = unwrap_value(value)
        results.append((var_name, unwrapped))
      elif ast_node['type'] == 'FUNCTION_DEF':
        func_name = ast_node['value']['name']
        results.append((func_name, f"<function {func_name}>"))
    except Exception as e:
      raise RuntimeError(f"Runtime error: {e}") from e

  return env, results


# ============================================================================
# FACTORY FUNCTIONS (for compatibility with main.py)
# ============================================================================

def create_interpreter(debug: bool = False):
  """Factory function returning an interpreter"""
  def interpreter(ast_nodes):
    try:
      env, results = eval_program(ast_nodes, debug)
      return results
    except RuntimeError as e:
      raise REI1RuntimeError(str(e))

  def interpret_program_func(ast_nodes):
    try:
      # Unwrap ASTNode objects if they're wrapped
      unwrapped = []
      for node in ast_nodes:
        if hasattr(node, 'type') and hasattr(node, 'value'):
          # It's a wrapped object, convert to dict preserving all attributes
          node_dict = {
              'type': node.type,
              'value': node.value
          }
          # Preserve other attributes if they exist
          if hasattr(node, 'children'):
            node_dict['children'] = node.children
          if hasattr(node, 'span'):
            node_dict['span'] = node.span
          if hasattr(node, 'type_info'):
            node_dict['type_info'] = node.type_info
          if hasattr(node, 'resolved_names'):
            node_dict['resolved_names'] = node.resolved_names
          unwrapped.append(node_dict)
        else:
          # It's already a dict
          unwrapped.append(node)

      env, results = eval_program(unwrapped, debug)
      # Return dict of bindings for compatibility
      return {name: val for name, val in results}
    except RuntimeError as e:
      raise REI1RuntimeError(str(e))

  def eval_node_wrapper(ast_node, env_obj):
    """Wrap eval_ast result in object-like structure for main.py compatibility"""
    # Extract actual dict from wrapper if needed
    actual_env = env_obj._dict if hasattr(
        env_obj, '_dict') else create_builtin_runtime_env()
    # Convert ast_node back to dict if it's a wrapper
    actual_ast = {'type': ast_node.type, 'value': ast_node.value} if hasattr(
        ast_node, 'type') else ast_node

    val, new_env = eval_ast(actual_ast, actual_env, debug)
    # Create mock object with .value and .type_info.name attributes
    return type('Value', (), {
        'value': val['value'],
        'type_info': type('TypeInfo', (), {'name': val['type']})()
    })()

  # Create environment wrapper that exposes bindings as attribute
  builtin_env = create_builtin_runtime_env()
  env_wrapper = type('Environment', (), {
      'bindings': builtin_env['bindings'],
      '_dict': builtin_env
  })()

  return type('Interpreter', (), {
      'interpret': lambda self, ast_nodes: interpreter(ast_nodes),
      'interpret_program': lambda self, ast_nodes: interpret_program_func(ast_nodes),
      'environment': {},
      'global_env': env_wrapper,
      '_eval_node': lambda self, ast_node, env: eval_node_wrapper(ast_node, env)
  })()


def create_debug_interpreter():
  """Factory function returning a debug interpreter"""
  return create_interpreter(debug=True)
