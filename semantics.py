"""
REI1 Semantics Analysis - Pure Functional Style
No classes, only pure functions and immutable data structures
"""

from typing import Any, Dict, List, Optional, Tuple
from parsing import CSTNode, SourceSpan
from utilities import extract_from_wrapper, unpack_typed_tuple, safe_dict_get


# ============================================================================
# DATA STRUCTURES (Immutable Dictionaries)
# ============================================================================

def make_type_info(name: str, parameters: Optional[List[Dict]] = None, constraints: Optional[Dict] = None) -> Dict:
  """Create an immutable type info dictionary"""
  return {
      'name': name,
      'parameters': parameters or [],
      'constraints': constraints or {}
  }


def make_ast_node(node_type: str, value: Any, children: Optional[List[Dict]] = None,
                  span: Optional[SourceSpan] = None, type_info: Optional[Dict] = None,
                  resolved_names: Optional[Dict[str, str]] = None) -> Dict:
  """Create an immutable AST node dictionary"""
  return {
      'type': node_type,
      'value': value,
      'children': children or [],
      'span': span,
      'type_info': type_info,
      'resolved_names': resolved_names or {}
  }


def make_environment(parent: Optional[Dict] = None, bindings: Optional[Dict] = None,
                     types: Optional[Dict] = None, modules: Optional[Dict] = None) -> Dict:
  """Create an immutable environment dictionary"""
  return {
      'parent': parent,
      'bindings': bindings or {},
      'types': types or {},
      'modules': modules or {}
  }


# ============================================================================
# ENVIRONMENT OPERATIONS (Pure Functions)
# ============================================================================

def env_bind(env: Dict, name: str, type_info: Dict) -> Dict:
  """Return new environment with name bound to type info"""
  return {
      **env,
      'bindings': {**env['bindings'], name: type_info}
  }


def env_bind_type(env: Dict, name: str, type_info: Dict) -> Dict:
  """Return new environment with type name bound"""
  return {
      **env,
      'types': {**env['types'], name: type_info}
  }


def env_lookup(env: Dict, name: str) -> Optional[Dict]:
  """Look up a name in the environment chain"""
  if name in env['bindings']:
    return env['bindings'][name]
  elif env['parent']:
    return env_lookup(env['parent'], name)
  return None


def env_lookup_type(env: Dict, name: str) -> Optional[Dict]:
  """Look up a type name in the environment chain"""
  if name in env['types']:
    return env['types'][name]
  elif env['parent']:
    return env_lookup_type(env['parent'], name)
  return None


# ============================================================================
# BUILT-IN SETUP
# ============================================================================

def create_builtin_env() -> Dict:
  """Create the global environment with built-in types and functions"""
  env = make_environment()

  # Built-in types
  num_type = make_type_info("Num")
  char_type = make_type_info("Char")
  bool_type = make_type_info("Bool")

  env = env_bind_type(env, "Num", num_type)
  env = env_bind_type(env, "Char", char_type)
  env = env_bind_type(env, "Bool", bool_type)

  # String is List Char
  list_type = make_type_info("List", [make_type_info("a")])
  string_type = make_type_info("List", [char_type])
  env = env_bind_type(env, "List", list_type)
  env = env_bind_type(env, "String", string_type)

  # Built-in functions
  func_type = make_type_info("Function")
  for op in ["+", "-", "*", "/", "==", "!=", "<", ">", "<=", ">="]:
    env = env_bind(env, op, func_type)

  return env


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_identifier_name(identifier_data) -> Optional[str]:
  """Extract identifier name from various representations"""
  return extract_from_wrapper(identifier_data, "IDENTIFIER")


def extract_pattern_name(pattern_data) -> Optional[str]:
  """Extract name from pattern data"""
  return extract_from_wrapper(pattern_data, "PATTERN_VAR")


# ============================================================================
# TYPE ANALYSIS
# ============================================================================

def analyze_type_expr(type_data) -> Optional[Dict]:
  """Analyze type expression and return type info"""
  if isinstance(type_data, tuple) and len(type_data) >= 2:
    type_kind, type_value = type_data[0], type_data[1]

    if type_kind == "TYPE_ATOM":
      return make_type_info(type_value)
    elif type_kind == "RETURN_TYPE" and isinstance(type_value, dict):
      return analyze_type_expr(type_value.get('type'))

  return None


# ============================================================================
# EXPRESSION ANALYSIS
# ============================================================================

def analyze_expression(expr, env: Dict, debug: bool = False) -> Dict:
  """Analyze any expression (tuple or CST node) and return AST node"""
  if isinstance(expr, CSTNode):
    return analyze_cst_node(expr, env, debug)
  elif isinstance(expr, tuple) and len(expr) >= 1:
    return analyze_tuple_expr(expr, env, debug)
  else:
    raise ValueError(f"Unable to analyze expression: {expr}")


def analyze_tuple_expr(expr: Tuple, env: Dict, debug: bool = False) -> Dict:
  """Analyze tuple-based expressions from parser"""
  expr_type = expr[0]
  expr_data = expr[1] if len(expr) > 1 else None

  handlers = {
      "IDENTIFIER": lambda: make_ast_node("IDENTIFIER", expr_data or "", [], None,
                                          env_lookup(env, expr_data or "") or make_type_info("Unknown")),
      "NUMBER": lambda: make_ast_node("NUMBER", expr_data, [], None, make_type_info("Num")),
      "STRING": lambda: make_ast_node("STRING", expr_data, [], None, make_type_info("String")),
      "FUNCTION_CALL": lambda: analyze_function_call_tuple(expr_data, env, debug),
      "FUNCTION_NAME": lambda: analyze_function_name_tuple(expr_data, env, debug),
      "PARENTHESIZED": lambda: analyze_expression(expr_data, env, debug),
      "OPERATION": lambda: analyze_operation(expr_data, env, debug),
      "LAMBDA": lambda: analyze_lambda_tuple(expr_data, env, debug),
      "CASE": lambda: analyze_case_tuple(expr_data, env, debug),
      "LIST": lambda: analyze_list_tuple(expr_data, env, debug),
      "RECORD": lambda: analyze_record_tuple(expr_data, env, debug),
      "CONTRACT": lambda: analyze_contract(expr, env, None, debug),
      "PIPELINE": lambda: analyze_pipeline_tuple(expr_data, env, debug),
      "DOLLAR_APP": lambda: analyze_dollar_app_tuple(expr_data, env, debug),
  }

  if expr_type in handlers:
    return handlers[expr_type]()
  else:
    return make_ast_node(expr_type, expr_data, [], None, make_type_info("Unknown"))


def analyze_function_call_tuple(call_data, env: Dict, debug: bool = False) -> Dict:
  """Analyze function call from tuple representation"""
  # Handle dict format from contract parser: {'name': ..., 'args': [...]}
  if isinstance(call_data, dict):
    if 'name' in call_data and 'args' in call_data:
      func_ast = analyze_expression(call_data['name'], env, debug)
      arg_asts = [analyze_expression(arg, env, debug)
                  for arg in call_data['args']]
    else:
      raise ValueError(f"Invalid function call dict format: {call_data}")
  # Handle list format: [func, arg1, arg2, ...]
  elif isinstance(call_data, list) and len(call_data) >= 1:
    func_ast = analyze_expression(call_data[0], env, debug)
    arg_asts = [analyze_expression(arg, env, debug) for arg in call_data[1:]]
  else:
    raise ValueError(f"Invalid function call data: {call_data}")

  return make_ast_node(
      "FUNCTION_CALL",
      {"function": func_ast, "args": arg_asts},
      [func_ast] + arg_asts,
      None,
      make_type_info("Unknown")
  )


def analyze_function_name_tuple(name_data, env: Dict, debug: bool = False) -> Dict:
  """Analyze function name from tuple representation"""
  name_str = extract_from_wrapper(name_data, "IDENTIFIER", default="") or ""

  type_info = env_lookup(env, name_str) or make_type_info("Function")
  return make_ast_node("FUNCTION_NAME", name_str, [], None, type_info)


def analyze_function_name(cst_node: CSTNode, env: Dict, debug: bool = False) -> Dict:
  """Analyze function name from CST node"""
  # Extract name from tuple format ('IDENTIFIER', 'name') or use directly
  name_str = extract_from_wrapper(cst_node.value, "IDENTIFIER", default="") or ""

  type_info = env_lookup(env, name_str) or make_type_info("Function")
  return make_ast_node("FUNCTION_NAME", name_str, [], cst_node.span, type_info)


def analyze_operation(operation_data, env: Dict, debug: bool = False) -> Dict:
  """Analyze binary operation (e.g., y != 0, x + 1)"""
  if not isinstance(operation_data, dict):
    raise ValueError(f"Invalid operation data: {operation_data}")

  left_ast = analyze_expression(operation_data.get('left'), env, debug)
  right_ast = analyze_expression(operation_data.get('right'), env, debug)
  op = operation_data.get('op')

  # Determine result type
  if op in ['==', '!=', '<', '>', '<=', '>=']:
    result_type = make_type_info("Bool")
  elif op in ['+', '-', '*', '/']:
    result_type = make_type_info("Num")
  else:
    result_type = make_type_info("Unknown")

  return make_ast_node(
      "OPERATION",
      {"left": left_ast, "op": op, "right": right_ast},
      [left_ast, right_ast],
      None,
      result_type
  )


def analyze_lambda_tuple(lambda_data, env: Dict, debug: bool = False) -> Dict:
  """Analyze lambda expression"""
  if not isinstance(lambda_data, dict):
    raise ValueError(f"Invalid lambda data: {lambda_data}")

  params = lambda_data.get("params", [])
  body = lambda_data.get("body")

  if body is None:
    raise ValueError(f"Lambda expression missing body: {lambda_data}")

  # Create new environment for lambda parameters
  lambda_env = env
  for param in params:
    if isinstance(param, tuple) and len(param) == 2 and param[0] == "PATTERN_VAR":
      param_name = param[1]
      lambda_env = env_bind(lambda_env, param_name, make_type_info("Unknown"))

  body_ast = analyze_expression(body, lambda_env, debug)

  return make_ast_node(
      "LAMBDA",
      {"params": params, "body": body_ast},
      [body_ast],
      None,
      make_type_info("Function")
  )


def analyze_case_tuple(case_data, env: Dict, debug: bool = False) -> Dict:
  """Analyze case expression"""
  if not isinstance(case_data, dict):
    raise ValueError(f"Invalid case data: {case_data}")

  # Handle both 'scrutinee' and 'expr' keys
  scrutinee = case_data.get('scrutinee') or case_data.get('expr')
  branches = case_data.get('branches') or case_data.get('arms', [])

  if scrutinee is None:
    raise ValueError(f"Case expression missing scrutinee: {case_data}")

  scrutinee_ast = analyze_expression(scrutinee, env, debug)

  # Analyze branches
  branch_asts = []
  for branch in branches:
    if isinstance(branch, tuple) and len(branch) >= 2:
      branch_type, branch_value = branch[0], branch[1]
      if branch_type in ["CASE_BRANCH", "CASE_ARM"] and isinstance(branch_value, dict):
        pattern = branch_value.get('pattern')
        body = branch_value.get('body')

        # Create branch environment (TODO: extract pattern variables)
        branch_env = env

        pattern_ast = analyze_expression(
            pattern, branch_env, debug) if pattern else None
        body_ast = analyze_expression(
            body, branch_env, debug) if body else None

        branch_asts.append(make_ast_node(
            "CASE_BRANCH",
            {"pattern": pattern_ast, "body": body_ast},
            [ast for ast in [pattern_ast, body_ast] if ast],
            None,
            body_ast['type_info'] if body_ast else make_type_info("Unknown")
        ))

  result_type = branch_asts[0]['type_info'] if branch_asts else make_type_info(
      "Unknown")

  return make_ast_node(
      "CASE",
      {"scrutinee": scrutinee_ast, "branches": branch_asts},
      [scrutinee_ast] + branch_asts,
      None,
      result_type
  )


def analyze_list_tuple(list_data, env: Dict, debug: bool = False) -> Dict:
  """Analyze list expression"""
  if not isinstance(list_data, list):
    raise ValueError(f"Invalid list data: {list_data}")

  element_asts = [analyze_expression(elem, env, debug) for elem in list_data]
  elem_type = element_asts[0]['type_info'] if element_asts else make_type_info(
      "Unknown")
  list_type = make_type_info("List", [elem_type])

  return make_ast_node(
      "LIST",
      element_asts,
      element_asts,
      None,
      list_type
  )


def analyze_record_tuple(record_data, env: Dict, debug: bool = False) -> Dict:
  """Analyze record expression from tuple format
  record_data is a list of RECORD_FIELD tuples
  """
  if not isinstance(record_data, list):
    raise ValueError(f"Invalid record data: {record_data}")

  fields = {}
  analyzed_fields = []

  for field_tuple in record_data:
    if isinstance(field_tuple, tuple) and field_tuple[0] == "RECORD_FIELD":
      field_data = field_tuple[1]
      key_raw = field_data["key"]

      # Extract string from key (handle both str and tuple)
      if isinstance(key_raw, str):
        key = key_raw
      elif isinstance(key_raw, tuple):
        key = key_raw[1] if len(key_raw) > 1 else str(key_raw)
      else:
        key = str(key_raw)

      value_expr = field_data["value"]

      # Analyze the value expression
      analyzed_value = analyze_expression(value_expr, env, debug)

      fields[key] = analyzed_value
      analyzed_fields.append({
          "key": key,
          "value": analyzed_value
      })

  return make_ast_node(
      "RECORD",
      fields,
      analyzed_fields,
      None,
      make_type_info("Record")
  )


def analyze_contract(contract_data, env: Dict, return_type: Optional[Dict] = None, debug: bool = False) -> Dict:
  """Analyze contract with preconditions and postconditions"""
  # Handle both tuple format ('CONTRACT', {...}) and direct dict format {...}
  if isinstance(contract_data, tuple) and len(contract_data) >= 2:
    contract_type, contract_value = contract_data[0], contract_data[1]
    if contract_type != "CONTRACT" or not isinstance(contract_value, dict):
      raise ValueError(f"Invalid contract tuple format: {contract_data}")
  elif isinstance(contract_data, dict):
    contract_value = contract_data
  else:
    raise ValueError(f"Invalid contract format: {contract_data}")

  # Analyze precondition
  pre_ast = None
  if contract_value.get("pre"):
    pre_ast = analyze_expression(contract_value["pre"], env, debug)

  # Analyze postcondition with 'result' variable
  post_ast = None
  if contract_value.get("post"):
    post_env = env_bind(
        env, "result", return_type or make_type_info("Unknown"))
    post_ast = analyze_expression(contract_value["post"], post_env, debug)

  return make_ast_node(
      "CONTRACT",
      {"pre": pre_ast, "post": post_ast},
      [ast for ast in [pre_ast, post_ast] if ast is not None],
      None,
      make_type_info("Contract")
  )


# ============================================================================
# CST NODE ANALYSIS
# ============================================================================

def analyze_cst_node(cst_node: CSTNode, env: Dict, debug: bool = False) -> Dict:
  """Analyze a single CST node and return AST node"""
  if debug:
    print(f"Analyzing CST node: {cst_node.type} with value: {cst_node.value}")

  handlers = {
      "FUNCTION_DEF": analyze_function_def,
      "VALUE_BINDING": analyze_value_binding,
      "TYPE_DEF": analyze_type_def,
      "MODULE_SIG": analyze_module_sig,
      "INLINE_EVAL": analyze_inline_eval,
      "IDENTIFIER": analyze_identifier,
      "NUMBER": analyze_number,
      "STRING": analyze_string,
      "FUNCTION_CALL": analyze_function_call,
      "FUNCTION_NAME": analyze_function_name,
      "LIST": analyze_list,
      "RECORD": analyze_record,
      "PARENTHESIZED": analyze_parenthesized,
      "PIPELINE": analyze_pipeline,
  }

  if cst_node.type in handlers:
    return handlers[cst_node.type](cst_node, env, debug)
  else:
    # Default: convert to AST with children
    children = [analyze_cst_node(child, env, debug)
                for child in cst_node.children]
    return make_ast_node(cst_node.type, cst_node.value, children, cst_node.span)


def analyze_function_def(cst_node: CSTNode, env: Dict, debug: bool = False) -> Dict:
  """Analyze function definition"""
  if not isinstance(cst_node.value, dict):
    raise ValueError(f"Invalid function definition structure")

  func_data = cst_node.value
  func_name = extract_identifier_name(func_data.get('name'))

  if not func_name:
    raise ValueError("Function definition missing name")

  # Analyze parameters
  func_env = env
  param_types = []
  params_for_interpreter = []  # Keep raw params for interpreter pattern matching

  for param in func_data.get('params', []):
    if isinstance(param, tuple) and len(param) >= 2:
      param_type, param_data = param[0], param[1]
      if param_type == "TYPED_PARAM" and isinstance(param_data, dict):
        param_name = extract_pattern_name(param_data.get('name'))
        param_type_info = analyze_type_expr(param_data.get('type'))
        if param_name and param_type_info:
          func_env = env_bind(func_env, param_name, param_type_info)
          param_types.append(param_type_info)
        # Store the parameter for interpreter (including pattern info)
        params_for_interpreter.append(param)
      elif param_type == "UNTYPED_PARAM":
        # Untyped parameters - extract pattern for interpreter
        # param_data could be ('IDENTIFIER', 'x'), ('PATTERN_LIST', []), ('PATTERN_LITERAL', ...), etc.
        param_name = extract_pattern_name(param_data)
        if param_name:
          # Simple identifier pattern - bind with Unknown type
          func_env = env_bind(func_env, param_name, make_type_info("Unknown"))
        param_types.append(make_type_info("Unknown"))
        # Store the parameter for interpreter pattern matching
        params_for_interpreter.append(param)

  # Analyze return type
  return_type = None
  if func_data.get('return_type'):
    return_type = analyze_type_expr(func_data['return_type'])

  # Analyze guard
  guard_ast = None
  if func_data.get('guard'):
    guard_tuple = func_data['guard']
    if isinstance(guard_tuple, tuple) and len(guard_tuple) >= 2:
      # Guard is ('GUARD', expression)
      guard_expr = guard_tuple[1]
      guard_ast = analyze_expression(guard_expr, func_env, debug)

  # Analyze contract
  contract_ast = None
  if func_data.get('contract'):
    contract_ast = analyze_contract(
        func_data['contract'], func_env, return_type, debug)

  # Analyze body
  body_ast = None
  if func_data.get('body'):
    body_ast = analyze_expression(func_data['body'], func_env, debug)

  # Create function type
  func_type = make_type_info(
      "Function", param_types + ([return_type] if return_type else []))

  # Bind function in environment (return new env)
  new_env = env_bind(env, func_name, func_type)

  # Analyze CST children
  children = [analyze_cst_node(child, env, debug)
              for child in cst_node.children]

  return make_ast_node(
      "FUNCTION_DEF",
      {
          "name": func_name,
          "params": param_types,
          # Raw params for interpreter pattern matching
          "raw_params": params_for_interpreter,
          "return_type": return_type,
          "guard": guard_ast,
          "contract": contract_ast,
          "body": body_ast
      },
      children,
      cst_node.span,
      func_type
  )


def analyze_value_binding(cst_node: CSTNode, env: Dict, debug: bool = False) -> Dict:
  """Analyze value binding"""
  if not isinstance(cst_node.value, dict):
    raise ValueError("Invalid value binding structure")

  binding_data = cst_node.value
  var_name = extract_identifier_name(binding_data.get('name'))

  if not var_name:
    raise ValueError("Value binding missing name")

  # Analyze value expression
  value_ast = None
  if binding_data.get('value'):
    value_ast = analyze_expression(binding_data['value'], env, debug)

  # Infer type from value
  value_type = value_ast['type_info'] if value_ast else make_type_info(
      "Unknown")

  # Bind in environment
  new_env = env_bind(env, var_name, value_type)

  # Analyze CST children
  children = [analyze_cst_node(child, env, debug)
              for child in cst_node.children]

  return make_ast_node(
      "VALUE_BINDING",
      {
          "name": var_name,
          "value": value_ast
      },
      children,
      cst_node.span,
      value_type
  )


def analyze_type_def(cst_node: CSTNode, env: Dict, debug: bool = False) -> Dict:
  """Analyze type definition"""
  if not isinstance(cst_node.value, dict):
    raise ValueError("Invalid type definition structure")

  type_data = cst_node.value
  type_name = extract_identifier_name(type_data.get('name'))

  if not type_name:
    raise ValueError("Type definition missing name")

  # Create type info
  type_info = make_type_info(type_name)

  # Bind type in environment
  new_env = env_bind_type(env, type_name, type_info)

  # Analyze children
  children = [analyze_cst_node(child, env, debug)
              for child in cst_node.children]

  return make_ast_node(
      "TYPE_DEF",
      {"name": type_name, "constructors": type_data.get('constructors', [])},
      children,
      cst_node.span,
      type_info
  )


def analyze_module_sig(cst_node: CSTNode, env: Dict, debug: bool = False) -> Dict:
  """Analyze module signature"""
  children = [analyze_cst_node(child, env, debug)
              for child in cst_node.children]
  return make_ast_node(
      "MODULE_SIG",
      cst_node.value,
      children,
      cst_node.span,
      make_type_info("ModuleSignature")
  )


def analyze_inline_eval(cst_node: CSTNode, env: Dict, debug: bool = False) -> Dict:
  """Analyze inline evaluation statement (> expr.)"""
  # The value is the expression to evaluate
  expr_ast = analyze_expression(cst_node.value, env, debug)
  return make_ast_node("INLINE_EVAL", expr_ast, [], cst_node.span)


def analyze_identifier(cst_node: CSTNode, env: Dict, debug: bool = False) -> Dict:
  """Analyze identifier"""
  name = cst_node.value if isinstance(
      cst_node.value, str) else str(cst_node.value)
  type_info = env_lookup(env, name) or make_type_info("Unknown")
  return make_ast_node("IDENTIFIER", name, [], cst_node.span, type_info)


def analyze_number(cst_node: CSTNode, env: Dict, debug: bool = False) -> Dict:
  """Analyze number literal"""
  return make_ast_node("NUMBER", cst_node.value, [], cst_node.span, make_type_info("Num"))


def analyze_string(cst_node: CSTNode, env: Dict, debug: bool = False) -> Dict:
  """Analyze string literal"""
  return make_ast_node("STRING", cst_node.value, [], cst_node.span, make_type_info("String"))


def analyze_function_call(cst_node: CSTNode, env: Dict, debug: bool = False) -> Dict:
  """Analyze function call from CST"""
  if isinstance(cst_node.value, list):
    return analyze_function_call_tuple(cst_node.value, env, debug)
  elif cst_node.children:
    # Handle children-based function call (from expression parsing)
    # First child is the function, rest are arguments
    if len(cst_node.children) >= 1:
      func_ast = analyze_cst_node(cst_node.children[0], env, debug)
      arg_asts = [analyze_cst_node(child, env, debug)
                  for child in cst_node.children[1:]]

      return make_ast_node(
          "FUNCTION_CALL",
          {"function": func_ast, "args": arg_asts},
          [func_ast] + arg_asts,
          cst_node.span,
          make_type_info("Unknown")
      )

  # Fallback for unexpected structure
  children = [analyze_cst_node(child, env, debug)
              for child in cst_node.children]
  return make_ast_node("FUNCTION_CALL", cst_node.value, children, cst_node.span)


def analyze_list(cst_node: CSTNode, env: Dict, debug: bool = False) -> Dict:
  """Analyze list from CST"""
  children = [analyze_cst_node(child, env, debug)
              for child in cst_node.children]
  elem_type = children[0]['type_info'] if children else make_type_info(
      "Unknown")
  list_type = make_type_info("List", [elem_type])
  return make_ast_node("LIST", cst_node.value, children, cst_node.span, list_type)


def analyze_record(cst_node: CSTNode, env: Dict, debug: bool = False) -> Dict:
  """Analyze record literal from CST
  Record: { key = value, key = value }
  """
  fields = {}
  analyzed_fields = []

  for child in cst_node.children:
    if child.type == "RECORD_FIELD":
      # child.value should be a dict like {"key": "action1", "value": expr_cst}
      field_data = child.value
      key_raw = field_data["key"]

      # Extract string from key (handle both str and tuple)
      if isinstance(key_raw, str):
        key = key_raw
      elif isinstance(key_raw, tuple):
        key = key_raw[1] if len(key_raw) > 1 else str(key_raw)
      else:
        key = str(key_raw)

      value_cst = field_data["value"]

      # Analyze the value expression
      analyzed_value = analyze_tuple_expr(value_cst, env, debug)

      fields[key] = analyzed_value
      analyzed_fields.append({
          "key": key,
          "value": analyzed_value
      })

  return make_ast_node("RECORD", fields, analyzed_fields, cst_node.span, make_type_info("Record"))


def analyze_parenthesized(cst_node: CSTNode, env: Dict, debug: bool = False) -> Dict:
  """Analyze parenthesized expression from CST

  Parenthesized nodes have their actual expression in the value as a tuple like:
  ('FUNCTION_CALL', [('FUNCTION_NAME', ...), ('NUMBER', 3)])
  """
  if isinstance(cst_node.value, tuple):
    # Parse the tuple expression
    return analyze_tuple_expr(cst_node.value, env, debug)
  elif cst_node.children:
    # Fall back to analyzing first child if it has children
    return analyze_cst_node(cst_node.children[0], env, debug)
  else:
    # Shouldn't happen, but handle gracefully
    return make_ast_node("UNKNOWN", cst_node.value, [], cst_node.span)


def analyze_pipeline(cst_node: CSTNode, env: Dict, debug: bool = False) -> Dict:
  """Analyze pipeline expression
  Desugar: x |> f becomes f(x) - just a function call
  """
  if isinstance(cst_node.value, dict):
    input_ast = analyze_expression(cst_node.value["input"], env, debug)
    func_ast = analyze_expression(cst_node.value["function"], env, debug)

    # Desugar pipeline to function call: x |> f => f(x)
    return make_ast_node(
        "FUNCTION_CALL",
        {"function": func_ast, "args": [input_ast]},
        [func_ast, input_ast],
        cst_node.span,
        make_type_info("Unknown")
    )
  else:
    children = [analyze_cst_node(child, env, debug)
                for child in cst_node.children]
    return make_ast_node("PIPELINE", cst_node.value, children, cst_node.span)


def analyze_pipeline_tuple(pipeline_data: Any, env: Dict, debug: bool = False) -> Dict:
  """Analyze pipeline expression from tuple representation
  Desugar: x |> f becomes f(x) - just a function call
  """
  if isinstance(pipeline_data, dict):
    input_ast = analyze_expression(pipeline_data["input"], env, debug)
    func_ast = analyze_expression(pipeline_data["function"], env, debug)

    # Desugar pipeline to function call: x |> f => f(x)
    return make_ast_node(
        "FUNCTION_CALL",
        {"function": func_ast, "args": [input_ast]},
        [func_ast, input_ast],
        None,
        make_type_info("Unknown")
    )
  else:
    # Unexpected format
    return make_ast_node("PIPELINE", pipeline_data, [], None, make_type_info("Unknown"))


def analyze_dollar_app_tuple(dollar_data: Any, env: Dict, debug: bool = False) -> Dict:
  """Analyze dollar application expression from tuple representation
  Desugar: f $ x becomes f(x) - just function application with lower precedence
  This is already desugared to FUNCTION_CALL, keeping for consistency
  """
  if isinstance(dollar_data, dict):
    func_ast = analyze_expression(dollar_data["function"], env, debug)
    arg_ast = analyze_expression(dollar_data["argument"], env, debug)

    # Desugar dollar application to function call: f $ x => f(x)
    return make_ast_node(
        "FUNCTION_CALL",
        {"function": func_ast, "args": [arg_ast]},
        [func_ast, arg_ast],
        None,
        make_type_info("Unknown")
    )
  else:
    # Unexpected format
    return make_ast_node("DOLLAR_APP", dollar_data, [], None, make_type_info("Unknown"))


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_program(cst_nodes: List[CSTNode], debug: bool = False) -> Tuple[List[Dict], Dict]:
  """
  Analyze a program (list of CST nodes) and return AST nodes and final environment.
  This is a pure function that threads the environment through all analyses.
  """
  env = create_builtin_env()
  ast_nodes = []

  for cst_node in cst_nodes:
    ast_node = analyze_cst_node(cst_node, env, debug)
    ast_nodes.append(ast_node)

    # Thread environment: if this node binds something, update env
    if ast_node['type'] == 'FUNCTION_DEF':
      func_name = ast_node['value']['name']
      func_type = ast_node['type_info']
      env = env_bind(env, func_name, func_type)
    elif ast_node['type'] == 'VALUE_BINDING':
      var_name = ast_node['value']['name']
      var_type = ast_node['type_info']
      env = env_bind(env, var_name, var_type)
    elif ast_node['type'] == 'TYPE_DEF':
      type_name = ast_node['value']['name']
      type_info = ast_node['type_info']
      env = env_bind_type(env, type_name, type_info)

  return ast_nodes, env


# ============================================================================
# EXCEPTION CLASS (for compatibility)
# ============================================================================

class REI1SemanticsError(Exception):
  """REI1 semantics analysis error"""

  def __init__(self, message: str, span: Optional[SourceSpan] = None):
    self.message = message
    self.span = span
    super().__init__(self._format_error())

  def _format_error(self) -> str:
    if self.span:
      return f"Semantics error at {self.span}: {self.message}"
    return f"Semantics error: {self.message}"


# ============================================================================
# FACTORY FUNCTIONS (for compatibility with main.py)
# ============================================================================

def create_analyzer(debug: bool = False):
  """Factory function returning an analyzer function"""
  def analyzer(cst_nodes):
    try:
      ast_nodes, _ = analyze_program(cst_nodes, debug)
      return ast_nodes
    except (ValueError, KeyError) as e:
      raise REI1SemanticsError(str(e))

  def analyzer_module(cst_node):
    try:
      env = create_builtin_env()
      ast_node = analyze_cst_node(cst_node, env, debug)
      # Wrap in object-like structure with all attributes
      return type('ASTNode', (), {
          'type': ast_node.get('type', 'Unknown'),
          'value': ast_node.get('value', {}),
          'children': ast_node.get('children', []),
          'span': ast_node.get('span'),
          'type_info': ast_node.get('type_info'),
          'resolved_names': ast_node.get('resolved_names', {})
      })()
    except (ValueError, KeyError) as e:
      raise REI1SemanticsError(str(e))

  def analyze_node_wrapper(cst_node, env):
    """Wrap analyze_cst_node result in object-like structure for main.py compatibility"""
    ast_node = analyze_cst_node(cst_node, env, debug)
    # Create mock object with .type and .value attributes
    return type('ASTNode', (), {
        'type': ast_node.get('type', 'Unknown'),
        'value': ast_node.get('value', {})
    })()

  return type('Analyzer', (), {
      'analyze': lambda self, cst_nodes: analyzer(cst_nodes),
      'analyze_module': lambda self, cst_node: analyzer_module(cst_node),
      'global_env': create_builtin_env(),
      '_analyze_node': lambda self, cst_node, env: analyze_node_wrapper(cst_node, env)
  })()


def create_debug_analyzer():
  """Factory function returning a debug analyzer"""
  return create_analyzer(debug=True)
