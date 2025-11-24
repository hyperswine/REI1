from parsing import create_parser
from semantics import create_analyzer
from interpreter import create_interpreter

parser = create_parser()
analyzer = create_analyzer()
interpreter = create_interpreter()

code = '''
double x = * x 2.
numbers = [1, 2, 3, 4, 5].
result = par-map double numbers.
'''

cst = parser.parse_string(code)
print("CST:", cst)
ast = analyzer.analyze(cst)
print("AST:", ast)
result = interpreter.interpret_program(ast)
print("Result:", result)
print("result['result']:", result.get('result'))
