REI1 is a pure functional programming language combining ML's composability, Haskell's syntax elegance, and LISP's prefix operations. It addresses modern development pain points through orthogonal features while maintaining high simplicity.

It focuses on expressiveness with minimal actual core concepts.
The features are meant to address specific pain points and remains orthogonal to other features.

Basic data looks like:

```rei1
# Numbers (unified Num type)
42, -17, 3.14159, -2.5

# Strings (List Char)
"hello world", "multi\nline"

# Characters
`a, `\n, `\t

# Lists
[1, 2, 3], ["a", "b"], []

# Booleans
True, False
```

Operators and basic functions:

```rei1
# Low Precedence Right Assoc Function application
f $ g $ h x. # f(g(h(x)))

# Pipeline
x |> f _ |> g _ |> h _. # h(g(f(x)))

# Mid precedence reverse pipeline
x <| f <| g.

# Partial application
increment = 1 + ?.
add-to-end = ? + " END".

# Case expressions (only control flow)
result = case value of
  Some x => process x
| None => default-value.

# Not implemented placeholder
todo-function x = ??.
```

-------

**Expression-based** - No statements, everything is an expression:

```rei1
result =
  x = case input of
    Some x => + x 1
  | None => 0;
  x + 1.
```

Note the `;` means let expression.

ADT:

```rei1
# name, age, email
User = Type (User String Num String).

# Usage
alice = User "Alice" 30 "alice@example.com".
name = User.name alice.
```

**Multiple dispatch** on argument types using guards:
```rei1
add : (x: 'a), (y: 'a) -> (z: 'a).

# == operator works for parameterized types on both sides
add x y | type x == Num and type y == Num = x + y.
add x y | type x == Str and type y == Str = x + y.
add x y | type x == List and type y == List = x + y.

result1 = add 3 4.
result2 = add "hi" "bye".
```

**Unsafe blocks** for side effects:
```rei1
# $ for right assoc less bind
# <| for more bind

main = unsafe $
  content = IO.read "config.txt";
  parsed = pure <| parse content;
  IO.write "output.txt" (process parsed);
  println "Done".
```

**Actor-based concurrency**:
```rei1
w msg = unsafe <| case msg of
  Job data reply-to => result = process data; Proc.send reply-to (Result result)
| Shutdown => Proc.send Proc.self Terminate.

ws = unsafe $ map (lambda _ => Proc.spawn w) (List.from 1 4).
```

**Signatures and modules**:
```rei1
# Signature defines interface
DB = Sig {
  connect = String -> Connection,
  query = Connection, String -> List Row,
  close = Connection -> Unit
}.

# you can declare aliases and splice them like a macro. But at runtime since they exist
ng = native-postgres.

# Module implements signature
PostgresDB = DB {
  connect url = unsafe <| ${ng}-connect url,
  query conn sql = unsafe <| ${ng}-query conn sql,
  close conn = unsafe <| ${ng}-close conn
}.
```

----

I want to push more things to the runtime if possible. Without as much hardcoding or custom syntactic constructs. Pattern matching for example could just take your value that you want to match on, then for each branch, try to match the pattern tree to the value tree (given that those are what they are represented in). No real special things are required, just parse the patterns and constructors to some relatively clean AST and maybe flatten it a bit in memory (through array like encoding). For modules, notice how they are almost exactly the same as a record if you zoom out. You could essentially just have a record which itself could be a list or array, then point the data at the fields of the module. For multiple dispatch, use a dispatch/3 function - the dispatch function takes a function name, its args and a list of types. It then pattern matches on those list of types and calls the right function to handle it. How could the codebase be updated to be more like that
