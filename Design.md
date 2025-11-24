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
# Function application
f $ g $ h x    # f(g(h(x)))

# Pipeline
x |> f |> g |> h    # h(g(f(x)))

# Partial application
increment = + 1 _.
add-to-end = ++ _ " END".

# Case expressions (only control flow)
result = case value of
  Some x => process x,
  None => default-value.

# Not implemented placeholder
todo-function x = ??.
```

-------

**Expression-based** - No statements, everything is an expression:

```rei1
result =
  x = case input of
    Some x => + x 1,
    None => 0;
  + x 1.
```

Note the `;` means let expression.

ADT:

```rei1
User = Type $ User String Num String.  # name, age, email

# Usage
alice = User "Alice" 30 "alice@example.com".
name = User.name alice.
```

**Multiple dispatch** on argument types:
```rei1
add (x : Num) (y : Num) = + x y.
add (x : String) (y : String) = + x y.
add (x : List a) (y : List a) = + x y.

result1 = add 3 4.        # Uses numeric addition
result2 = add "hi" "bye". # Uses string concatenation
```

**Unsafe blocks** for side effects:
```rei1
main = unsafe $
  content <- IO.read "config.txt";
  parsed <- pure $ parse content;
  IO.write "output.txt" (process parsed);
  println "Done".
```

**Actor-based concurrency**:
```rei1
w msg = unsafe $ case msg of
  Job data reply-to => result = process data; Proc.send reply-to (Result result),
  Shutdown => Proc.send Proc.self Terminate.

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
  connect url = unsafe $ {ng}-connect url,
  query conn sql = unsafe $ {ng}-query conn sql,
  close conn = unsafe $ {ng}-close conn
}.
```