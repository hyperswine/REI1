#!/bin/bash
# Test script for REPL functionality

echo "Testing REPL with function definition and call..."
echo -e "double x = * x 2.\ndouble 4\nexit." | uv run python main.py -i
