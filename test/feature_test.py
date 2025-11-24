"""
Comprehensive feature test for REI1 parser implementation
Tests all features listed in DESIGN.md systematically
Pure functional style
"""

from typing import Dict, List, Tuple
from debug_lang import REI1Debugger, debug_full_parse
from parsing import REI1Grammar


# ============================================================================
# PURE TEST FUNCTIONS
# ============================================================================

def run_feature_test(name: str, code: str, grammar: REI1Grammar, verbose: bool = False) -> Dict:
    """Test a specific feature and return result - pure function"""
    try:
        result = debug_full_parse(grammar, code, "<test>", verbose=verbose)
        success = result["status"] == "success"
        return {
            'name': name,
            'code': code,
            'success': success,
            'result': result
        }
    except Exception as e:
        return {
            'name': name,
            'code': code,
            'success': False,
            'result': {"status": "error", "error": "exception", "details": str(e)}
        }


def print_test_result(test_result: Dict) -> None:
    """Print test result with formatting"""
    print(f"\n{'='*60}")
    print(f"TESTING: {test_result['name']}")
    print(f"{'='*60}")
    print(f"Code: {test_result['code']}")
    print("-" * 40)

    if test_result['success']:
        print(f"✅ SUCCESS: {test_result['name']}")
    else:
        print(f"❌ FAILED: {test_result['name']}")
        error_details = test_result['result'].get('details', 'Unknown error')
        print(f"Error: {error_details}")


def get_test_cases() -> List[Tuple[str, str]]:
    """Get list of test cases - pure function"""
    return [
        # Basic syntax
        ("Value binding", "pi = 3.14159."),
        ("Simple function", "add x y = + x y."),
        ("Comments", "// This is a comment\nvalue = 42."),

        # Literals
        ("Number literals", "nums = [42, -17, 3.14, -2.5]."),
        ("String literals", 'greeting = "hello world".'),
        ("Boolean literals", "flags = [True, False]."),
        ("Character literals", "chars = [`a, `\\n, `\\t]."),
        ("List literals", "myList = [1, 2, 3]."),

        # Operators
        ("Pipeline operator", "result = x |> f |> g."),
        ("Dollar operator", "result = f $ g $ x."),
        ("Partial application", "increment = add 1 _."),

        # Pattern matching
        ("Case expressions", """factorial n = case n of
            0 => 1,
            n => * n (factorial (- n 1))."""),

        # Types
        ("Function types", """add : Num -> (Num -> Num) = lam x => lam y => + x y."""),
        ("Type definitions", "User = Type $ User String Num String."),

        # Module system
        ("Module signatures", """MathSig = Sig (
            add : Num -> Num -> Num,
            zero : Num
        )."""),

        # Contracts
        ("Design by contract", """divide (x : Num) (y : Num) -> Num
            where {pre: y /= 0, post: * result y ~= x} =
            / x y."""),

        # Lambda expressions
        ("Lambda expressions", "map-func = lam x => + x 1."),

        # Advanced features
        ("List patterns", """head lst = case lst of
            [] => ??,
            (x :: _) => x."""),
        ("Constructor patterns", """process opt = case opt of
            Nothing => "empty",
            (Just x) => show x."""),
        ("Tuple types", "pair : (Num, String) = (42, \"hello\")."),
    ]


def run_all_tests(verbose: bool = False) -> List[Dict]:
    """Run all tests and return results - pure function"""
    grammar = REI1Grammar()
    tests = get_test_cases()
    results = []

    for name, code in tests:
        test_result = run_feature_test(name, code, grammar, verbose)
        if not verbose:
            print_test_result(test_result)
        results.append(test_result)

    return results


def print_summary(results: List[Dict]) -> None:
    """Print test summary"""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for r in results if r['success'])
    total = len(results)

    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print("\n✅ WORKING FEATURES:")
    for result in results:
        if result['success']:
            print(f"  • {result['name']}")

    print("\n❌ MISSING FEATURES:")
    for result in results:
        if not result['success']:
            print(f"  • {result['name']}")

    print(f"\n{'='*60}")
    print("NEXT PRIORITIES:")
    print("Based on failed tests, implement these features next:")
    failed_features = [r['name'] for r in results if not r['success']]
    for i, feature in enumerate(failed_features[:5], 1):
        print(f"  {i}. {feature}")


# ============================================================================
# COMPATIBILITY WRAPPER
# ============================================================================

def test_feature(name: str, code: str, debugger: REI1Debugger) -> bool:
    """Compatibility wrapper for old test_feature function"""
    result = run_feature_test(name, code, debugger.grammar, verbose=False)
    print_test_result(result)
    return result['success']


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Run comprehensive feature tests"""
    print("REI1 PARSER FEATURE ASSESSMENT")
    print("=" * 60)
    print("Testing implementation against DESIGN.md requirements...")

    results = run_all_tests(verbose=False)
    print_summary(results)


if __name__ == "__main__":
    main()
