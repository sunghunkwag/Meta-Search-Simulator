import sys
import os
import ast
import random

# Add current directory to path
sys.path.append(os.getcwd())

try:
    import UNIFIED_RSI_EXTENDED
    from UNIFIED_RSI_EXTENDED import EvolutionarySearcher
    print("SUCCESS: Module loaded.")
except ImportError:
    print("CRITICAL: Module load failed.")
    sys.exit(1)

def test():
    print("Test 1: Generation")
    searcher = EvolutionarySearcher({})
    ast_mod = searcher.generate_random_ast()
    print("  - AST generated.")
    code = ast.unparse(ast_mod)
    print(f"  - Code: {code}")

    print("Test 2: Mutation")
    mutated = searcher.mutate(ast_mod)
    m_code = ast.unparse(mutated)
    print(f"  - Mutated: {m_code}")

    if code != m_code:
        print("SUCCESS: Mutation verified.")
    else:
        print("WARNING: No mutation observed (random chance?).")

if __name__ == "__main__":
    test()
