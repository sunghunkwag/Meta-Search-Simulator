import sys
import os
import time
import ast
import random

# Ensure we can import from the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from UNIFIED_RSI_EXTENDED import EvolutionarySearcher, InventionRepresentation, Task, InventionEvaluator
    print("SUCCESS: Module loaded.")
except ImportError as e:
    # Try adding parent directory if running from subdirectory but UNIFIED_RSI_EXTENDED is in parent
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from UNIFIED_RSI_EXTENDED import EvolutionarySearcher, InventionRepresentation, Task, InventionEvaluator
        print("SUCCESS: Module loaded (from parent).")
    except ImportError as e2:
        print(f"CRITICAL: Module load failed: {e}")
        print(f"CRITICAL: Module load failed (parent attempt): {e2}")
        sys.exit(1)

def test_loop():
    print("Starting Verification Loop Test (Infinite)...")

    # Initialize components
    repr_obj = InventionRepresentation()
    searcher = EvolutionarySearcher(repr_obj)
    evaluator = InventionEvaluator(timeout=0.1)

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        try:
            # 1. Generate Candidate
            code, idx = searcher.generate_candidate()
            print(f"  [Gen] Strategy {idx} generated.")

            # 2. Parse Check
            try:
                tree = ast.parse(code)
                print("  [Parse] AST valid.")
            except SyntaxError as se:
                print(f"  [Parse] SyntaxError: {se}")
                continue

            # 3. Refine Candidate (Simulation)
            # Create a simple task: f(x) = x + 1
            inputs = [1, 2, 3]
            outputs = [2, 3, 4]
            task = Task(kind='sequence', input=inputs, expected=outputs)

            refined_code = searcher.refine_candidate(code, [task])

            # 4. Check if refinement did anything (it might not if strategy doesn't have holes or logic is complex)
            if refined_code != code:
                print("  [Refine] Code modified by synthesizer.")
            else:
                print("  [Refine] No modification (or no holes).")

            # 5. Execute/Evaluate (Safe check)
            score = evaluator.evaluate(refined_code, [task])
            print(f"  [Eval] Score: {score}")

        except Exception as e:
            print(f"  [ERROR] Exception in loop: {e}")

        # Sleep to prevent tight loop CPU hogging
        time.sleep(1)

if __name__ == "__main__":
    test_loop()
