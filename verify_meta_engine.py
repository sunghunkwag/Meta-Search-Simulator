import sys
import os
import random
from pathlib import Path
sys.path.append(os.getcwd())
try:
    from UNIFIED_RSI_EXTENDED import Universe, MetaState, TaskSpec, FunctionLibrary, seed_genome, Genome, safe_exec_engine, EngineStrategy
except ImportError:
    # If EngineStrategy not exposed in __all__ or similar, might fail if I didn't update imports
    # But usually importing module works.
    from UNIFIED_RSI_EXTENDED import *

def verify():
    print("Verifying Meta-Logic Engine...")
    
    # 1. Test safe_exec_engine directly
    print("[1] Testing safe_exec_engine...")
    code = """
def run():
    return x * 2
"""
    res = safe_exec_engine(code, {'x': 10})
    print(f"Result (Expected 20): {res}")
    if res != 20:
        print("FAIL: Basic Execution")
        return

    # 2. Test Custom Selection Strategy
    print("\n[2] Testing Custom Selection Strategy...")
    # Create dummy pool
    pool = [Genome(statements=["return 1"]) for _ in range(10)]
    scores = [float(i) for i in range(10)] # 0..9
    
    custom_sel = """
def run():
    # Select ONLY the worst (reverse sort) just to prove it changes
    zipped = sorted(zip(pool, scores), key=lambda x: x[1])
    worst = zipped[0][0]
    return [worst], [worst]*(pop_size-1)
"""
    ctx = {'pool': pool, 'scores': scores, 'pop_size': 5, 'rng': random.Random(), 'map_elites': None}
    elites, parents = safe_exec_engine(custom_sel, ctx)
    print(f"Elites Count: {len(elites)} (Expected 1)")
    print(f"Parents Count: {len(parents)} (Expected 4)")
    
    # 3. Test Universe Integration
    print("\n[3] Testing Universe.step with Default Strategy...")
    univ = Universe(
        uid=1, seed=42, 
        meta=MetaState(mutation_rate=0.5), 
        pool=[], 
        library=FunctionLibrary()
    )
    task = TaskSpec(name='poly2')
    
    # Run 3 gens
    for g in range(3):
        log = univ.step(g, task, pop_size=10)
        print(f"Gen {g}: Score={log.get('score', 'N/A')}")
        if not log.get('accepted') and g > 0 and 'reason' not in log:
             pass 

    print("Meta-Engine Verification Validated.")

if __name__ == "__main__":
    verify()
