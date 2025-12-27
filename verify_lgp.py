import sys
import os
import random
from pathlib import Path
sys.path.append(os.getcwd())
from UNIFIED_RSI_EXTENDED import Universe, MetaState, TaskSpec, FunctionLibrary, seed_genome, Genome, validate_code, safe_exec

def verify():
    print("Verifying LGP Implementation...")
    # 1. Verify Code Validation
    code_good = "def run(x):\n    v0=x\n    return v0*2"
    ok, err = validate_code(code_good)
    print(f"Validation Good: {ok}")
    if not ok: print(err)
    
    code_bad = "def run(x):\n    import os\n    return x"
    ok, err = validate_code(code_bad)
    print(f"Validation Bad (Expected False): {ok}, Err: {err}")

    # 2. Verify Execution
    res = safe_exec(code_good, 10.0)
    print(f"Exec Good (Expected 20.0): {res}")

    # 3. Verify Universe Loop
    print("\nInitializing Universe...")
    univ = Universe(
        uid=1, seed=42, 
        meta=MetaState(mutation_rate=0.5), 
        pool=[], 
        library=FunctionLibrary()
    )
    task = TaskSpec(name='poly2', x_min=-5, x_max=5)
    
    # Debug Seed
    sg = seed_genome(random.Random(42))
    print(f"Seed Genome Code:\n{sg.code}")
    from UNIFIED_RSI_EXTENDED import evaluate, sample_batch
    b = sample_batch(random.Random(42), task)
    res = evaluate(sg, b)
    print(f"Seed Eval Result: ok={res.ok}, score={res.score}, err={res.err}")

    # Run 5 gens
    for g in range(10):
        log = univ.step(g, task, pop_size=20)
        if 'score' in log:
            print(f"Gen {g}: Score={log['score']:.4f} CodeLineCount={len(log['code'].splitlines())}")
            if 'while' in log['code'] or 'if' in log['code']:
                print("Found Control Flow!")
        else:
            print(f"Gen {g}: Reseed! Reason={log.get('reason')}")

if __name__ == "__main__":
    verify()
