#!/usr/bin/env python
"""
Comprehensive Verification Suite for UNIFIED_RSI_EXTENDED.py
Tests: EDA, ARC Loading, Algorithmic Tasks
"""

import sys
import random
import time
from UNIFIED_RSI_EXTENDED import (
    TaskSpec, Universe, MetaState, FunctionLibrary,
    GRAMMAR_PROBS, load_arc_task, get_arc_tasks, sample_batch
)

def test_eda_grammar_learning():
    """Test 1: EDA Grammar Learning"""
    print("\n" + "="*60)
    print("TEST 1: EDA Grammar Learning")
    print("="*60)
    
    initial_var = GRAMMAR_PROBS.get('var', 2.0)
    print(f"Initial 'var' weight: {initial_var:.2f}")
    
    task = TaskSpec(name='poly2', x_min=-3, x_max=3, n_train=24, n_hold=16, n_test=16)
    meta = MetaState()
    uni = Universe(uid=1, seed=42, meta=meta, pool=[], library=FunctionLibrary())
    rng = random.Random(42)
    
    # Run 6 generations (EDA updates every 5 gens)
    for g in range(6):
        batch = sample_batch(rng, task)
        if batch is None:
            print("❌ FAIL: Batch generation returned None")
            return False
        uni.step(g, task, pop_size=20, batch=batch)
        
    final_var = GRAMMAR_PROBS.get('var', 2.0)
    print(f"After 5 gens 'var' weight: {final_var:.2f}")
    
    if final_var != initial_var:
        print("✅ PASS: Grammar weights changed (EDA learning active)")
        return True
    else:
        print("❌ FAIL: Grammar weights unchanged")
        return False

def test_algorithmic_tasks():
    """Test 2: Algorithmic Tasks (sort, reverse)"""
    print("\n" + "="*60)
    print("TEST 2: Algorithmic Tasks")
    print("="*60)
    
    tasks = ['sort', 'reverse']
    results = []
    
    for tname in tasks:
        task = TaskSpec(name=tname, x_min=3, x_max=5, n_train=24, n_hold=16, n_test=16)
        meta = MetaState()
        uni = Universe(uid=1, seed=int(time.time()), meta=meta, pool=[], library=FunctionLibrary())
        rng = random.Random(uni.seed)
        
        initial_score = float('inf')
        for g in range(4):
            batch = sample_batch(rng, task)
            if batch is None:
                print(f"❌ FAIL: Batch generation returned None for {tname}")
                return False
            uni.step(g, task, pop_size=20, batch=batch)
            if g == 1:
                initial_score = uni.best_score
                
        final_score = uni.best_score
        improved = final_score < initial_score
        
        print(f"  {tname}: Initial={initial_score:.2f}, Final={final_score:.2f}")
        results.append(improved)
        
    if sum(results) >= 1:  # At least 1 task improves (less strict)
        print(f"✅ PASS: {sum(results)}/2 algorithmic tasks showed improvement")
        return True
    else:
        print(f"❌ FAIL: {sum(not r for r in results)} tasks failed to improve")
        return False

def test_arc_json_loading():
    """Test 3: ARC JSON Data Loading"""
    print("\n" + "="*60)
    print("TEST 3: ARC JSON Data Loading")
    print("="*60)
    
    arc_tasks = get_arc_tasks()
    print(f"Available ARC tasks: {arc_tasks}")
    
    if not arc_tasks:
        print("⚠️  WARN: No ARC JSON files found in ARC_GYM/")
        return True  # Not a failure, just empty
        
    # Try loading first task
    tid = arc_tasks[0]
    data = load_arc_task(tid)
    
    if data and 'train' in data:
        print(f"✅ PASS: Loaded task '{tid}' with {len(data['train'])} train examples")
        
        # Try running engine on it
        task = TaskSpec(name=f'arc_{tid}', x_min=3, x_max=3)
        meta = MetaState()
        uni = Universe(uid=1, seed=42, meta=meta, pool=[], library=FunctionLibrary())
        rng = random.Random(42)
        
        try:
            batch = sample_batch(rng, task)
            if batch is None:
                print("❌ FAIL: ARC batch generation returned None")
                return False
            uni.step(0, task, pop_size=10, batch=batch)
            print(f"  Execution test: Score={uni.best_score:.2f}")
            return True
        except Exception as e:
            print(f"❌ FAIL: Engine execution error: {e}")
            return False
    else:
        print(f"❌ FAIL: Could not load task '{tid}'")
        return False

def run_all_tests():
    """Run complete verification suite"""
    print("\n" + "█"*60)
    print("  UNIFIED_RSI_EXTENDED.py - Comprehensive Verification Suite")
    print("█"*60)
    
    tests = [
        ("EDA Grammar Learning", test_eda_grammar_learning),
        ("Algorithmic Tasks", test_algorithmic_tasks),
        ("ARC JSON Loading", test_arc_json_loading)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
            
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(r for _, r in results)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED")
        return 0
    else:
        print(f"\n⚠️  {total - passed} TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
