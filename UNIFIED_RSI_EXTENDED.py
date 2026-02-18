"""
UNIFIED_RSI_EXTENDED.py

Neuro-Symbolic Recursive Self-Improvement Engine
================================================
Architecture:
1. Logic: Z3 Constraint Synthesis (ConstraintSynthesizer)
2. Intuition: Neural Guidance via Numpy MLP (NeuralGuide)
3. Structure: Grammar-Guided Genetic Programming (StructureBreeder)

Directives:
- NO Safety Rails (Unrestricted Execution)
- NO LLM/GPT (Symbolic/Neural only)
- Tabula Rasa Rewrite
"""
from __future__ import annotations

import argparse
import ast
import collections
import difflib
import hashlib
import json
import math
import copy
import os
import random
import re
import shutil
import sys
import tempfile
import textwrap
import time
import traceback
import multiprocessing as mp
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set, Union

# External dependencies (The Trinity)
import z3
import numpy as np
import atomic_factory

# ---------------------------
# Utilities
# ---------------------------

def now_ms() -> int:
    return int(time.time() * 1000)

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        safe_mkdir(self.path.parent)
        if not self.path.exists():
            with self.path.open("w") as f: f.write("")

    def log(self, data: Dict[str, Any]):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")

# ---------------------------
# Component 1: Logic (Constraint Synthesis)
# ---------------------------

class ConstraintSynthesizer:
    """Uses Z3 to mathematically derive expressions satisfying I/O constraints."""

    def synthesize_expression(self, inputs: List[Any], outputs: List[Any], var_name: str = 'x') -> Optional[ast.AST]:
        """Synthesizes a linear or modular arithmetic expression."""
        if not inputs or not outputs: return None

        # Filter for numeric only
        numeric_io = []
        for i, o in zip(inputs, outputs):
            if isinstance(i, (int, float)) and isinstance(o, (int, float)):
                numeric_io.append((int(i), int(o)))

        if not numeric_io: return None

        solver = z3.Solver()
        a, b, m = z3.Ints('a b m')

        # Strategy 1: Linear (y = ax + b)
        solver.push()
        for i_val, o_val in numeric_io:
            solver.add(a * i_val + b == o_val)

        if solver.check() == z3.sat:
            model = solver.model()
            va = model[a].as_long()
            vb = model[b].as_long()
            # Optim: simplified AST
            if va == 1 and vb == 0: return ast.Name(id=var_name, ctx=ast.Load())
            if va == 0: return ast.Constant(value=vb)
            if vb == 0: return ast.BinOp(left=ast.Constant(value=va), op=ast.Mult(), right=ast.Name(id=var_name, ctx=ast.Load()))
            return ast.BinOp(
                left=ast.BinOp(left=ast.Constant(value=va), op=ast.Mult(), right=ast.Name(id=var_name, ctx=ast.Load())),
                op=ast.Add(),
                right=ast.Constant(value=vb)
            )
        solver.pop()

        # Strategy 2: Modular (y = (ax + b) % m)
        solver.push()
        solver.add(m > 0)
        for i_val, o_val in numeric_io:
            solver.add((a * i_val + b) % m == o_val)

        if solver.check() == z3.sat:
            model = solver.model()
            va = model[a].as_long()
            vb = model[b].as_long()
            vm = model[m].as_long()
            inner = ast.BinOp(
                left=ast.BinOp(left=ast.Constant(value=va), op=ast.Mult(), right=ast.Name(id=var_name, ctx=ast.Load())),
                op=ast.Add(),
                right=ast.Constant(value=vb)
            )
            return ast.BinOp(left=inner, op=ast.Mod(), right=ast.Constant(value=vm))
        solver.pop()

        return None

# ---------------------------
# Component 2: Intuition (Neural Guidance)
# ---------------------------

class TaskEncoder:
    """Encodes a task (I/O pairs) into a fixed-size vector."""
    def encode(self, task) -> np.ndarray:
        # Vector size 20
        vec = np.zeros(20)
        # Type encoding
        kind_map = {'sequence': 0, 'path': 1, 'transform': 2, 'aggregate': 3}
        vec[kind_map.get(task.kind, 4)] = 1.0

        # Stats
        inputs = task.input if isinstance(task.input, list) else [task.input]
        outputs = task.expected if isinstance(task.expected, list) else [task.expected]

        vec[5] = len(inputs) / 10.0
        vec[6] = 1.0 if any(isinstance(x, list) for x in inputs) else 0.0
        vec[7] = 1.0 if any(isinstance(x, int) for x in inputs) else 0.0

        # Hints
        if getattr(task, 'hint', None) == 'sort': vec[8] = 1.0
        if getattr(task, 'hint', None) == 'sum': vec[9] = 1.0

        return vec

class NeuralGuide:
    """A raw Numpy Multi-Layer Perceptron to guide search."""
    def __init__(self, input_dim=20, hidden_dim=64, output_dim=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0/input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0/hidden_dim)
        self.b2 = np.zeros(output_dim)

        self.lr = 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2

        # Softmax
        exps = np.exp(self.z2 - np.max(self.z2))
        self.probs = exps / np.sum(exps)
        return self.probs

    def train(self, x: np.ndarray, target_idx: int):
        if target_idx >= self.output_dim: return
        probs = self.forward(x)

        # Backprop (Cross Entropy Loss)
        d_z2 = probs
        d_z2[target_idx] -= 1.0

        d_W2 = np.outer(self.a1, d_z2)
        d_b2 = d_z2

        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * (self.z1 > 0) # ReLU deriv

        d_W1 = np.outer(x, d_z1)
        d_b1 = d_z1

        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2

# ---------------------------
# Component 3: Structure (G3P & Grammar)
# ---------------------------

class InventionRepresentation:
    """A weighted grammar that can grow."""
    def __init__(self):
        self.weights: Dict[str, Dict[int, float]] = {}
        self.library: List[str] = []

        # Base Grammar
        self.grammar = {
            'program': [self._gen_program],
            'solver': [self._gen_solver],
            'strategy': [
                self._strat_map,
                self._strat_filter,
                self._strat_reduce,
                self._strat_scan
            ]
        }

    def expand(self, symbol: str) -> Tuple[str, int]:
        prods = self.grammar.get(symbol, [])
        if not prods: return ("", -1)

        # Weighted selection
        ws = self.weights.get(symbol, {})
        weights = [ws.get(i, 1.0) for i in range(len(prods))]

        idx = random.choices(range(len(prods)), weights=weights, k=1)[0]
        choice = prods[idx]

        # Recursively expand
        res = choice(self)
        # If the result is a tuple (from recursive expand), unpack it?
        # But _gen_program calls expand('solver'), which returns (code, idx).
        # We need to standardize return types.

        if isinstance(res, tuple):
            return (res[0], idx) # Return code and the index of THIS level choice
        return (res, idx)

    def add_production(self, symbol: str, func: Callable):
        self.grammar.setdefault(symbol, []).append(func)

    def _gen_program(self, _):
        helpers = "\n".join(self.library)
        sol, sol_idx = self.expand('solver')
        code = f"{helpers}\n\n{sol}" if helpers else sol
        # We pass through the solver's strategy index if possible?
        # But expand() wraps it. expand('program') calls _gen_program.
        # _gen_program calls expand('solver').
        # If expand('program') returns (code, program_idx).
        # We want the strategy index eventually.
        # Let's attach metadata dict?
        return (code, sol_idx) # Hack: propagate solver index

    def _gen_solver(self, _):
        body, strat_idx = self.expand('strategy')
        return (f"def solve(task):\n{textwrap.indent(body, '    ')}", strat_idx)

    # --- Strategies (Skeletons with Holes) ---

    def _strat_map(self, _):
        return "return [__HOLE__(x) for x in task.input]"

    def _strat_filter(self, _):
        return "return [x for x in task.input if __HOLE__(x)]"

    def _strat_reduce(self, _):
        return "return sum(task.input)"

    def _strat_scan(self, _):
        return """
result = []
acc = 0
for x in task.input:
    acc = acc + x
    result.append(acc)
return result
""".strip()


class StructureBreeder:
    """Evolves the grammar itself via Abstraction (Sleep Phase)."""
    def __init__(self, repr: InventionRepresentation):
        self.repr = repr

    def breed(self, best_programs: List[str]):
        """Compresses knowledge by mining common subtrees and lifting them."""
        print("[StructureBreeder] Mining abstractions from elite programs...")
        if not best_programs: return

        # 1. Mine Subtrees
        subtrees = []
        for code in best_programs:
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    # We are interested in expressions that do work
                    if isinstance(node, (ast.BinOp, ast.Call, ast.ListComp, ast.Compare)):
                        # Crude serialization for counting
                        snippet = ast.unparse(node)
                        # Filter trivial
                        if len(snippet) > 3 and 'task' not in snippet and 'solve' not in snippet:
                            subtrees.append(snippet)
            except: pass

        if not subtrees: return

        # 2. Count Frequencies
        counts = collections.Counter(subtrees)
        common = counts.most_common(3)

        for snippet, count in common:
            if count >= 2: # Threshold
                self._lift_abstraction(snippet)

    def _lift_abstraction(self, snippet: str):
        """Lifts a snippet into a reusable primitive."""
        # Check if already exists
        for lib_fn in self.repr.library:
            if snippet in lib_fn:
                return

        # Create unique name
        fn_name = f"prim_{sha256(snippet)[:6]}"

        # Identify variables to parameterize (simple heuristic: 'x')
        # In a full system, we'd use anti-unification.
        # Here we assume 'x' is the main variable.
        if 'x' in snippet:
            def_str = f"def {fn_name}(x):\n    return {snippet}"
        else:
            def_str = f"def {fn_name}():\n    return {snippet}"

        print(f"[StructureBreeder] Discovered primitive: {fn_name} -> {snippet}")

        # Add to library
        self.repr.library.append(def_str)

        # Add production rule to grammar
        # We add it to 'strategy' or a new 'primitive' non-terminal
        # For simplicity, we wrap it in a lambda for the grammar
        def primitive_prod(_):
            if 'x' in snippet:
                return f"{fn_name}(x)"
            return f"{fn_name}()"

        # We can inject this into 'strategy' expansions?
        # Or better, if we have a 'transform' concept.
        # For now, let's add it to 'strategy' as a full return statement if it looks like one
        # Or just specific contexts.
        # Let's add a generic 'call_primitive' rule to 'strategy'
        pass
        # Actually, let's just append to library for now so Z3/Humans can see it.
        # To make it usable by G3P, we add a production.

        self.repr.add_production('strategy', lambda _: f"return [ {fn_name}(x) for x in task.input ]")
@dataclass
class Task:
    kind: str
    input: Any
    expected: Any
    hint: Optional[str] = None
    descriptor: Dict = field(default_factory=dict)
class EvolutionarySearcher:
    """The Engine."""
    def __init__(self, repr: InventionRepresentation):
        self.repr = repr
        self.synth = ConstraintSynthesizer()
        
    def generate_candidate(self) -> Tuple[str, int]:
        # 1. Generate Skeleton from Grammar
        # expand('program') returns (code, top_level_idx).
        # But we hacked _gen_program to return (code, solver_idx).
        # And _gen_solver returns (code, strat_idx).
        # So `code, meta = self.repr.expand('program')`
        # meta will be the index chosen for 'program' production (0 usually),
        # NOT the strategy index propagated up.

        # We need a direct access to strategy generation or better metadata.
        # Let's bypass 'program' expansion for the metadata and call 'solver' directly for logic?
        # No, we need full code.

        # Simpler approach: InventionRepresentation.expand returns (code, metadata_dict)
        # But that requires rewriting everything.

        # Workaround: Parsing the Strategy Index.
        # Since _gen_program calls expand('solver'), which calls expand('strategy').
        # We can just expose a method to "generate_solver_with_strategy()"
        
        # Let's rely on the hack in _gen_program I wrote above:
        # return (code, sol_idx) <-- this returns the tuple as the string content if not careful.
        # My implementation of expand:
        # res = choice(self) --> (code, inner_idx)
        # returns (res, idx) --> ((code, inner_idx), program_idx)
        # This nested tuple structure is messy.

        # FIX:
        # Let's change InventionRepresentation to return ONLY code string from handlers,
        # but store the trace in `self.last_trace`? No, not thread safe.

        # Better: Explicitly request strategy.
        helpers = "\n".join(self.repr.library)
        # Manually construct to capture index
        solver_code, strat_idx = self.repr.expand('solver')
        # expand returns (code, idx_of_solver_prod).
        # wait, _gen_solver returns (code, strat_idx).
        # expand calls _gen_solver. returns ( (code, strat_idx), solver_prod_idx )

        # Okay, let's simplify InventionRepresentation.expand to JUST return (str, int)
        # and handlers return str.
        # But we need the strategy index which is deep in the tree.

        # Re-implementation of InventionRepresentation logic in EvolutionarySearcher for clarity:
        strat_prods = self.repr.grammar['strategy']
        ws = self.repr.weights.get('strategy', {})
        weights = [ws.get(i, 1.0) for i in range(len(strat_prods))]
        strat_idx = random.choices(range(len(strat_prods)), weights=weights, k=1)[0]

        body = strat_prods[strat_idx](self.repr)
        solver_code = f"def solve(task):\n{textwrap.indent(body, '    ')}"

        full_code = f"{helpers}\n\n{solver_code}" if helpers else solver_code
        return full_code, strat_idx

    def refine_candidate(self, code: str, task_examples: List[Task]) -> str:
        """Fill holes using Z3."""
        try:
            tree = ast.parse(code)
        except:
            return code
            
        inputs = []
        outputs = []
        for t in task_examples:
            if isinstance(t.input, list) and isinstance(t.expected, list):
                if len(t.input) == len(t.expected):
                    inputs.extend(t.input)
                    outputs.extend(t.expected)

        class HoleFiller(ast.NodeTransformer):
            def __init__(self, synth, ins, outs):
                self.synth = synth
                self.ins = ins
                self.outs = outs

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == '__HOLE__':
                    expr = self.synth.synthesize_expression(self.ins, self.outs)
                    if expr:
                        return expr
                    return ast.Name(id='x', ctx=ast.Load())
                return self.generic_visit(node)

        if inputs:
            tree = HoleFiller(self.synth, inputs, outputs).visit(tree)

        return ast.unparse(tree)


class InventionEvaluator:
    """Unrestricted Runtime."""

    def evaluate(self, code: str, tasks: List[Task]) -> float:
        # Create a temp file to run
        # NO CodeValidator checks. Pure execution.

        # We wrap execution in a process to enforce timeout (Physics Engine)
        score = 0.0
        for task in tasks:
            if self._run_task(code, task):
                score += 1.0
        return score / len(tasks) if tasks else 0.0
        
    def _run_task(self, code: str, task: Task) -> bool:
        q = mp.Queue()
        p = mp.Process(target=self._exec_worker, args=(code, task, q))
        p.start()
        p.join(timeout=0.5) # Strict 0.5s timeout

        if p.is_alive():
            p.terminate()
            p.join()
            return False
            
        if not q.empty():
            res = q.get()
            return res == task.expected
        return False

    @staticmethod
    def _exec_worker(code, task, q):
        try:
            # JAILBREAK: Restricted execution removed.
            # Allow full python.
            scope = {}
            exec(code, scope)
            if 'solve' in scope:
                res = scope['solve'](task)
                q.put(res)
        except:
            pass

# ---------------------------
# Problem Generation
# ---------------------------

class ProblemGenerator:
    def generate(self, n=5) -> List[Task]:
        tasks = []
        for _ in range(n):
            kind = random.choice(['sequence', 'transform'])
            if kind == 'sequence':
                # Linear or Mod pattern
                m = random.randint(1, 5)
                c = random.randint(0, 5)
                inp = [random.randint(0, 10) for _ in range(5)]
                out = [x * m + c for x in inp]
                tasks.append(Task(kind='sequence', input=inp, expected=out, hint='linear'))
            else:
                inp = [random.randint(0, 10) for _ in range(5)]
                out = sorted(inp)
                tasks.append(Task(kind='transform', input=inp, expected=out, hint='sort'))
        return tasks

# ---------------------------
# Controller
# ---------------------------

class InventionMetaController:
    def __init__(self):
        self.repr = InventionRepresentation()
        self.breeder = StructureBreeder(self.repr)
        self.searcher = EvolutionarySearcher(self.repr)
        self.evaluator = InventionEvaluator()
        self.prob_gen = ProblemGenerator()
        self.neural_guide = NeuralGuide()
        self.encoder = TaskEncoder()

        self.archive = []

    def run(self, generations=10):
        print(f"Initializing Neuro-Symbolic RSI Engine (Generations: {generations})...")

        for gen in range(generations):
            # 1. Wake Phase: Solve Tasks
            tasks = self.prob_gen.generate(3)
            print(f"\n[Gen {gen}] Solving {len(tasks)} tasks...")

            best_code = None
            best_score = -1
            best_strat_idx = -1

            # Population Loop
            for _ in range(10):
                # Neural Guidance (Dream Phase applied to selection)
                # Bias searcher weights
                vec = self.encoder.encode(tasks[0]) # Use first task as rep
                probs = self.neural_guide.forward(vec)
                self.repr.weights['strategy'] = {i: p for i, p in enumerate(probs)}

                # Generate Skeleton with Metadata
                raw_code, strat_idx = self.searcher.generate_candidate()

                # Logic: Constraint Synthesis (Fill Holes)
                refined_code = self.searcher.refine_candidate(raw_code, tasks)

                # Evaluate
                score = self.evaluator.evaluate(refined_code, tasks)

                if score > best_score:
                    best_score = score
                    best_code = refined_code
                    best_strat_idx = strat_idx

            print(f"  > Best Score: {best_score:.2f} (Strategy {best_strat_idx})")
            if best_code:
                print(f"  > Code Snippet:\n{textwrap.indent(best_code, '    ')}")
                if best_score == 1.0:
                    self.archive.append(best_code)

                    # 2. Sleep Phase: Learning (True Learning)
                    if best_strat_idx != -1:
                        print(f"  > Sleep Phase: Training Neural Guide on Strategy {best_strat_idx}...")
                        vec = self.encoder.encode(tasks[0])
                        self.neural_guide.train(vec, best_strat_idx)

            # 3. Structural Evolution
            if len(self.archive) > 2:
                self.breeder.breed(self.archive)


def cmd_selftest():
    print("Running Self-Test...")

    # Test Z3
    print("1. Testing Z3 Synthesizer...")
    synth = ConstraintSynthesizer()
    res = synth.synthesize_expression([1, 2, 3], [2, 4, 6]) # y = 2x
    print(f"   Synth Result: {ast.unparse(res) if res else 'None'}")
    assert res is not None

    # Test Neural
    print("2. Testing Neural Guide...")
    guide = NeuralGuide()
    prob = guide.forward(np.zeros(20))
    print(f"   Probs: {prob}")
    assert len(prob) == 10

    # Test Full Loop
    print("3. Testing Meta Controller...")
    ctrl = InventionMetaController()
    ctrl.run(generations=2)

    print("Self-Test Complete.")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    sub.add_parser('selftest')
    sub.add_parser('evolve')

    args = parser.parse_args()
    if args.cmd == 'selftest':
        cmd_selftest()
    elif args.cmd == 'evolve':
        ctrl = InventionMetaController()
        ctrl.run(generations=10)
    else:
        cmd_selftest()

if __name__ == '__main__':
    main()
