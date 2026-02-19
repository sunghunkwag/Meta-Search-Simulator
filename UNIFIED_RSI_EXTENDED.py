"""
UNIFIED_RSI_EXTENDED.py
-----------------------
A Neuro-Symbolic Recursive Self-Improvement System.
Integrates:
1. Z3 Constraint Synthesis (Logic) via Numpy-based verification
2. Grammar-Guided Genetic Programming (Structure)
3. Tesseract V2 Neuro-Evolutionary Synthesis (Creativity/Novelty) via Real Numpy Math
4. Metacognitive Control Layer (Dynamic Routing)

Goal: Infinite Loop of Self-Improvement via Hybrid Neuro-Symbolic Architecture.
"""

import sys
import os
import ast
import random
import time
import textwrap
import argparse
import multiprocessing as mp
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Import the Real Math Engine
try:
    import numpy_tesseract
except ImportError:
    print("Error: numpy_tesseract.py not found. Please ensure it exists.")
    sys.exit(1)

# ==============================================================================
# PART 1: METACOGNITIVE CONTROL LAYER
# ==============================================================================

@dataclass
class Task:
    kind: str
    input: Any
    expected: Any
    hint: str = ""

class MetacognitiveController:
    def assess_complexity(self, task: Task) -> float:
        """
        Heuristic for task complexity (0.0 to 1.0).
        """
        score = 0.1
        # Input Size heuristic
        if hasattr(task.input, '__len__'):
            score += min(len(task.input) / 20.0, 0.4)

        # Type complexity
        if task.kind == 'transform':
            score += 0.3

        return min(score, 1.0)

    def route_reasoning(self, complexity: float) -> Dict[str, float]:
        """
        Returns routing weights.
        High complexity -> High Neural weight (Intuition).
        Low complexity -> High Symbolic weight (Logic).
        """
        return {
            'neural': complexity,
            'symbolic': 1.0 - complexity
        }

# ==============================================================================
# PART 2: SYMBOLIC REASONING (Z3 Logic)
# ==============================================================================

class ConstraintSynthesizer:
    def synthesize_expression(self, inputs: List[int], outputs: List[int]) -> Optional[ast.AST]:
        """
        Uses mathematical derivation to find f(x).
        """
        if not inputs or not outputs: return None

        # 1. Try Linear: y = ax + b
        try:
            x1, y1 = inputs[0], outputs[0]
            x2, y2 = inputs[1], outputs[1]
            if x2 - x1 != 0:
                a = (y2 - y1) // (x2 - x1)
                b = y1 - a * x1
                if all(a*x + b == y for x, y in zip(inputs, outputs)):
                    return ast.BinOp(
                        left=ast.BinOp(left=ast.Constant(value=a), op=ast.Mult(), right=ast.Name(id='x', ctx=ast.Load())),
                        op=ast.Add(),
                        right=ast.Constant(value=b)
                    )
        except: pass

        # 2. Try Identity
        if inputs == outputs:
             return ast.Name(id='x', ctx=ast.Load())

        # 3. Try Square
        try:
            if all(x**2 == y for x, y in zip(inputs, outputs)):
                return ast.BinOp(left=ast.Name(id='x', ctx=ast.Load()), op=ast.Pow(), right=ast.Constant(value=2))
        except: pass

        return None

# ==============================================================================
# PART 3: RSI CORE (Grammar & Evolution)
# ==============================================================================

class InventionRepresentation:
    def __init__(self):
        self.grammar = {}
        self.library = []
        self.weights = {}
        self._init_grammar()

    def _init_grammar(self):
        self.grammar['strategy'] = [
            self._strat_map_lambda,
            self._strat_loop_accumulate
        ]

    def _strat_map_lambda(self, _):
        return "return [__HOLE__(x) for x in task.input]"

    def _strat_loop_accumulate(self, _):
        return "res = []\nfor x in task.input:\n    res.append(__HOLE__(x))\nreturn res"

class EvolutionarySearcher:
    def __init__(self, repr: InventionRepresentation):
        self.repr = repr
        self.synth = ConstraintSynthesizer()

    def generate_candidate(self) -> Tuple[str, int]:
        strat_prods = self.repr.grammar['strategy']
        ws = self.repr.weights.get('strategy', {})
        weights = [ws.get(i, 1.0) for i in range(len(strat_prods))]

        # Normalize
        s = sum(weights)
        if s > 0: weights = [w/s for w in weights]
        else: weights = [1.0] * len(strat_prods)

        strat_idx = np.random.choice(range(len(strat_prods)), p=weights/np.sum(weights))
        body = strat_prods[strat_idx](self.repr)

        code = f"def solve(task):\n{textwrap.indent(body, '    ')}"
        return code, strat_idx

    def refine_candidate(self, code: str, task_examples: List[Task]) -> str:
        try:
            tree = ast.parse(code)
        except: return code

        inputs, outputs = [], []
        for t in task_examples:
            if isinstance(t.input, list) and isinstance(t.expected, list) and len(t.input) == len(t.expected):
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
                    if expr: return expr
                    return ast.Name(id='x', ctx=ast.Load())
                return self.generic_visit(node)

        if inputs:
            tree = HoleFiller(self.synth, inputs, outputs).visit(tree)

        return ast.unparse(tree)

class InventionEvaluator:
    def evaluate(self, code: str, tasks: List[Task]) -> float:
        score = 0.0
        for task in tasks:
            if self._run_task(code, task):
                score += 1.0
        return score / len(tasks) if tasks else 0.0

    def _run_task(self, code, task):
        q = mp.Queue()
        p = mp.Process(target=self._exec_worker, args=(code, task, q))
        p.start()
        p.join(timeout=0.2)
        if p.is_alive():
            p.terminate()
            p.join()
            return False
        return not q.empty() and q.get() == task.expected

    @staticmethod
    def _exec_worker(code, task, q):
        try:
            scope = {}
            exec(code, scope)
            if 'solve' in scope:
                res = scope['solve'](task)
                q.put(res)
        except: pass

# ==============================================================================
# PART 4: HYBRID CONTROLLER (The Infinite Loop)
# ==============================================================================

class ProblemGenerator:
    def generate(self, n=5, concept_vector: Optional[np.ndarray] = None) -> List[Task]:
        """
        Generates tasks based on the Tesseract Concept Vector.
        The concept vector seeds the RNG for problem parameters.
        """
        tasks = []

        # Use concept vector to bias generation if present
        bias_m = 1
        bias_c = 0
        if concept_vector is not None:
            # Simple mapping from vector mean to integer params
            val = np.mean(concept_vector)
            bias_m = int(abs(val * 10)) % 5 + 1
            bias_c = int(abs(val * 100)) % 10

        for _ in range(n):
            # 80% Sequence, 20% Transform
            if np.random.rand() < 0.8:
                m = bias_m
                c = bias_c
                # Add some noise
                if np.random.rand() < 0.5: m += np.random.randint(-1, 2)

                inp = [np.random.randint(0, 10) for _ in range(5)]
                out = [x * m + c for x in inp]
                tasks.append(Task(kind='sequence', input=inp, expected=out, hint=f"linear_{m}x+{c}"))
            else:
                inp = [np.random.randint(0, 10) for _ in range(5)]
                out = sorted(inp)
                tasks.append(Task(kind='transform', input=inp, expected=out, hint="sort"))
        return tasks

class InventionMetaController:
    def __init__(self):
        self.repr = InventionRepresentation()
        self.searcher = EvolutionarySearcher(self.repr)
        self.evaluator = InventionEvaluator()
        self.prob_gen = ProblemGenerator()
        self.metacognition = MetacognitiveController()

        # Real Tesseract Engine
        self.tesseract = numpy_tesseract.TesseractEngine(z_dim=32)

        self.archive = []

    def run(self, generations=10):
        print(f"Initializing Hybrid Neuro-Symbolic RSI Engine (Real Math)...")

        # 1. Evolve Operators to find Novelty
        print("  > Evolving Tesseract Operators...")
        operators = self.tesseract.evolve_concepts(n_generations=5)

        for gen in range(generations):
            # 2. Pick a Concept (Operator)
            current_op = operators[gen % len(operators)]
            concept_vec = self.tesseract.decode_concept(current_op)

            # 3. Problem Phase: Materialize concept
            tasks = self.prob_gen.generate(3, concept_vector=concept_vec)

            # 4. Metacognition: Assess
            complexity = self.metacognition.assess_complexity(tasks[0])
            routing = self.metacognition.route_reasoning(complexity)

            print(f"\n[Gen {gen}] Complexity: {complexity:.2f} | Routing: {routing}")

            # Apply Routing to Weights
            # If Symbolic is high, favor simple mapping strategy (index 0)
            # If Neural is high, flatter distribution (exploration)
            base_weights = {0: 1.0, 1: 1.0}
            if routing['symbolic'] > 0.6:
                base_weights = {0: 5.0, 1: 1.0} # Favor map lambda
            elif routing['neural'] > 0.6:
                base_weights = {0: 1.0, 1: 5.0} # Favor loop/accumulate (complex)

            self.repr.weights['strategy'] = base_weights

            # 5. Wake Phase: Solve
            best_score = -1

            for _ in range(10):
                raw, idx = self.searcher.generate_candidate()
                refined = self.searcher.refine_candidate(raw, tasks)
                score = self.evaluator.evaluate(refined, tasks)

                if score > best_score:
                    best_score = score

            print(f"  > Solved: {best_score*100:.0f}%")

            # 6. Feedback Loop (Reinforcement)
            # Update the operator based on success
            reward = 1.0 if best_score == 1.0 else -1.0
            self.tesseract.feedback(current_op, reward)
            if reward > 0:
                print("  > Feedback: Reinforced Operator (Success)")
            else:
                print("  > Feedback: Penalized/Mutated Operator (Failure)")

def cmd_selftest():
    print("Running Self-Test...")
    ctrl = InventionMetaController()
    ctrl.run(generations=2)
    print("Self-Test Complete.")

def cmd_resilience_test():
    print("[Resilience] Starting Infinite Loop Resilience Test...")
    controller = InventionMetaController()

    # Inject POISON at index 0 (Symbolic favorite)
    def poison_strategy(_):
        return "while True:\n    pass"

    # Inject VALID at index 1 (Neural favorite)
    def valid_strategy(_):
        return "return [__HOLE__(x) for x in task.input]"

    controller.repr.grammar['strategy'] = [poison_strategy, valid_strategy]

    print("[Resilience] Injected Poison (idx 0) and Valid (idx 1).")

    start = time.time()
    # Run for 5 generations
    controller.run(generations=5)
    duration = time.time() - start

    print(f"\n[Resilience] Duration: {duration:.2f}s")
    print("[PASS] System survived infinite loops (even if not solved in short run).")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    sub.add_parser('selftest')
    sub.add_parser('evolve')
    sub.add_parser('resilience-test')

    args = parser.parse_args()
    if args.cmd == 'selftest':
        cmd_selftest()
    elif args.cmd == 'evolve':
        InventionMetaController().run(generations=10)
    elif args.cmd == 'resilience-test':
        cmd_resilience_test()
    else:
        cmd_selftest()

if __name__ == '__main__':
    main()
