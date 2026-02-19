"""
UNIFIED_RSI_EXTENDED.py
-----------------------
A Neuro-Symbolic Recursive Self-Improvement System.
Integrates:
1. Constraint Synthesis (Logic) via Numpy-based verification
2. Grammar-Guided Genetic Programming (Structure)
3. Tesseract V2 Neuro-Evolutionary Synthesis (Creativity/Novelty) via Real Numpy Math
4. Metacognitive Control Layer (Dynamic Routing)
5. Performance-Based Self-Improvement (Real RSI via Hill-Climbing)
6. Void-Logic Expansion (Non-Human-Readable Structures)

Goal: Infinite Loop of Self-Improvement via Hybrid Neuro-Symbolic Architecture.

Changelog:
  [FIX-01] archive now populated with successful solutions in run()
  [FIX-02] SelfImprover: backup first, validate syntax before overwrite, restore on error
  [FIX-03] ConstraintSynthesizer: modulo synthesis added (difficulty >= 3 support)
  [FIX-04] ConstraintSynthesizer: identity check now precedes linear (dead-code fixed)
  [FIX-07] ProblemGenerator: base_m fixed per generate() call (BLOCKER-A)
  [FIX-08] PerformanceSelfImprover: real hill-climbing RSI, replaces random mutation
  [FIX-09] ConstraintSynthesizer: quadratic ax^2+bx+c synthesis (difficulty 2)
  [FIX-10] InventionEvaluator: configurable timeout
  [FIX-11] TesseractEngine.feedback: lr parameter exposed for RSI control
  [PHASE-02] Void-Logic Synthesis: added synthesize_void_expression and RSI controls
"""

import sys
import os
import ast
import random
import time
import textwrap
import argparse
import multiprocessing as mp
import shutil
import pickle
import math
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
# UTILITY: CODE COMPLEXITY METRIC
# ==============================================================================

def code_complexity_metric(code: str) -> float:
    """
    Returns a scalar complexity score based on:
      - AST depth
      - total number of nodes
      - number of distinct operator types

    Higher = more structurally complex (harder for humans to parse).
    """
    try:
        tree = ast.parse(code)
    except:
        return 0.0

    nodes = 0
    max_depth = 0
    ops = set()

    def visit(node, depth):
        nonlocal nodes, max_depth, ops
        nodes += 1
        max_depth = max(max_depth, depth)

        # Collect operator types
        if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
                             ast.USub, ast.UAdd, ast.BitXor, ast.BitAnd, ast.BitOr)):
            ops.add(type(node))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                ops.add(node.func.id)  # Function names as "ops"

        for child in ast.iter_child_nodes(node):
            visit(child, depth + 1)

    visit(tree, 1)

    # Score calculation
    # Normalized roughly: depth(1-10), nodes(1-50), ops(1-5)
    score = max_depth + 0.1 * nodes + 0.5 * len(ops)
    return score

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
        if hasattr(task.input, '__len__'):
            score += min(len(task.input) / 20.0, 0.4)
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
# PART 2: SYMBOLIC REASONING (Constraint Synthesis)
# ==============================================================================

class ConstraintSynthesizer:
    def synthesize_expression(self, inputs: List[int], outputs: List[int]) -> Optional[ast.AST]:
        """
        Uses mathematical derivation to find f(x).
        Supports: identity, linear (ax+b), square (x^2),
                  quadratic (ax^2+bx+c) [FIX-09], modulo ((a*x)%m) [FIX-03].
        """
        if not inputs or not outputs:
            return None

        # [FIX-04] Check Identity FIRST
        if inputs == outputs:
            return ast.Name(id='x', ctx=ast.Load())

        # 1. Try Linear: y = ax + b
        try:
            x1, y1 = inputs[0], outputs[0]
            x2, y2 = inputs[1], outputs[1]
            if x2 - x1 != 0:
                a = (y2 - y1) // (x2 - x1)
                b = y1 - a * x1
                if all(a * x + b == y for x, y in zip(inputs, outputs)):
                    return ast.BinOp(
                        left=ast.BinOp(
                            left=ast.Constant(value=a),
                            op=ast.Mult(),
                            right=ast.Name(id='x', ctx=ast.Load())
                        ),
                        op=ast.Add(),
                        right=ast.Constant(value=b)
                    )
        except:
            pass

        # 2. Try Square: y = x^2
        try:
            if all(x ** 2 == y for x, y in zip(inputs, outputs)):
                return ast.BinOp(
                    left=ast.Name(id='x', ctx=ast.Load()),
                    op=ast.Pow(),
                    right=ast.Constant(value=2)
                )
        except:
            pass

        # [FIX-09] Try Quadratic: y = a*x^2 + b*x + c  (difficulty 2 support)
        try:
            if len(inputs) >= 3:
                x0, y0 = inputs[0], outputs[0]
                x1, y1 = inputs[1], outputs[1]
                x2, y2 = inputs[2], outputs[2]
                A = np.array(
                    [[x0**2, x0, 1],
                     [x1**2, x1, 1],
                     [x2**2, x2, 1]], dtype=float
                )
                b_vec = np.array([y0, y1, y2], dtype=float)
                if abs(np.linalg.det(A)) > 1e-6:
                    coeffs = np.linalg.solve(A, b_vec)
                    a_c = int(round(coeffs[0]))
                    b_c = int(round(coeffs[1]))
                    c_c = int(round(coeffs[2]))
                    if all(a_c * x**2 + b_c * x + c_c == y for x, y in zip(inputs, outputs)):
                        # Build AST: a*x**2 + b*x + c
                        quad = ast.BinOp(
                            left=ast.BinOp(
                                left=ast.Constant(value=a_c),
                                op=ast.Mult(),
                                right=ast.BinOp(
                                    left=ast.Name(id='x', ctx=ast.Load()),
                                    op=ast.Pow(),
                                    right=ast.Constant(value=2)
                                )
                            ),
                            op=ast.Add(),
                            right=ast.BinOp(
                                left=ast.BinOp(
                                    left=ast.Constant(value=b_c),
                                    op=ast.Mult(),
                                    right=ast.Name(id='x', ctx=ast.Load())
                                ),
                                op=ast.Add(),
                                right=ast.Constant(value=c_c)
                            )
                        )
                        return quad
        except:
            pass

        # [FIX-03] Try Modulo: y = (a*x) % m  (difficulty 3+ support)
        try:
            for a in range(1, 11):
                for m in range(2, 21):
                    if all((x * a) % m == y for x, y in zip(inputs, outputs)):
                        return ast.BinOp(
                            left=ast.BinOp(
                                left=ast.Constant(value=a),
                                op=ast.Mult(),
                                right=ast.Name(id='x', ctx=ast.Load())
                            ),
                            op=ast.Mod(),
                            right=ast.Constant(value=m)
                        )
        except:
            pass

        return None

    def synthesize_void_expression(
        self,
        inputs: List[float],
        outputs: List[float]
    ) -> Optional[ast.AST]:
        """
        Synthesizes 'void' expressions: non-trivial, non-linear, and
        not obviously representable as identity/linear/quadratic/modulo.

        Objectives (in this order):
          1. Minimize prediction error on (inputs, outputs).
          2. Maximize structural complexity (AST depth and node count).
          3. Avoid collapsing to existing human-friendly patterns
             (identity, linear, quadratic, modulo).
        """
        best_ast = None
        best_mse = float('inf')
        best_complexity = -1.0

        # Operator pool for Void Logic
        binary_ops = [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow]
        unary_funcs = ['sin', 'cos', 'tanh', 'exp', 'log', 'abs']

        # Max attempts to find a void candidate
        attempts = 50

        for _ in range(attempts):
            # Randomly generate an AST
            try:
                candidate_ast = self._generate_random_ast(depth=random.randint(2, 5))
                # Fix locations
                ast.fix_missing_locations(candidate_ast)

                # Evaluate MSE
                mse = self._evaluate_ast_mse(candidate_ast, inputs, outputs)
                if mse is None: continue

                # Calculate Complexity
                complexity = 0
                for node in ast.walk(candidate_ast):
                    complexity += 1

                # Selection Criteria
                # 1. Low MSE (allow small float error due to transcendental functions)
                if mse < 1e-5:
                    # 2. Check if it's trivially reducible (simple heuristic)
                    if self._is_trivial(candidate_ast, inputs, outputs):
                        continue

                    # 3. Maximize complexity
                    if mse < best_mse or (abs(mse - best_mse) < 1e-9 and complexity > best_complexity):
                        best_mse = mse
                        best_complexity = complexity
                        best_ast = candidate_ast

            except Exception:
                continue

        return best_ast

    def _generate_random_ast(self, depth):
        if depth <= 0 or random.random() < 0.3:
            # Leaf: variable 'x' or constant
            if random.random() < 0.6:
                return ast.Name(id='x', ctx=ast.Load())
            else:
                return ast.Constant(value=random.randint(-5, 5))

        op_type = random.choice(['binary', 'unary', 'ternary'])

        if op_type == 'binary':
            op_cls = random.choice([ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow])
            left = self._generate_random_ast(depth - 1)
            right = self._generate_random_ast(depth - 1)
            return ast.BinOp(left=left, op=op_cls(), right=right)

        elif op_type == 'unary':
            func_name = random.choice(['sin', 'cos', 'tanh', 'exp', 'log', 'abs'])
            arg = self._generate_random_ast(depth - 1)
            return ast.Call(
                func=ast.Name(id=func_name, ctx=ast.Load()),
                args=[arg], keywords=[]
            )

        elif op_type == 'ternary': # if x > c else
             test_val = random.randint(-5, 5)
             test = ast.Compare(
                 left=ast.Name(id='x', ctx=ast.Load()),
                 ops=[ast.Gt()],
                 comparators=[ast.Constant(value=test_val)]
             )
             body = self._generate_random_ast(depth - 1)
             orelse = self._generate_random_ast(depth - 1)
             return ast.IfExp(test=test, body=body, orelse=orelse)

        return ast.Name(id='x', ctx=ast.Load())

    def _evaluate_ast_mse(self, tree, inputs, outputs):
        # Wrap in lambda x: ...
        code_obj = compile(ast.Expression(body=tree), filename="<string>", mode="eval")

        mse_sum = 0
        count = 0
        try:
            # Prepare safe math context
            safe_dict = {
                'sin': math.sin, 'cos': math.cos, 'tanh': math.tanh,
                'exp': math.exp, 'log': lambda x: math.log(x) if x > 0 else 0,
                'abs': abs
            }

            for x_val, y_true in zip(inputs, outputs):
                safe_dict['x'] = x_val
                y_pred = eval(code_obj, {"__builtins__": {}}, safe_dict)

                # Check for NaN/Inf
                if not isinstance(y_pred, (int, float)) or math.isnan(y_pred) or math.isinf(y_pred):
                    return None

                mse_sum += (y_pred - y_true) ** 2
                count += 1

            return mse_sum / count if count > 0 else float('inf')
        except:
            return None

    def _is_trivial(self, tree, inputs, outputs):
        # Heuristic: Check if linear regression fits perfectly
        try:
            # Linear check
            x1, y1 = inputs[0], outputs[0]
            x2, y2 = inputs[1], outputs[1]
            if x2 != x1:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
                is_linear = True
                for x, y in zip(inputs, outputs):
                    if abs((m * x + c) - y) > 1e-5:
                        is_linear = False
                        break
                if is_linear: return True

            # Identity check
            if all(x == y for x, y in zip(inputs, outputs)):
                return True

        except:
            pass
        return False

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

        s = sum(weights)
        if s > 0:
            weights = [w / s for w in weights]
        else:
            weights = [1.0] * len(strat_prods)

        w_arr = np.array(weights)
        strat_idx = np.random.choice(range(len(strat_prods)), p=w_arr / np.sum(w_arr))
        body = strat_prods[strat_idx](self.repr)

        code = f"def solve(task):\n{textwrap.indent(body, '    ')}"
        return code, strat_idx

    def refine_candidate(self, code: str, task_examples: List[Task], use_void: bool = False) -> str:
        try:
            tree = ast.parse(code)
        except:
            return code

        inputs, outputs = [], []
        for t in task_examples:
            if isinstance(t.input, list) and isinstance(t.expected, list) and len(t.input) == len(t.expected):
                inputs.extend(t.input)
                outputs.extend(t.expected)

        class HoleFiller(ast.NodeTransformer):
            def __init__(self, synth, ins, outs, use_void: bool = False):
                self.synth = synth
                self.ins = ins
                self.outs = outs
                self.use_void = use_void

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == '__HOLE__':
                    if self.use_void:
                        expr = self.synth.synthesize_void_expression(self.ins, self.outs)
                    else:
                        expr = self.synth.synthesize_expression(self.ins, self.outs)

                    if expr:
                        return expr
                    return ast.Name(id='x', ctx=ast.Load())
                return self.generic_visit(node)

        if inputs:
            tree = HoleFiller(self.synth, inputs, outputs, use_void=use_void).visit(tree)

        return ast.unparse(tree)

class InventionEvaluator:
    def __init__(self, timeout: float = 0.2):
        # [FIX-10] Configurable timeout, controlled by PerformanceSelfImprover at runtime
        self.timeout = timeout

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
        p.join(timeout=self.timeout)  # [FIX-10] Now adaptive
        if p.is_alive():
            p.terminate()
            p.join()
            return False
        return not q.empty() and q.get() == task.expected

    @staticmethod
    def _exec_worker(code, task, q):
        try:
            scope = {}
            # Inject math imports for Void Logic
            import math
            scope['sin'] = math.sin
            scope['cos'] = math.cos
            scope['tanh'] = math.tanh
            scope['exp'] = math.exp
            scope['log'] = math.log

            exec(code, scope)
            if 'solve' in scope:
                res = scope['solve'](task)
                q.put(res)
        except:
            pass

# ==============================================================================
# PART 4: REAL RSI — Performance-Based Self-Improvement
# ==============================================================================

class PerformanceSelfImprover:
    """
    Real performance-based Recursive Self-Improvement.

    Implements gradient-free hill-climbing over the system's own hyperparameters.
    Unlike the previous random constant mutation approach, every parameter change
    is evaluated against measured avg_success and REVERTED if it causes a
    performance drop >= 5%.

    Tunable parameters:
      - num_candidates:        How many code candidates per generation (search depth)
      - task_timeout:          Subprocess execution timeout in seconds
      - feedback_lr:           TesseractEngine operator learning rate
      - curriculum_threshold:  avg_success required to increase difficulty [FIX-08]
      - void_preference:       Probability to use Void-Logic synthesis [PHASE-02]

    [FIX-08] Replaces SelfImprover (random AST constant mutation) with
    performance-guided parameter optimization. curriculum_threshold default
    reduced from 0.9 to 0.75 to allow auto-curriculum to actually progress.
    """

    PARAM_BOUNDS = {
        'num_candidates':       (5,   50,   1),
        'task_timeout':         (0.1,  2.0,  0.05),
        'feedback_lr':          (0.01, 0.3,  0.01),
        'curriculum_threshold': (0.5,  0.95, 0.05),
        'void_preference':      (0.0,  1.0,  0.05), # [PHASE-02]
    }

    def __init__(self):
        self.params = {
            'num_candidates':       10,
            'task_timeout':         0.2,
            'feedback_lr':          0.05,
            'curriculum_threshold': 0.75,  # [FIX-08] was 0.9 -> now 0.75
            'void_preference':      0.0,   # [PHASE-02] default 0%
        }
        self._pending_change = None    # (param_name, old_val, new_val)
        self._pre_change_perf = None   # avg_success snapshot before change
        self.improvement_log: List[Dict] = []  # Full audit trail

    def step(self, avg_success: float, avg_complexity: float = 0.0) -> Tuple[Optional[str], Optional[str]]:
        """
        Called every N generations.
        Phase 1: Evaluate pending change (keep or revert).
        Phase 2: Propose a new parameter change.
        Returns (eval_msg, proposal_msg).
        """
        eval_msg = None

        # Phase 1: Evaluate pending change
        if self._pending_change is not None:
            name, old_val, new_val = self._pending_change
            delta = avg_success - self._pre_change_perf

            if delta >= -0.05:  # Keep if not significantly worse (5% tolerance)
                action = 'KEPT'
            else:
                self.params[name] = old_val  # Revert
                action = 'REVERTED'

            self.improvement_log.append({
                'param': name,
                'old': old_val,
                'new': new_val,
                'pre_perf': round(self._pre_change_perf, 3),
                'post_perf': round(avg_success, 3),
                'delta': round(delta, 3),
                'action': action
            })
            eval_msg = f"[RSI-EVAL] {action:8s} {name:<22} {old_val} -> {new_val}  (delta={delta:+.3f})"
            self._pending_change = None
            self._pre_change_perf = None

        # Phase 2: Propose new change, guided by current performance AND complexity
        # Low performance: focus on search capacity
        # High performance: fine-tune learning dynamics and curriculum
        # High performance but Low Complexity: increase Void Preference

        priority = list(self.PARAM_BOUNDS.keys())

        if avg_success < 0.4:
            priority = ['num_candidates', 'task_timeout']
        elif avg_success > 0.8 and avg_complexity < 5.0:
            # Success is high, but code is too simple (human-like). Push Void.
            priority = ['void_preference']
        elif avg_success < 0.7:
             priority = ['num_candidates', 'feedback_lr', 'curriculum_threshold']

        name = random.choice(priority)
        lo, hi, step_size = self.PARAM_BOUNDS[name]
        current = self.params[name]

        direction = 1 if random.random() < 0.5 else -1
        new_val = round(current + direction * step_size, 4)
        new_val = max(lo, min(hi, new_val))

        proposal_msg = None
        if new_val != current:
            self._pending_change = (name, current, new_val)
            self._pre_change_perf = avg_success
            self.params[name] = new_val
            proposal_msg = f"[RSI-PROP] PROPOSE {name:<22} {current} -> {new_val}  (perf={avg_success:.3f}, cmplx={avg_complexity:.1f})"
        else:
            proposal_msg = f"[RSI-PROP] {name} at boundary ({current}), skipping"

        return eval_msg, proposal_msg

    def get(self, key: str, default=None):
        return self.params.get(key, default)

    def summary(self) -> str:
        lines = ["[RSI] Current Hyperparameters:"]
        for k, v in self.params.items():
            lo, hi, _ = self.PARAM_BOUNDS[k]
            lines.append(f"  {k:<30} = {v}  (range: [{lo}, {hi}])")
        if self.improvement_log:
            kept = sum(1 for e in self.improvement_log if e['action'] == 'KEPT')
            reverted = len(self.improvement_log) - kept
            lines.append(f"  changes applied: {len(self.improvement_log)}  kept: {kept}  reverted: {reverted}")
        return "\n".join(lines)

# ==============================================================================
# PART 5: HYBRID CONTROLLER (The Infinite Loop)
# ==============================================================================

class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.history = []

    def save(self, controller, generation, score):
        filename = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{generation}.pkl")
        state = {
            "weights": controller.repr.weights,
            "grammar": controller.repr.grammar,
            "difficulty": controller.difficulty,
            "success_history": controller.success_history,
            "rsi_params": controller.rsi.params,
            "rsi_log": controller.rsi.improvement_log,
            "generation": generation,
            "score": score,
            "complexity_history": controller.complexity_history # [PHASE-02]
        }
        try:
            with open(filename, "wb") as f:
                pickle.dump(state, f)

            src_backup = os.path.join(self.checkpoint_dir, f"source_gen_{generation}.py")
            shutil.copy(__file__, src_backup)

            self.history.append({'gen': generation, 'score': score, 'file': filename, 'src': src_backup})
            if len(self.history) > 5:
                oldest = self.history.pop(0)
                if os.path.exists(oldest['file']):
                    os.remove(oldest['file'])
                if os.path.exists(oldest['src']):
                    os.remove(oldest['src'])
        except Exception as e:
            print(f"[Checkpoint] Error saving: {e}")

    def load_best(self):
        if not self.history:
            return None
        best = sorted(self.history, key=lambda x: x['score'], reverse=True)[0]
        try:
            with open(best['file'], "rb") as f:
                return pickle.load(f)
        except:
            return None

    def rollback_source(self):
        if not self.history:
            return False
        best = sorted(self.history, key=lambda x: x['score'], reverse=True)[0]
        try:
            shutil.copy(best['src'], __file__)
            print(f"[Rollback] Restored source code from generation {best['gen']}")
            return True
        except:
            return False

class ProblemGenerator:
    def generate(self, n=5, concept_vector: Optional[np.ndarray] = None, difficulty: int = 1) -> List[Task]:
        """
        Generates tasks based on the Tesseract Concept Vector and Difficulty Level.

        [FIX-07] BLOCKER-A fix: for difficulty==1, base_m (slope) is computed
        ONCE per generate() call instead of per-task.
        Previously, random m offsets were applied independently to each task,
        causing HoleFiller to receive contradictory input/output pairs and
        fail to synthesize any expression (success rate ~5%).
        After fix: all tasks in one generation share the same slope.
        """
        tasks = []
        bias_m = 1
        bias_c = 0
        if concept_vector is not None:
            val = np.mean(concept_vector)
            bias_m = int(abs(val * 10)) % 5 + 1
            bias_c = int(abs(val * 100)) % 10

        if difficulty == 1:
            # [FIX-07] Decide base_m ONCE — consistent across all n tasks
            base_m = bias_m
            if np.random.rand() < 0.5:
                base_m += np.random.randint(-1, 2)
            base_m = max(1, base_m)

            for _ in range(n):
                inp = [np.random.randint(0, 10) for _ in range(5)]
                out = [x * base_m + bias_c for x in inp]
                tasks.append(Task(kind='sequence', input=inp, expected=out, hint=f"linear_{base_m}x+{bias_c}"))

        elif difficulty == 2:
            a = np.random.randint(1, 3)
            for _ in range(n):
                inp = [np.random.randint(0, 10) for _ in range(5)]
                out = [a * x ** 2 + bias_m * x + bias_c for x in inp]
                tasks.append(Task(kind='sequence', input=inp, expected=out, hint=f"quad_{a}x^2+{bias_m}x+{bias_c}"))

        elif difficulty >= 3:
            m = max(2, bias_m)
            mod = max(2, bias_c + 3)
            for _ in range(n):
                inp = [np.random.randint(0, 20) for _ in range(5)]
                out = [(x * m) % mod for x in inp]
                tasks.append(Task(kind='sequence', input=inp, expected=out, hint=f"modulo_{m}x%{mod}"))

        return tasks

class InventionMetaController:
    def __init__(self):
        self.repr = InventionRepresentation()
        self.searcher = EvolutionarySearcher(self.repr)
        self.evaluator = InventionEvaluator(timeout=0.2)
        self.prob_gen = ProblemGenerator()
        self.metacognition = MetacognitiveController()
        self.tesseract = numpy_tesseract.TesseractEngine(z_dim=32)
        self.archive = []  # [FIX-01] Populated in run() when best_score >= 1.0

        # [FIX-08] Real RSI: PerformanceSelfImprover replaces SelfImprover
        self.rsi = PerformanceSelfImprover()
        self.checkpoint_manager = CheckpointManager()
        self.difficulty = 1
        self.success_history = []
        self.complexity_history = [] # [PHASE-02]

    def run(self, generations=10):
        print(f"Initializing Hybrid Neuro-Symbolic RSI Engine (Real Math)...")
        print("  > Evolving Tesseract Operators...")
        operators = self.tesseract.evolve_concepts(n_generations=5)

        gen = 0
        while True:
            current_op = operators[gen % len(operators)]
            concept_vec = self.tesseract.decode_concept(current_op)

            tasks = self.prob_gen.generate(3, concept_vector=concept_vec, difficulty=self.difficulty)

            complexity = self.metacognition.assess_complexity(tasks[0])
            routing = self.metacognition.route_reasoning(complexity)

            print(f"\n[Gen {gen}] Diff: {self.difficulty} | Complexity: {complexity:.2f} | Routing: {routing}")

            base_weights = {0: 1.0, 1: 1.0}
            if routing['symbolic'] > 0.6:
                base_weights = {0: 5.0, 1: 1.0}
            elif routing['neural'] > 0.6:
                base_weights = {0: 1.0, 1: 5.0}
            self.repr.weights['strategy'] = base_weights

            # [FIX-08] Adaptive search depth via RSI
            num_candidates = int(self.rsi.get('num_candidates', 10))

            best_score = -1
            best_code = None  # [FIX-01]

            # [PHASE-02] Determine Void Preference
            void_pref = float(self.rsi.get('void_preference', 0.0))

            for _ in range(num_candidates):
                raw, idx = self.searcher.generate_candidate()

                # [PHASE-02] Probabilistic Void Synthesis
                use_void = random.random() < void_pref

                refined = self.searcher.refine_candidate(raw, tasks, use_void=use_void)
                score = self.evaluator.evaluate(refined, tasks)
                if score > best_score:
                    best_score = score
                    best_code = refined  # [FIX-01]

            print(f"  > Solved: {best_score * 100:.0f}%  (candidates: {num_candidates}, void_pref: {void_pref:.2f})")

            # [FIX-01] Populate archive
            if best_score >= 1.0 and best_code is not None:
                self.archive.append({
                    'gen': gen,
                    'code': best_code,
                    'score': best_score,
                    'difficulty': self.difficulty,
                })
                # [PHASE-02] Metric tracking
                code_complexity = code_complexity_metric(best_code)
                self.complexity_history.append(code_complexity)
                if len(self.complexity_history) > 20:
                     self.complexity_history.pop(0)

                print(f"  > [ARCHIVE] Solution stored. Total archived: {len(self.archive)} (Complexity: {code_complexity:.2f})")

            # Feedback to TesseractEngine with adaptive lr [FIX-11]
            reward = 1.0 if best_score == 1.0 else -1.0
            feedback_lr = float(self.rsi.get('feedback_lr', 0.05))
            self.tesseract.feedback(current_op, reward, lr=feedback_lr)
            if reward > 0:
                print("  > Feedback: Reinforced Operator (Success)")
            else:
                print("  > Feedback: Penalized/Mutated Operator (Failure)")

            # Auto-Curriculum with adaptive threshold [FIX-08]
            self.success_history.append(best_score)
            if len(self.success_history) > 20:
                self.success_history.pop(0)

            avg_success = sum(self.success_history) / len(self.success_history)
            curriculum_threshold = float(self.rsi.get('curriculum_threshold', 0.75))

            if avg_success > curriculum_threshold and len(self.success_history) >= 10:
                self.difficulty += 1
                self.success_history = []
                print(f"  >>> [AUTO-CURRICULUM] Increasing Difficulty to {self.difficulty}")

            elif avg_success < 0.2 and self.difficulty > 1 and len(self.success_history) >= 10:
                self.difficulty -= 1
                self.success_history = []
                print(f"  >>> [AUTO-CURRICULUM] Decreasing Difficulty to {self.difficulty}")

            # [FIX-08] RSI step: hill-climbing over hyperparameters every 5 gens
            # [PHASE-02] Pass complexity
            if gen > 0 and gen % 5 == 0:
                avg_complexity = sum(self.complexity_history) / len(self.complexity_history) if self.complexity_history else 0.0
                eval_msg, proposal_msg = self.rsi.step(avg_success, avg_complexity)
                if eval_msg:
                    print(f"  {eval_msg}")
                if proposal_msg:
                    print(f"  {proposal_msg}")

                # Sync evaluator timeout from RSI params
                self.evaluator.timeout = float(self.rsi.get('task_timeout', 0.2))

            # Checkpointing every 10 gens
            if gen > 0 and gen % 10 == 0:
                self.checkpoint_manager.save(self, gen, avg_success)
                print(f"  [Checkpoint] Saved gen {gen} | RSI params: {self.rsi.params}")

            # Collapse Management
            if len(self.success_history) >= 20 and avg_success == 0.0:
                print("  !!! [COLLAPSE DETECTED] Initiating Rollback !!!")
                if self.checkpoint_manager.rollback_source():
                    state = self.checkpoint_manager.load_best()
                    if state:
                        self.repr.weights = state['weights']
                        self.difficulty = state['difficulty']
                        self.rsi.params = state.get('rsi_params', self.rsi.params)
                        self.success_history = []
                        self.complexity_history = state.get('complexity_history', []) # [PHASE-02]
                        print("  !!! [ROLLBACK] State Restored.")

            gen += 1
            if generations > 0 and gen >= generations:
                break

        print("\n" + self.rsi.summary())

def cmd_selftest():
    print("Running Self-Test...")
    ctrl = InventionMetaController()
    ctrl.run(generations=2)
    print("Self-Test Complete.")

def cmd_resilience_test():
    print("[Resilience] Starting Infinite Loop Resilience Test...")
    controller = InventionMetaController()

    def poison_strategy(_):
        return "while True:\n    pass"

    def valid_strategy(_):
        return "return [__HOLE__(x) for x in task.input]"

    controller.repr.grammar['strategy'] = [poison_strategy, valid_strategy]
    print("[Resilience] Injected Poison (idx 0) and Valid (idx 1).")

    start = time.time()
    controller.run(generations=5)
    duration = time.time() - start

    solved = len(controller.archive) > 0
    print(f"\n[Resilience] Duration: {duration:.2f}s")
    print(f"[Resilience] Solved (archive non-empty): {solved}")
    print("[PASS] System survived infinite loops.")

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
        InventionMetaController().run(generations=0)
    elif args.cmd == 'resilience-test':
        cmd_resilience_test()
    else:
        cmd_selftest()

if __name__ == '__main__':
    main()
