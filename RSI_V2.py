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
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

try:
    import numpy_tesseract
except ImportError:
    print("Error: numpy_tesseract.py not found")
    sys.exit(1)

from population_optimizer import PopulationOptimizer, Individual
from grammar_evolution import GrammarEvolutionEngine, StrategyGene
from arc_benchmark import ARCBenchmark, ARCTask

@dataclass
class Task:
    kind: str
    input: Any
    expected: Any
    hint: str = ""

class MetacognitiveController:
    def assess_complexity(self, task: Task) -> float:
        score = 0.1
        if hasattr(task.input, '__len__'):
            score += min(len(task.input) / 20.0, 0.4)
        if task.kind == 'transform':
            score += 0.3
        return min(score, 1.0)

    def route_reasoning(self, complexity: float) -> Dict[str, float]:
        return {'neural': complexity, 'symbolic': 1.0 - complexity}

class ConstraintSynthesizer:
    def __init__(self, synthesis_order: List[str] = None):
        self.synthesis_order = synthesis_order or ['identity', 'linear', 'quadratic', 'modulo']
    
    def synthesize_expression(self, inputs: List[int], outputs: List[int]) -> Optional[ast.AST]:
        if not inputs or not outputs:
            return None
        
        for method in self.synthesis_order:
            result = self._try_method(method, inputs, outputs)
            if result is not None:
                return result
        return None
    
    def _try_method(self, method: str, inputs: List[int], outputs: List[int]) -> Optional[ast.AST]:
        if method == 'identity':
            if inputs == outputs:
                return ast.Name(id='x', ctx=ast.Load())
        
        elif method == 'linear':
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
        
        elif method == 'quadratic' and len(inputs) >= 3:
            try:
                x0, y0 = inputs[0], outputs[0]
                x1, y1 = inputs[1], outputs[1]
                x2, y2 = inputs[2], outputs[2]
                A = np.array([[x0**2, x0, 1], [x1**2, x1, 1], [x2**2, x2, 1]], dtype=float)
                b_vec = np.array([y0, y1, y2], dtype=float)
                if abs(np.linalg.det(A)) > 1e-6:
                    coeffs = np.linalg.solve(A, b_vec)
                    a_c = int(round(coeffs[0]))
                    b_c = int(round(coeffs[1]))
                    c_c = int(round(coeffs[2]))
                    if all(a_c * x**2 + b_c * x + c_c == y for x, y in zip(inputs, outputs)):
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
        
        elif method == 'modulo':
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

class EvolutionarySearcher:
    def __init__(self, grammar_engine: GrammarEvolutionEngine, synthesis_order: List[str]):
        self.grammar_engine = grammar_engine
        self.synth = ConstraintSynthesizer(synthesis_order=synthesis_order)

    def generate_candidate(self, grammar_weights: Dict[str, float]) -> Tuple[str, int]:
        strategies = self.grammar_engine.get_best_strategies(top_k=min(5, len(self.grammar_engine.strategies)))
        if not strategies:
            strategies = self.grammar_engine.strategies[:3]
        weights = [s.fitness + 0.1 for s in strategies]
        w_sum = sum(weights)
        if w_sum > 0:
            weights = [w / w_sum for w in weights]
        else:
            weights = [1.0 / len(strategies)] * len(strategies)
        strat_idx = np.random.choice(range(len(strategies)), p=weights)
        body = strategies[strat_idx].code_template
        code = f"def solve(task):\n{textwrap.indent(body, '    ')}"
        return code, strat_idx

    def refine_candidate(self, code: str, task_examples: List[Task]) -> str:
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
    def __init__(self, timeout: float = 0.2):
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
        p.join(timeout=self.timeout)
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
        except:
            pass

class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.history = []

    def save(self, controller, generation, score):
        filename = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{generation}.pkl")
        state = {
            "grammar_strategies": controller.grammar_engine.strategies,
            "difficulty": controller.difficulty,
            "success_history": controller.success_history,
            "population": controller.pop_optimizer.population,
            "generation": generation,
            "score": score
        }
        try:
            with open(filename, "wb") as f:
                pickle.dump(state, f)
            self.history.append({'gen': generation, 'score': score, 'file': filename})
            if len(self.history) > 5:
                oldest = self.history.pop(0)
                if os.path.exists(oldest['file']):
                    os.remove(oldest['file'])
        except Exception as e:
            print(f"[Checkpoint] Error: {e}")

    def load_best(self):
        if not self.history:
            return None
        best = sorted(self.history, key=lambda x: x['score'], reverse=True)[0]
        try:
            with open(best['file'], "rb") as f:
                return pickle.load(f)
        except:
            return None

class ProblemGenerator:
    def generate(self, n=5, concept_vector: Optional[np.ndarray] = None, difficulty: int = 1) -> List[Task]:
        tasks = []
        bias_m = 1
        bias_c = 0
        if concept_vector is not None:
            val = np.mean(concept_vector)
            bias_m = int(abs(val * 10)) % 5 + 1
            bias_c = int(abs(val * 100)) % 10
        
        if difficulty == 1:
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
        self.grammar_engine = GrammarEvolutionEngine()
        self.pop_optimizer = PopulationOptimizer(pop_size=20, elite_fraction=0.2)
        self.evaluator = InventionEvaluator(timeout=0.2)
        self.prob_gen = ProblemGenerator()
        self.metacognition = MetacognitiveController()
        self.tesseract = numpy_tesseract.TesseractEngine(z_dim=32)
        self.archive = []
        self.checkpoint_manager = CheckpointManager()
        self.difficulty = 1
        self.success_history = []
        self.arc_benchmark = ARCBenchmark()

    def run(self, generations=10):
        print("Initializing RSI System V2...")
        print("Evolving Tesseract Operators...")
        operators = self.tesseract.evolve_concepts(n_generations=5)
        
        gen = 0
        while True:
            current_op = operators[gen % len(operators)]
            concept_vec = self.tesseract.decode_concept(current_op)
            tasks = self.prob_gen.generate(3, concept_vector=concept_vec, difficulty=self.difficulty)
            complexity = self.metacognition.assess_complexity(tasks[0])
            routing = self.metacognition.route_reasoning(complexity)
            
            print(f"\n[Gen {gen}] Diff: {self.difficulty} | Complexity: {complexity:.2f}")
            
            def evaluate_individual(ind: Individual) -> float:
                searcher = EvolutionarySearcher(self.grammar_engine, ind.synthesis_order)
                self.evaluator.timeout = ind.task_timeout
                best_score = -1
                for _ in range(ind.num_candidates):
                    raw, idx = searcher.generate_candidate(ind.grammar_weights)
                    refined = searcher.refine_candidate(raw, tasks)
                    score = self.evaluator.evaluate(refined, tasks)
                    if score > best_score:
                        best_score = score
                return best_score
            
            if gen == 0:
                self.pop_optimizer.evaluate_population(evaluate_individual)
            
            best_ind = self.pop_optimizer.get_best()
            print(f"Best fitness: {best_ind.fitness:.2f} | Pop mean: {np.mean([i.fitness for i in self.pop_optimizer.population]):.2f}")
            
            if best_ind.fitness >= 1.0:
                print(f"[Archive] Solution stored. Total: {len(self.archive)}")
            
            reward = 1.0 if best_ind.fitness == 1.0 else -1.0
            self.tesseract.feedback(current_op, reward, lr=best_ind.feedback_lr)
            
            self.success_history.append(best_ind.fitness)
            if len(self.success_history) > 20:
                self.success_history.pop(0)
            
            avg_success = sum(self.success_history) / len(self.success_history)
            
            if avg_success > best_ind.curriculum_threshold and len(self.success_history) >= 10:
                self.difficulty += 1
                self.success_history = []
                print(f"[Curriculum] Difficulty increased to {self.difficulty}")
            
            elif avg_success < 0.2 and self.difficulty > 1 and len(self.success_history) >= 10:
                self.difficulty -= 1
                self.success_history = []
                print(f"[Curriculum] Difficulty decreased to {self.difficulty}")
            
            if gen > 0 and gen % 5 == 0:
                pop_stats = self.pop_optimizer.evolve_generation()
                print(f"[Population] {pop_stats}")
                self.pop_optimizer.evaluate_population(evaluate_individual)
                self.grammar_engine.evolve_grammar(generations=3)
            
            if gen > 0 and gen % 20 == 0:
                self.checkpoint_manager.save(self, gen, avg_success)
                print(f"[Checkpoint] Saved generation {gen}")
                print(self.pop_optimizer.get_summary())
            
            if gen > 0 and gen % 50 == 0:
                print("\n[ARC Benchmark Evaluation]")
                # Would need to implement ARC solver wrapper
                print("ARC evaluation placeholder")
            
            if len(self.success_history) >= 20 and avg_success == 0.0:
                print("[Collapse] Initiating rollback")
                state = self.checkpoint_manager.load_best()
                if state:
                    self.difficulty = state['difficulty']
                    self.success_history = []
                    print("[Rollback] State restored")
            
            gen += 1
            if generations > 0 and gen >= generations:
                break
        
        print("\n" + self.pop_optimizer.get_summary())

def cmd_selftest():
    print("Running Self-Test...")
    ctrl = InventionMetaController()
    ctrl.run(generations=2)
    print("Self-Test Complete")

def cmd_evolve():
    ctrl = InventionMetaController()
    ctrl.run(generations=0)

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    sub.add_parser('selftest')
    sub.add_parser('evolve')
    args = parser.parse_args()
    if args.cmd == 'selftest':
        cmd_selftest()
    elif args.cmd == 'evolve':
        cmd_evolve()
    else:
        cmd_selftest()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
