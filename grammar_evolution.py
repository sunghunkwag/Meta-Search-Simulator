import ast
import random
import textwrap
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np

@dataclass
class StrategyGene:
    name: str
    code_template: str
    complexity: int
    success_count: int = 0
    failure_count: int = 0
    novelty_score: float = 0.0
    
    @property
    def fitness(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        success_rate = self.success_count / total
        return success_rate - (self.complexity * 0.01)
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class GrammarEvolutionEngine:
    def __init__(self):
        self.strategies: List[StrategyGene] = []
        self.archive: Set[str] = set()
        self._initialize_base_strategies()
    
    def _initialize_base_strategies(self):
        self.strategies = [
            StrategyGene(
                name="map_transform",
                code_template="return [__HOLE__(x) for x in task.input]",
                complexity=1
            ),
            StrategyGene(
                name="loop_accumulate",
                code_template="res = []\nfor x in task.input:\n    res.append(__HOLE__(x))\nreturn res",
                complexity=2
            ),
            StrategyGene(
                name="fold_aggregate",
                code_template="res = []\nfor i, x in enumerate(task.input):\n    res.append(__HOLE__(x) if i == 0 else __HOLE__(x) + res[-1])\nreturn res",
                complexity=3
            ),
        ]
    
    def mutate_strategy(self, strategy: StrategyGene) -> Optional[StrategyGene]:
        try:
            template = f"def solve(task):\n{textwrap.indent(strategy.code_template, '    ')}"
            tree = ast.parse(template)
            mutation_type = random.choice(['add_filter', 'add_condition', 'add_accumulator', 'change_loop'])
            
            if mutation_type == 'add_filter':
                mutated_code = strategy.code_template.replace(
                    "for x in task.input]",
                    "for x in task.input if x > 0]"
                )
            elif mutation_type == 'add_condition':
                if 'append' in strategy.code_template:
                    mutated_code = strategy.code_template.replace(
                        "res.append(__HOLE__(x))",
                        "res.append(__HOLE__(x) if x != 0 else 0)"
                    )
                else:
                    mutated_code = strategy.code_template
            elif mutation_type == 'add_accumulator':
                if 'for' in strategy.code_template and 'acc' not in strategy.code_template:
                    mutated_code = "acc = 0\n" + strategy.code_template.replace(
                        "res.append(__HOLE__(x))",
                        "acc += x\n    res.append(__HOLE__(x) + acc)"
                    )
                else:
                    mutated_code = strategy.code_template
            else:
                if '[' in strategy.code_template and 'for' in strategy.code_template:
                    mutated_code = "res = []\nfor x in task.input:\n    res.append(__HOLE__(x))\nreturn res"
                else:
                    mutated_code = strategy.code_template
            
            test_code = f"def solve(task):\n{textwrap.indent(mutated_code, '    ')}"
            ast.parse(test_code)
            
            new_strategy = StrategyGene(
                name=f"{strategy.name}_mut_{random.randint(1000,9999)}",
                code_template=mutated_code,
                complexity=strategy.complexity + 1,
                novelty_score=self._compute_novelty(mutated_code)
            )
            return new_strategy
        except SyntaxError:
            return None
    
    def crossover_strategies(self, parent1: StrategyGene, parent2: StrategyGene) -> Optional[StrategyGene]:
        try:
            if 'for' in parent1.code_template and '__HOLE__' in parent2.code_template:
                mutated_code = parent1.code_template
                child = StrategyGene(
                    name=f"cross_{random.randint(1000,9999)}",
                    code_template=mutated_code,
                    complexity=(parent1.complexity + parent2.complexity) // 2
                )
                return child
            return None
        except:
            return None
    
    def _compute_novelty(self, code: str) -> float:
        if not self.archive:
            return 1.0
        tokens = set(code.split())
        archive_tokens = set()
        for archived in self.archive:
            archive_tokens.update(archived.split())
        novel_tokens = tokens - archive_tokens
        return len(novel_tokens) / max(len(tokens), 1)
    
    def evolve_grammar(self, generations: int = 10):
        for gen in range(generations):
            self.strategies.sort(key=lambda s: s.fitness + s.novelty_score * 0.3, reverse=True)
            survivors = self.strategies[:len(self.strategies)//2]
            offspring = []
            for _ in range(len(self.strategies) - len(survivors)):
                if random.random() < 0.7:
                    parent = random.choice(survivors)
                    child = self.mutate_strategy(parent)
                    if child:
                        offspring.append(child)
                else:
                    if len(survivors) >= 2:
                        p1, p2 = random.sample(survivors, 2)
                        child = self.crossover_strategies(p1, p2)
                        if child:
                            offspring.append(child)
            self.strategies = survivors + offspring
            for strat in self.strategies:
                self.archive.add(strat.code_template)
    
    def get_best_strategies(self, top_k: int = 5) -> List[StrategyGene]:
        sorted_strats = sorted(self.strategies, key=lambda s: s.fitness, reverse=True)
        return sorted_strats[:top_k]
    
    def get_strategy_distribution(self) -> dict:
        return {
            'total_strategies': len(self.strategies),
            'unique_patterns': len(self.archive),
            'avg_complexity': np.mean([s.complexity for s in self.strategies]),
            'avg_success_rate': np.mean([s.success_rate for s in self.strategies if s.success_count + s.failure_count > 0])
        }
