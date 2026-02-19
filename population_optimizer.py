import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import copy

@dataclass
class Individual:
    num_candidates: int = 10
    task_timeout: float = 0.2
    feedback_lr: float = 0.05
    curriculum_threshold: float = 0.75
    grammar_weights: Dict[str, float] = field(default_factory=lambda: {'map': 1.0, 'loop': 1.0, 'fold': 0.5})
    synthesis_order: List[str] = field(default_factory=lambda: ['identity', 'linear', 'quadratic', 'modulo'])
    tesseract_latent_dim: int = 32
    fitness: float = 0.0
    age: int = 0
    generation_born: int = 0
    
    def mutate(self, mutation_rate: float = 0.15) -> 'Individual':
        child = copy.deepcopy(self)
        child.age = 0
        child.generation_born = self.generation_born
        
        if np.random.rand() < mutation_rate:
            child.num_candidates = max(5, min(50, int(self.num_candidates + np.random.randn() * 5)))
        if np.random.rand() < mutation_rate:
            child.task_timeout = max(0.1, min(2.0, self.task_timeout + np.random.randn() * 0.1))
        if np.random.rand() < mutation_rate:
            child.feedback_lr = max(0.01, min(0.3, self.feedback_lr + np.random.randn() * 0.02))
        if np.random.rand() < mutation_rate:
            child.curriculum_threshold = max(0.5, min(0.95, self.curriculum_threshold + np.random.randn() * 0.05))
        if np.random.rand() < mutation_rate:
            for key in child.grammar_weights:
                child.grammar_weights[key] *= np.exp(np.random.randn() * 0.3)
                child.grammar_weights[key] = max(0.1, min(5.0, child.grammar_weights[key]))
        if np.random.rand() < mutation_rate * 0.5:
            if np.random.rand() < 0.5 and len(child.synthesis_order) > 1:
                i, j = np.random.choice(len(child.synthesis_order), 2, replace=False)
                child.synthesis_order[i], child.synthesis_order[j] = child.synthesis_order[j], child.synthesis_order[i]
        if np.random.rand() < mutation_rate * 0.3:
            child.tesseract_latent_dim = np.random.choice([16, 32, 48, 64])
        return child
    
    def crossover(self, other: 'Individual') -> 'Individual':
        child = Individual()
        child.generation_born = self.generation_born
        child.num_candidates = self.num_candidates if np.random.rand() < 0.5 else other.num_candidates
        child.task_timeout = self.task_timeout if np.random.rand() < 0.5 else other.task_timeout
        child.feedback_lr = self.feedback_lr if np.random.rand() < 0.5 else other.feedback_lr
        child.curriculum_threshold = self.curriculum_threshold if np.random.rand() < 0.5 else other.curriculum_threshold
        child.grammar_weights = {}
        for key in self.grammar_weights:
            alpha = np.random.rand()
            child.grammar_weights[key] = alpha * self.grammar_weights[key] + (1 - alpha) * other.grammar_weights.get(key, 1.0)
        child.synthesis_order = self.synthesis_order if np.random.rand() < 0.5 else other.synthesis_order
        child.tesseract_latent_dim = self.tesseract_latent_dim if np.random.rand() < 0.5 else other.tesseract_latent_dim
        return child


class PopulationOptimizer:
    def __init__(self, pop_size: int = 20, elite_fraction: float = 0.2):
        self.pop_size = pop_size
        self.elite_size = max(2, int(pop_size * elite_fraction))
        self.population: List[Individual] = []
        self.generation = 0
        self.best_ever: Optional[Individual] = None
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.stagnation_counter = 0
        self._initialize_population()
    
    def _initialize_population(self):
        for i in range(self.pop_size):
            ind = Individual()
            ind.num_candidates = int(5 + (45 / self.pop_size) * i)
            ind.task_timeout = 0.1 + (1.9 / self.pop_size) * i
            ind.feedback_lr = 0.01 + (0.29 / self.pop_size) * i
            ind.curriculum_threshold = 0.5 + (0.45 / self.pop_size) * i
            ind.tesseract_latent_dim = [16, 32, 48, 64][i % 4]
            ind.generation_born = 0
            self.population.append(ind)
    
    def evaluate_population(self, fitness_fn):
        for ind in self.population:
            if ind.fitness == 0.0:
                ind.fitness = fitness_fn(ind)
                ind.age += 1
        current_best = max(self.population, key=lambda x: x.fitness)
        if self.best_ever is None or current_best.fitness > self.best_ever.fitness:
            self.best_ever = copy.deepcopy(current_best)
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        fitnesses = [ind.fitness for ind in self.population]
        self.fitness_history.append(np.mean(fitnesses))
        self.diversity_history.append(np.std(fitnesses))
    
    def evolve_generation(self) -> Dict[str, any]:
        self.generation += 1
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        elites = self.population[:self.elite_size]
        offspring = []
        base_mutation_rate = 0.15
        if self.stagnation_counter > 5:
            mutation_rate = min(0.5, base_mutation_rate * (1 + self.stagnation_counter * 0.1))
        else:
            mutation_rate = base_mutation_rate
        while len(offspring) < self.pop_size - self.elite_size:
            if np.random.rand() < 0.7:
                tournament = np.random.choice(elites, size=min(3, len(elites)), replace=False)
                parent = max(tournament, key=lambda x: x.fitness)
                child = parent.mutate(mutation_rate=mutation_rate)
            else:
                parents = np.random.choice(elites, size=2, replace=False)
                child = parents[0].crossover(parents[1])
                if np.random.rand() < 0.3:
                    child = child.mutate(mutation_rate=mutation_rate * 0.5)
            child.generation_born = self.generation
            offspring.append(child)
        self.population = elites + offspring
        return {
            'generation': self.generation,
            'best_fitness': self.best_ever.fitness if self.best_ever else 0.0,
            'mean_fitness': np.mean([ind.fitness for ind in self.population]),
            'diversity': np.std([ind.fitness for ind in self.population]),
            'mutation_rate': mutation_rate,
            'stagnation': self.stagnation_counter
        }
    
    def get_best(self) -> Individual:
        return self.best_ever if self.best_ever else self.population[0]
    
    def get_summary(self) -> str:
        best = self.get_best()
        lines = [
            f"\nGeneration {self.generation}",
            f"Best Fitness: {best.fitness:.4f} (Gen {best.generation_born})",
            f"Mean Fitness: {np.mean([i.fitness for i in self.population]):.4f}",
            f"Diversity: {np.std([i.fitness for i in self.population]):.4f}",
            f"Stagnation: {self.stagnation_counter}",
            f"\nBest Config:",
            f"  num_candidates: {best.num_candidates}",
            f"  task_timeout: {best.task_timeout:.3f}",
            f"  feedback_lr: {best.feedback_lr:.3f}",
            f"  curriculum_threshold: {best.curriculum_threshold:.3f}",
            f"  tesseract_latent_dim: {best.tesseract_latent_dim}",
            f"  grammar_weights: {best.grammar_weights}",
            f"  synthesis_order: {best.synthesis_order}\n"
        ]
        return "\n".join(lines)
