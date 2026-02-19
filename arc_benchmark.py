import json
import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

@dataclass
class ARCTask:
    task_id: str
    train_examples: List[Tuple[list, list]]
    test_examples: List[Tuple[list, list]]
    difficulty: str
    description: str


class ARCBenchmark:
    def __init__(self):
        self.tasks = self._create_tasks()
    
    def _create_tasks(self) -> List[ARCTask]:
        tasks = []
        tasks.append(ARCTask(
            task_id="repeat_pattern",
            train_examples=[([1, 2, 1, 2], [1, 2]), ([3, 3, 3, 3], [3]), ([5, 6, 7, 5, 6, 7], [5, 6, 7])],
            test_examples=[([4, 5, 4, 5, 4, 5], [4, 5]), ([9, 9], [9])],
            difficulty='easy',
            description="Extract repeating pattern"
        ))
        tasks.append(ARCTask(
            task_id="arithmetic_prog",
            train_examples=[([2, 4, 6, 8], [10]), ([5, 10, 15], [20]), ([1, 3, 5, 7], [9])],
            test_examples=[([3, 6, 9, 12], [15]), ([10, 20, 30], [40])],
            difficulty='easy',
            description="Predict next in arithmetic sequence"
        ))
        tasks.append(ARCTask(
            task_id="count_unique",
            train_examples=[([1, 2, 1, 3, 2], [3]), ([5, 5, 5, 5], [1]), ([1, 2, 3, 4, 5], [5])],
            test_examples=[([7, 8, 7, 9], [3]), ([2, 2, 2], [1])],
            difficulty='medium',
            description="Count unique elements"
        ))
        tasks.append(ARCTask(
            task_id="fibonacci_rule",
            train_examples=[([1, 1], [2]), ([1, 1, 2], [3]), ([1, 1, 2, 3], [5])],
            test_examples=[([1, 1, 2, 3, 5], [8]), ([2, 3, 5, 8], [13])],
            difficulty='medium',
            description="Predict next Fibonacci number"
        ))
        tasks.append(ARCTask(
            task_id="find_mode",
            train_examples=[([1, 2, 2, 3], [2]), ([5, 5, 5, 6, 7], [5]), ([3, 4, 4, 3, 3], [3])],
            test_examples=[([8, 9, 8, 8, 7], [8])],
            difficulty='hard',
            description="Find most frequent element"
        ))
        return tasks
    
    def evaluate_on_task(self, task: ARCTask, solver_fn) -> Dict[str, float]:
        correct = 0
        total = len(task.test_examples)
        for input_seq, expected_output in task.test_examples:
            try:
                prediction = solver_fn(input_seq)
                if prediction == expected_output:
                    correct += 1
            except:
                pass
        accuracy = correct / total if total > 0 else 0.0
        return {
            'task_id': task.task_id,
            'difficulty': task.difficulty,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def full_evaluation(self, solver_fn) -> Dict[str, Any]:
        results = []
        for task in self.tasks:
            task_result = self.evaluate_on_task(task, solver_fn)
            results.append(task_result)
        accuracies = [r['accuracy'] for r in results]
        by_difficulty = {
            'easy': [r['accuracy'] for r in results if r['difficulty'] == 'easy'],
            'medium': [r['accuracy'] for r in results if r['difficulty'] == 'medium'],
            'hard': [r['accuracy'] for r in results if r['difficulty'] == 'hard']
        }
        return {
            'task_results': results,
            'overall_accuracy': np.mean(accuracies),
            'easy_accuracy': np.mean(by_difficulty['easy']) if by_difficulty['easy'] else 0.0,
            'medium_accuracy': np.mean(by_difficulty['medium']) if by_difficulty['medium'] else 0.0,
            'hard_accuracy': np.mean(by_difficulty['hard']) if by_difficulty['hard'] else 0.0,
            'tasks_solved': sum(1 for r in results if r['accuracy'] >= 0.5),
            'perfect_tasks': sum(1 for r in results if r['accuracy'] == 1.0)
        }
