from dataclasses import dataclass, field
from monty.json import MSONable
from jobflow import job


@dataclass
class BaseActiveLearningStrategy(MSONable):
    name : str = 'Base Active Learning Strategy'
    active_learning_loops : int = 1
    max_structures : int = 100
    target_atoms : int = 100
    gamma_low : float = 5
    gamma_high : float = 100
    base_calculator = None

    def sample_structures(self):
        pass