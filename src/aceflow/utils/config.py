from dataclasses import dataclass, field
from monty.json import MSONable


@dataclass
class TrainConfig(MSONable):
    num_basis : int = 10
    cutoff : int = 7
    loss_weight : float = 0.9
    max_steps: int = 2000
    batch_size: int = 200
    gpu_index: int = None


@dataclass
class ActiveLearningConfig(MSONable):
    active_learning_loops : int = 1
    max_points : int = 500
    max_structures : int = 200
    gamma_max : int = 5


@dataclass
class DataGenConfig(MSONable):
    step_skip : int = 1
    num_points : int = 5
    temperature : float = 2000
    md_steps : int = 10
    data_generator : str = 'MD' # set to none if you do not want to generate any new data for training

