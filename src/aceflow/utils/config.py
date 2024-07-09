from dataclasses import dataclass, field



@dataclass
class TrainConfig():
    num_basis : int = 10
    cutoff : int = 7
    loss_weight : float = 0.9
    max_steps: int = 2000
    batch_size: int = 200
    gpu_index: int = None


@dataclass
class ActiveLearningConfig():
    max_points : int = 500
    max_structures : int = 200
    gamma_max : int = 5


@dataclass
class DataGenConfig():
    step_skip : int = 1
    num_points : int = 5
    temperature : float = 2000
    md_steps : int = 10

