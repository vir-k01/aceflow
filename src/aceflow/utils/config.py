from dataclasses import dataclass
from monty.json import MSONable


@dataclass
class TrainConfig(MSONable):
    num_basis : int = 10
    cutoff : int = 7
    loss_weight : float = 0.9
    max_steps: int = 2000
    batch_size: int = 200
    gpu_index: int = None
    name : str = None


@dataclass
class ActiveLearningConfig(MSONable):
    active_learning_loops : int = 1
    max_points : int = 500
    max_structures : int = 500
    gamma_max : int = 1


@dataclass
class DataGenConfig(MSONable):
    step_skip : int = 20
    num_points : int = 5
    temperature : float = 2000
    max_energy_above_hull : float = 0.1
    md_steps : int = 200
    data_generator : str = 'MD' # set to none if you do not want to generate any new data for training
    kpoints: dict = {"gamma_only": True}
    incar_updates: dict = {
                            "ISPIN": 1, # Do not consider magnetism in AIMD simulations
                            "LREAL": "Auto", # Peform calculation in real space for AIMD due to large unit cell size
                            "LAECHG": False, # Don't need AECCAR for AIMD
                            "LCHARG": False,
                            "EDIFFG": None, # Does not apply to MD simulations, see: https://www.vasp.at/wiki/index.php/EDIFFG
                            "GGA": "PS", # Just let VASP decide based on POTCAR - the default, PS yields the error below
                            "KPAR": 1,
                            "NCORE": 8
                        }