from dataclasses import dataclass, field
from monty.json import MSONable
from typing import Union, Dict
from aceflow.reference_objects.BBasis_classes import *



@dataclass
class HeirarchicalFitConfig(MSONable):
    initial_potentials: dict = None
    start_order: list = None
    end_order: list = None


@dataclass
class TrainConfig(MSONable):
    num_basis : int = 500 # Number of basis functions to use per element. Default is 500, increase if you feel the model is underfitting. P.S.: If you end up using > 1000, you're probably doing something wrong.
    cutoff : int = 7 # Cutoff for neighbour list construction. Default is 7, increase if you have a large unit cell, or expect long range interactions in your system.
    loss_weight : float = 0.9 # Weight of the loss function. Default is 0.9.
    max_steps: int = 2000 # Maximum number of training steps. Default is 2000, increase if you feel the training needs more time to converge. Note: The training will stop if the loss function converges before this number of steps.
    batch_size: int = 200 # Batch size for training. Default is 200, decrease if you run out of memory.
    gpu_index: int = None # Index of the GPU to use. Default is None, which means the code will use the first available tf-compatible GPU. If you have multiple GPUs, you can specify the index of the GPU you want to use. If you don't have a GPU, only the CPU will be used.
    restart_failed_runs: bool = False # If True, the code will restart the failed runs from the last checkpoint. Default is False, set to True if you want to restart the failed runs.
    ladder_step: list = field(default_factory=lambda: [100, 0.2]) # Number of steps to take in the ladder. This will increase the size of the basis step by a fraction of 0.2 the current size or 100 functions, whichever is higher.
    ladder_type: str = None #'body_order' or 'power_order. If None, no ladder is used. If 'body_order', the ladder will increase the number of body order functions. If 'power_order', the ladder will increase the power order of the basis functions.
    test_size: float = 0.1 # Fraction of the data to use for testing. Default is 0.1, increase if you want to use more data for testing.
    upfit: bool = False # If True, the code will use the previous run as an initial guess for the current run and extend the basis set as per the ladder fitting scheme. 
    chemsys: Union[dict, list] = None # A dict mapping the elements to their reference energies. If only a list of elements is provided, the reference energies are taken from precomputed GGA PBEsol energies. 
    bbasis: Dict[str, FlowBBasisOrder] = field(default_factory=lambda: {'UNARY': UnaryBBasisOrder(), 'BINARY': BinaryBBasisOrder(), 'TERNARY': TernaryBBasisOrder(), 'QUATERNARY': QuaternaryBBasisOrder(), 'QINARY': QuinaryBBasisOrder(), 'ALL': AllBBasisOrder(), 'bonds': BBasisBonds(), 'embedding': BBasisEmbedding()}) # List of BBasisOrder objects to use for the training. 
    bbasis_train_orders: list = None # Order of the BBasis to train. Default is -1, which means all the bbasis functions are trained. If you want to train only a specific order, you can specify it here. Example: [0, 2] will train only the unary binary, ternary basis functions. [0, 0] will train only the unary basis functions. 
    heirarchical_fit: HeirarchicalFitConfig = None # Heirarchical fitting configuration. If None, no heirarchical fitting is used.
    name : str = None 


@dataclass
class ActiveLearningConfig(MSONable):
    active_learning_loops : int = 1 # Number of active learning loops to run. Default is 1, increase if you want to run multiple loops of active learning.
    sampling_mathod : str = 'compositional' # 'compositional' or 'structural' or 'all' 
    max_points : int = 500 # Maximum number of points to test the potential at in each loop. Default is 500, increase if you want to select more points in each loop. 
    max_structures : int = 500 # Maximum number of structures to add to the active set in each loop. These many DFT statics will be performed. To use every strucure selected by pace_select, set to -1. 
    gamma_max : int = 5 # Cutoff for extrapolation grade. Default is 5, decrease if you to consider more points for the active set.
    sampling_frequency : int = 5 # Frequency of sampling points for the active learning. Default is 5, increase if you want to sample more points for the active learning.


@dataclass
class DataGenConfig(MSONable):
    step_skip : int = 25  # Number of steps to sample frames from the MD trajectory. Default is 25, decrease if you want to include more frames for training, however the resulting data might be more correlated.
    num_points : int = 5 # Number of deformations to consider for each structure. Default is 5, increase if you want to consider more deformations, which leads to better generalization of the trained model.
    temperature : float = 2000 # Temperature of the MD simulation. Default is 2000 K, increase if you want to consider higher temperature simulations to sample more "liquid-like" out of equilibrium states for training. 
    max_energy_above_hull : float = 0.1 # Maximum energy above hull to consider for the structures. Default is 0.1 eV/atom, increase if you want to consider more metastable structures from the Materials Project for training.
    md_steps : int = 200 # Number of MD steps to run for each deformation. Default is 200, increase if you want to increas the size of the training set. 
    data_generator : str = 'MD' # 'MD' or 'Static' or 'Static_Defect', the corresponding input set is used to generate data. If None, no extra data is generated. 'Static_Defect' also generates structures with point defects (might lead to convergence issues in DFT!).
    kpoints: dict = field(default_factory=lambda: {"gamma_only": True}) # KPOINTS file for VASP calculations. Default is gamma_only=True, which means only the gamma point is used for the calculations. This works here since the cells being built are large (>1nm in length).
    #Below is the input set used for MD. This is essentially the same as the default input set for the MDMaker in atomate2. If you want to use a different input set for the MD, you can specify it here.
    incar_updates: dict = field(default_factory=lambda:{
                            "ISPIN": 1, # Do not consider magnetism in AIMD simulations
                            "LREAL": "Auto", # Peform calculation in real space for AIMD due to large unit cell size
                            "LAECHG": False, # Don't need AECCAR for AIMD
                            "LCHARG": False, # Don't need CHGCAR for AIMD
                            "EDIFFG": None, # Does not apply to MD simulations, see: https://www.vasp.at/wiki/index.php/EDIFFG
                            "GGA": "PS", # Just let VASP decide based on POTCAR - the default, PS yields the error below
                            "KPAR": 1, # Parallelization over K-points, use 1 since gamma-point only calculations are used
                            "NCORE": 8 # Number of cores to parralelize for the calculation. Default is 8, increase if you have more cores available. In this case, make sure NCORE is a divisor of the number of cores available.
                        })