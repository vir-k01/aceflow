from dataclasses import dataclass, field
from jobflow import Maker, Flow
from atomate2.vasp.jobs.md import MDMaker
from aceflow.flows.data import DataGenFlowMaker, ActiveStructuresFlowMaker
from aceflow.utils.config import TrainConfig, DataGenConfig, ActiveLearningConfig
from aceflow.jobs.data import read_MD_outputs
from aceflow.jobs.train import naive_train_ACE, check_training_output
import pandas as pd

@dataclass
class NaiveACEFlowMaker(Maker):
    '''
    Basic ACE trainer: Wrapper for pacemaker, also calls a series of md jobs on structures (both amorphous using packmol and crystalline, queried from MP). 
    Any precomputed data can be passed to the trainer too, as long as it is in the format required by pacemaker (pd.DataFrame with columns for energy, ase_atoms, forces and energy_corrected).
    The flow returns the directory with the output_potential.yaml, training log and reports.
    '''
    name : str = 'Naive ACE Trainer'
    step_skip : int = 1
    trainer_config: TrainConfig = field(default_factory=lambda: TrainConfig()) #dict = field(default_factory=lambda: {'md_maker': None, 'num_points': 5, 'temperature': 2000, 'max_steps': 2000, 'batch_size': 100, 'gpu_index': None})
    data_gen_config: DataGenConfig = field(default_factory=lambda: DataGenConfig())
    md_maker : Maker = None #MDMaker = field(default_factory=lambda: MDMaker())
    '''num_points : int = 5
    temperature : float = 2000
    max_steps : int = 2000
    batch_size : int = 100
    step_skip : int = 1
    gpu_index : int = None'''

    def make(self, compositions: list = None, precomputed_data: pd.DataFrame = None, structures: list = None):
        if compositions is None and structures is None:
            read_job = read_MD_outputs(precomputed_dataset=precomputed_data, step_skip=self.step_skip)
            trainer = naive_train_ACE(read_job.output, self.trainer_config)
            train_checker = check_training_output(trainer.output)
            return Flow([read_job, trainer, train_checker], output=train_checker.output, name=self.name)
        else: 
            data = DataGenFlowMaker(self.data_gen_config, md_maker=self.md_maker).make(compositions, structures)
            read_job = read_MD_outputs(md_outputs = data.output, precomputed_dataset=precomputed_data, step_skip=self.step_skip)
            trainer = naive_train_ACE(read_job.output, self.trainer_config)
            train_checker = check_training_output(trainer.output)
            return Flow([data, read_job, trainer, train_checker], output=train_checker.output, name=self.name)
        
@dataclass
class NaiveACETwoStepFlowMaker(NaiveACEFlowMaker):
    '''
    Two Step ACE Trainer: same as above, but now does the training in two steps: first with emphasis on forces (kappa=0.99), then emphasis on energies (kappa=0.3).
    The flow returns the directory with the output_potential.yaml, training log and reports.
    '''
    name : str = 'Naive ACE Two Step Trainer'
    loss_weights = [0.99, 0.3]

    def make(self, compositions: list = None, precomputed_data : pd.DataFrame = None, structures: list = None):
        if compositions is None and structures is None:
            read_job = read_MD_outputs(precomputed_dataset=precomputed_data, step_skip=self.step_skip)
            train_step_1 = naive_train_ACE(read_job.output, loss_weight=self.loss_weights[0], max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_checker_1 = check_training_output(train_step_1.output)
            train_step_2 = naive_train_ACE(read_job.output, loss_weight=self.loss_weights[1], prev_run_dict=train_checker_1.output, max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_checker_2 = check_training_output(train_step_2.output)
            return Flow([read_job, train_step_1, train_checker_1, train_step_2, train_checker_2], output=train_checker_2.output, name=self.name)
        else: 
            data = DataGenFlowMaker(num_points=self.num_points, temperature=self.temperature, md_maker=self.md_maker).make(compositions, structures)
            read_job = read_MD_outputs(md_outputs= data.output, precomputed_dataset=precomputed_data, step_skip=self.step_skip)
            train_step_1 = naive_train_ACE(read_job.output, loss_weight=self.loss_weights[0], max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_checker_1 = check_training_output(train_step_1.output)
            train_step_2 = naive_train_ACE(read_job.output, loss_weight=self.loss_weights[1], prev_run_dict=train_checker_1.output, max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_checker_2 = check_training_output(train_step_2.output)
            return Flow([data, read_job, train_step_1, train_checker_1, train_step_2, train_checker_2], output=train_checker_2.output, name=self.name)

@dataclass
class NaiveACENStepFlowMaker(NaiveACEFlowMaker):
    '''
    N Step ACE Trainer: same as above, but now does the training in N steps: first with emphasis on forces (kappa=0.99), followed by progressively increasing focus on energies.
    The flow returns the directory with the output_potential.yaml, training log and reports.
    '''
    name : str = 'Naive ACE N-Step Trainer'
    loss_weights = [0.99, 0.9, 0.3]

    def make(self, compositions: list = None, precomputed_data : pd.DataFrame = None, structures: list = None):
        if compositions is None and structures is None:
            read_job = read_MD_outputs(precomputed_dataset=precomputed_data, step_skip=self.step_skip)
            train_steps = []
            train_checkers = []
            for i in range(len(self.loss_weights)):
                train_steps.append(naive_train_ACE(read_job.output, loss_weight=self.loss_weights[i], max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index))
                train_checkers.append(check_training_output(train_steps[-1].output))
            return Flow([read_job, *train_steps, *train_checkers], output=train_checkers[-1].output, name=self.name)
        else: 
            data = DataGenFlowMaker(num_points=self.num_points, temperature=self.temperature, md_maker=self.md_maker).make(compositions, structures)
            read_job = read_MD_outputs(md_outputs=data.output, precomputed_dataset=precomputed_data, step_skip=self.step_skip)
            train_steps = []
            train_checkers = []
            for i in range(len(self.loss_weights)):
                train_steps.append(naive_train_ACE(read_job.output, loss_weight=self.loss_weights[i], max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index))
                train_checkers.append(check_training_output(train_steps[-1].output))
            return Flow([data, read_job, *train_steps, *train_checkers], output=train_checkers[-1].output, name=self.name)

@dataclass
class ACEMaker(NaiveACENStepFlowMaker):
    '''
    ACEMaker(train_cinfig=TrainConfig(**kwargs))
    Basic ACE trainer: Wrapper for pacemaker, also calls a series of md jobs on structures (both amorphous using packmol and crystalline, queried from MP). 
    Any precomputed data can be passed to the trainer too, as long as it is in the format required by pacemaker (pd.DataFrame with columns for energy, ase_atoms, forces and energy_corrected).
    The flow returns the directory with the output_potential.yaml, training log and reports.

    Inputs:
    compositions : list = None : List of compositions to train on. If None, structures must be provided.
    precomputed_data : pd.DataFrame = None : Precomputed data to train on. If None, structures must be provided.
    structures : list = None : List of structures to train on. If None, compositions must be provided.

    Config:
    md_maker : Maker = None : Maker for MD sampling jobs. If None, a default MDMaker is used.
    num_points : int = 5 : Number of points to use for volume deformations, which are used to generate MD sampling jobs.
    temperature : float = 2000 : Temperature to use for MD sampling jobs.
    max_steps : int = 2000 : Maximum number of steps to train ACE for.
    batch_size : int = 100 : Batch size to use for training. If batch_size is too big to fit in memory, it will be reduced by a factor of 1.618.
    step_skip : int = 1 : Number of steps to skip when reading MD outputs (reduces similairty between the sampled structures).
    gpu_index : int = None : Index of the GPU to use for training. If None, CPU is used. Make sure to set the correct CUDA_VISIBLE_DEVICES environment variable, and have the correct version of cudatoolkit and cudnn installed.
    loss_weights : list = [0.99, 0.3] : List of loss weights to use for training. If None, the default weights are used. The first weight is used for the first step, and the second weight is used for the second step.
    static_maker : Maker = None : Maker for static calculations for active learning. If None, a default StaticMaker is used. 
    max_points : int = 1 : Maximum number of points in composition space to sample for testing the potential in an active learning setting.
    max_structures : int = 10 : Maximum number of structures to select into the active set for retraining the potential with.
    gamma_max : int = 1 : Cutoff extrapolation grade beyond which a structure is considered to be in the active set.

    '''
    name : str = 'ACE Maker'
    trainer_config : TrainConfig = field(default_factory=lambda: TrainConfig())
    trainer_config: dict = field(default_factory=lambda: {'md_maker': None, 'num_points': 5, 'temperature': 2000, 'max_steps': 2000, 'batch_size': 100, 'step_skip': 1, 'gpu_index': None, 'loss_weights': [0.99, 0.3], 'static_maker': None, 'max_structures': 10, 'max_points': 1, 'gamma_max': 1})
    static_maker : Maker = None
    max_structures : int = 10
    max_points : int = 1
    gamma_max : int = 1

    def make(self, compositions: list = None, precomputed_data: pd.DataFrame = None, structures: list = None):
        if compositions is None and structures is None:
            read_job = read_MD_outputs(precomputed_dataset=precomputed_data, step_skip=self.step_skip)
            trainer = naive_train_ACE(read_job.output, max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_checker = check_training_output(trainer.output)
            active_set_flow = ActiveStructuresFlowMaker(static_maker=self.static_maker, max_structures=self.max_structures, max_points=self.max_points, gamma_max=self.gamma_max).make(compositions, prev_run_dict=train_checker.output)
            train_active = naive_train_ACE(computed_data_set=read_job.output, active_data_set=active_set_flow.output, prev_run_dict=train_checker.output, max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_active_checker = check_training_output(train_active.output)
            return Flow([read_job, trainer, train_checker, active_set_flow, train_active, train_active_checker], output=train_active_checker.output, name=self.name)
        else:  
            data = DataGenFlowMaker(num_points=self.num_points, temperature=self.temperature, md_maker=self.md_maker).make(compositions, structures)
            read_job = read_MD_outputs(md_outputs = data.output, precomputed_dataset=precomputed_data, step_skip=self.step_skip)
            trainer = naive_train_ACE(read_job.output, max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_checker = check_training_output(trainer.output)
            active_set_flow = ActiveStructuresFlowMaker(static_maker=self.static_maker, max_structures=self.max_structures, max_points=self.max_points, gamma_max=self.gamma_max).make(compositions, prev_run_dict=train_checker.output)
            train_active = naive_train_ACE(computed_data_set=read_job.output, active_data_set=active_set_flow.output, prev_run_dict=train_checker.output, max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_active_checker = check_training_output(train_active.output)
            return Flow([data, read_job, trainer, train_checker, active_set_flow, train_active, train_active_checker], output=train_active_checker.output, name=self.name)

