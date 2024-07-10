from dataclasses import dataclass, field
from jobflow import Maker, Flow
from atomate2.vasp.jobs.md import MDMaker
from aceflow.flows.data import DataGenFlowMaker, ActiveStructuresFlowMaker
from aceflow.utils.config import TrainConfig, DataGenConfig, ActiveLearningConfig
from aceflow.jobs.data import read_MD_outputs, consolidate_data
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
    trainer_config: TrainConfig = field(default_factory=lambda: TrainConfig())
    data_gen_config: DataGenConfig = field(default_factory=lambda: DataGenConfig())
    md_maker : Maker = None

    def make(self, compositions: list = None, precomputed_data: pd.DataFrame = None, structures: list = None):
        if compositions is None and structures is None:
            read_job = read_MD_outputs(precomputed_dataset=precomputed_data, step_skip=self.data_gen_config.step_skip)
            trainer = naive_train_ACE(read_job.output, trainer_config=self.trainer_config)
            train_checker = check_training_output(trainer.output)
            return Flow([read_job, trainer, train_checker], output=train_checker.output, name=self.name)
        else: 
            data = DataGenFlowMaker(data_gen_config=self.data_gen_config, md_maker=self.md_maker).make(compositions, structures)
            read_job = read_MD_outputs(md_outputs = data.output, precomputed_dataset=precomputed_data, step_skip=self.data_gen_config.step_skip)
            trainer = naive_train_ACE(read_job.output, trainer_config=self.trainer_config)
            train_checker = check_training_output(trainer.output)
            return Flow([data, read_job, trainer, train_checker], output=train_checker.output, name=self.name)
        
@dataclass
class NaiveACENStepFlowMaker(NaiveACEFlowMaker):
    '''
    N Step ACE Trainer: same as above, but now does the training in N steps: first with emphasis on forces (kappa=0.99), followed by progressively increasing focus on energies.
    The flow returns the directory with the output_potential.yaml, training log and reports.
    '''
    name : str = 'Naive ACE N-Step Trainer'
    loss_weights = [0.99, 0.9, 0.3]

    def make(self, compositions: list = None, precomputed_data : pd.DataFrame = None, structures: list = None):
        train_steps = []
        train_checkers = []
        prev_run_dict = None
        if compositions is None and structures is None:
            read_job = read_MD_outputs(precomputed_dataset=precomputed_data, step_skip=self.data_gen_config.step_skip)
            for i in range(len(self.loss_weights)):
                self.trainer_config.loss_weight = self.loss_weights[i]
                if i:
                    prev_run_dict = train_checkers[-1].output
                train_steps.append(naive_train_ACE(read_job.output, trainer_config=self.trainer_config, prev_run_dict=prev_run_dict))
                train_checkers.append(check_training_output(train_steps[-1].output))
            return Flow([read_job, *train_steps, *train_checkers], output=train_checkers[-1].output, name=self.name)
        else: 
            data = DataGenFlowMaker(data_gen_config=self.data_gen_config, md_maker=self.md_maker).make(compositions, structures)
            read_job = read_MD_outputs(md_outputs=data.output, precomputed_dataset=precomputed_data, step_skip=self.data_gen_config.step_skip)
            for i in range(len(self.loss_weights)):
                self.trainer_config.loss_weight = self.loss_weights[i]
                if i:
                    prev_run_dict = train_checkers[-1].output
                train_steps.append(naive_train_ACE(read_job.output, trainer_config=self.trainer_config, prev_run_dict=prev_run_dict))
                train_checkers.append(check_training_output(train_steps[-1].output))
            return Flow([data, read_job, *train_steps, *train_checkers], output=train_checkers[-1].output, name=self.name)

@dataclass
class ACEMaker(NaiveACENStepFlowMaker):
    '''
    Basic ACE trainer: Wrapper for pacemaker, also calls a series of md jobs on structures (both amorphous using packmol and crystalline, queried from MP). 
    Any precomputed data can be passed to the trainer too, as long as it is in the format required by pacemaker (pd.DataFrame with columns for energy, ase_atoms, forces and energy_corrected).
    By default, 2 steps are used for training, with loss weights [0.99, 0.3], followed by an active learning step in the space defined by the compositions given in the make() call.
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
    data_gen_config : DataGenConfig = field(default_factory=lambda: DataGenConfig())
    active_learning_config : ActiveLearningConfig = field(default_factory=lambda: ActiveLearningConfig())
    static_maker : Maker = None
    md_maker : Maker = None

    def make(self, compositions: list = None, precomputed_data: pd.DataFrame = None, structures: list = None):
        if compositions is None and structures is None:
            self.data_gen_config.data_generator = None
            if precomputed_data is None:
                raise ValueError("Precomputed data must be provided if no structures or compositions are given.")
        if self.data_gen_config.data_generator is None:
            read_job = read_MD_outputs(precomputed_dataset=precomputed_data, step_skip=self.data_gen_config.step_skip)
            trainer = naive_train_ACE(read_job.output, trainer_config=self.trainer_config)
            train_checker = check_training_output(trainer.output)
            active_set_flow = ActiveStructuresFlowMaker(static_maker=self.static_maker, active_learning_config=self.active_learning_config).make(compositions, prev_run_dict=train_checker.output)
            train_active = naive_train_ACE(computed_data_set=read_job.output, active_data_set=active_set_flow.output, prev_run_dict=train_checker.output, trainer_config=self.trainer_config)
            train_active_checker = check_training_output(train_active.output)
            return Flow([read_job, trainer, train_checker, active_set_flow, train_active, train_active_checker], output=train_active_checker.output, name=self.name)
        else:  
            data = DataGenFlowMaker(data_gen_config=self.data_gen_config, md_maker=self.md_maker).make(compositions, structures)
            read_job = read_MD_outputs(md_outputs = data.output, precomputed_dataset=precomputed_data, step_skip=self.data_gen_config.step_skip)
            trainer = naive_train_ACE(read_job.output, trainer_config=self.trainer_config)
            train_checker = check_training_output(trainer.output)
            active_set_flow = ActiveStructuresFlowMaker(static_maker=self.static_maker, active_learning_config = self.active_learning_config).make(compositions, prev_run_dict=train_checker.output)
            train_active = naive_train_ACE(computed_data_set=read_job.output, active_data_set=active_set_flow.output, prev_run_dict=train_checker.output, trainer_config=self.trainer_config)
            train_active_checker = check_training_output(train_active.output)
            return Flow([data, read_job, trainer, train_checker, active_set_flow, train_active, train_active_checker], output=train_active_checker.output, name=self.name)

@dataclass
class ProductionACEMaker(NaiveACENStepFlowMaker):
    name : str = 'ACE Maker'
    trainer_config : TrainConfig = field(default_factory=lambda: TrainConfig())
    data_gen_config : DataGenConfig = field(default_factory=lambda: DataGenConfig())
    active_learning_config : ActiveLearningConfig = field(default_factory=lambda: ActiveLearningConfig())
    static_maker : Maker = None
    md_maker : Maker = None
    loss_weights = [0.99, 0.3]

    def make(self, compositions: list = None, precomputed_data: pd.DataFrame = None, structures: list = None):

        trainers = []
        train_checkers = []
        active_set_flows = []
        consolidate_data_jobs = []
        #active_set_flow_outputs = []
        job_list = []
        prev_run_dict = None
        data_output = None

        if compositions is None and structures is None:
            self.data_gen_config.data_generator = None
            if precomputed_data is None:
                raise ValueError("Precomputed data must be provided if no structures or compositions are given.")
        
        if self.data_gen_config.data_generator:
            data = DataGenFlowMaker(data_gen_config=self.data_gen_config, md_maker=self.md_maker).make(compositions, structures)
            data_output = data.output
            job_list.append(data)

        read_job = read_MD_outputs(md_outputs=data_output, precomputed_dataset=precomputed_data, step_skip=self.data_gen_config.step_skip)
        consolidate_data_jobs.append(consolidate_data([read_job.output]))
        job_list.append(read_job)

        for i in range(len(self.loss_weights)):
            self.trainer_config.loss_weight = self.loss_weights[i]
            if i:
                prev_run_dict = train_checkers[-1].output
            trainers.append(naive_train_ACE(read_job.output, trainer_config=self.trainer_config, prev_run_dict=prev_run_dict))
            train_checkers.append(check_training_output(trainers[-1].output))

        if self.active_learning_config.active_learning_loops:
            for i in range(self.active_learning_config.active_learning_loops):
                active_set_flow = ActiveStructuresFlowMaker(static_maker=self.static_maker, active_learning_config=self.active_learning_config).make(compositions, prev_run_dict=train_checkers[-1].output)
                active_set_flows.append(active_set_flow)
                consolidate_data_jobs.append(consolidate_data([consolidate_data_jobs[-1].output, active_set_flow.output]))
                #active_set_flow_outputs.append(active_set_flow.output)
                for j in range(len(self.loss_weights)):
                    self.trainer_config.loss_weight = self.loss_weights[j]
                    prev_run_dict = train_checkers[-1].output
                    trainers.append(naive_train_ACE(computed_data_set=consolidate_data_jobs[-1].output, prev_run_dict=prev_run_dict, trainer_config=self.trainer_config))
                    train_checkers.append(check_training_output(trainers[-1].output))
            job_list.extend(active_set_flows)
            job_list.extend(consolidate_data_jobs)
        job_list.extend(trainers)
        job_list.extend(train_checkers)
        return Flow(job_list, output=train_checkers[-1].output, name=self.name)
