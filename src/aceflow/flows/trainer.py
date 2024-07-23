from dataclasses import dataclass, field
from jobflow import Maker, Flow
from atomate2.vasp.jobs.md import MDMaker
from atomate2.vasp.jobs.core import StaticMaker
from aceflow.flows.data import DataGenFlowMaker, ActiveStructuresFlowMaker
from aceflow.utils.config import TrainConfig, DataGenConfig, ActiveLearningConfig, HeirarchicalFitConfig
from aceflow.reference_objects.BBasis_classes import *
from aceflow.core.model import TrainedPotential
from aceflow.jobs.data import consolidate_data
from aceflow.jobs.train import naive_train_ACE, check_training_output
import pandas as pd
import os
import numpy as np
from typing import Union, Dict


@dataclass
class ACEMaker(Maker):
    name : str = 'ACE Maker'
    trainer_config : TrainConfig = field(default_factory=lambda: TrainConfig())
    loss_weights : list = field(default_factory=lambda: [0.99, 0.3])

    def make(self, data: Union[pd.DataFrame, str], pretrained_potential: Union[str, TrainedPotential] = None) -> Flow:
        trainers = []
        train_checkers = []
        job_list = []
        trained_potential = None

        if self.trainer_config.chemsys is None:
            raise ValueError("Chemical system must be provided in the trainer config.")
        
        if not isinstance(data, str):
            try:
                data_path = os.getcwd() + '/data.pckl.gzip'
                pd.to_pickle(data, data_path, compression='gzip', protocol=4)
            except:
                raise ValueError("Due to JobStore issues, data must be a path to a pickled dataframe in .pckl.gzip format OR an instance of a pd.DataFrame which is pickled in this call.")

        if pretrained_potential:
            if isinstance(pretrained_potential, str):
                trained_potential = TrainedPotential()
                trained_potential.read_potential(pretrained_potential)
                if os.path.isfile(pretrained_potential.replace(".yaml", ".asi")):
                    trained_potential.active_set_file = pretrained_potential.replace(".yaml", ".asi")

            if isinstance(pretrained_potential, TrainedPotential):
                trained_potential = pretrained_potential
            else:
                raise ValueError("Pretrained potential must be a path to a yaml file or an instance of the TrainedPotential class.")
        
        for i, loss in enumerate(self.loss_weights):
            self.trainer_config.loss_weight = loss
            if i:
                trained_potential = train_checkers[-1].output.trained_potential
            self.trainer_config.name = f"Step 0.{i} Trainer, Loss Weight: {self.loss_weights[i]}"
            trainers.append(naive_train_ACE(data, trainer_config=self.trainer_config, trained_potential=trained_potential))
            train_checkers.append(check_training_output(trainers[-1].output, trainer_config=self.trainer_config))

        job_list.extend(trainers)
        job_list.extend(train_checkers)
        return Flow(job_list, output=train_checkers[-1].output, name=self.name)


@dataclass
class ProductionACEMaker(Maker):
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
    name : str = 'Production ACE Maker'
    trainer_config : TrainConfig = field(default_factory=lambda: TrainConfig())
    data_gen_config : DataGenConfig = field(default_factory=lambda: DataGenConfig())
    active_learning_config : ActiveLearningConfig = field(default_factory=lambda: ActiveLearningConfig())
    static_maker : StaticMaker = None
    md_maker : MDMaker = None
    loss_weights : list = field(default_factory=lambda: [0.99, 0.3])

    def make(self, compositions: list = None, precomputed_data: Union[pd.DataFrame, str] = None, structures: list = None, pretrained_potential: Union[str, TrainedPotential] = None) -> Flow:

        trainers = []
        train_checkers = []
        active_set_flows = []
        consolidate_data_jobs = []
        job_list = []
        data_output = None
        trained_potential = None

        if compositions is None and structures is None:
            self.data_gen_config.data_generator = None
            if precomputed_data is None:
                raise ValueError("Precomputed data must be provided if no structures or compositions are given.")
        
        if self.data_gen_config.data_generator:
            data = DataGenFlowMaker(data_gen_config=self.data_gen_config, md_maker=self.md_maker, static_maker=self.static_maker).make(compositions, structures)
            job_list.append(data)
            data_output = data.output
        
        if not isinstance(precomputed_data, str):
            try:
                precomputed_data_path = os.getcwd() + '/precomputed_data.pckl.gzip'
                pd.to_pickle(precomputed_data, precomputed_data_path, compression='gzip', protocol=4)
            except:
                raise ValueError("Due to JobStore issues, precomputed data must be a path to a pickled dataframe in .pckl.gzip format OR an instance of a pd.DataFrame which is pickled in this call.")
    
        consolidate_data_jobs.append(consolidate_data([data_output, precomputed_data]))


        if pretrained_potential:
            if isinstance(pretrained_potential, str):
                trained_potential = TrainedPotential()
                trained_potential.read_potential(pretrained_potential)
                if os.path.isfile(pretrained_potential.replace(".yaml", ".asi")):
                    trained_potential.active_set_file = pretrained_potential.replace(".yaml", ".asi")

            if isinstance(pretrained_potential, TrainedPotential):
                trained_potential = pretrained_potential
            else:
                raise ValueError("Pretrained potential must be a path to a yaml file or an instance of the TrainedPotential class.")
        

        for i, loss in enumerate(self.loss_weights):
            self.trainer_config.loss_weight = loss
            if i:
                trained_potential = train_checkers[-1].output.trained_potential
            self.trainer_config.name = f"Step 0.{i} Trainer, Loss Weight: {self.loss_weights[i]}"
            trainers.append(naive_train_ACE(consolidate_data_jobs[-1].output.acedata, trainer_config=self.trainer_config, trained_potential=trained_potential))
            train_checkers.append(check_training_output(trainers[-1].output, trainer_config=self.trainer_config))

        if self.active_learning_config.active_learning_loops:
            for i in range(self.active_learning_config.active_learning_loops):
                active_set_flow = ActiveStructuresFlowMaker(static_maker=self.static_maker, active_learning_config=self.active_learning_config).make(compositions, trained_potential=train_checkers[-1].output.trained_potential)
                active_set_flows.append(active_set_flow)
                consolidate_data_jobs.append(consolidate_data([consolidate_data_jobs[-1].output.acedata, active_set_flow.output]))
      
                for j, loss in enumerate(self.loss_weights):
                    self.trainer_config.loss_weight = loss
                    trained_potential = train_checkers[-1].output.trained_potential
                    self.trainer_config.name = f"Active Step {i}.{j} Trainer, Loss Weight: {self.loss_weights[j]}"
                    trainers.append(naive_train_ACE(computed_data_set=consolidate_data_jobs[-1].output, trainer_config=self.trainer_config, trained_potential=trained_potential))
                    train_checkers.append(check_training_output(trainers[-1].output, trainer_config=self.trainer_config))

            job_list.extend(active_set_flows)
        job_list.extend(consolidate_data_jobs)

        job_list.extend(trainers)
        job_list.extend(train_checkers)
        return Flow(job_list, output=train_checkers[-1].output, name=self.name)
    

@dataclass
class HeirarchicalACEMaker(ACEMaker):
    name : str = 'Heirarchical ACE Maker'
    hconfig : HeirarchicalFitConfig = field(default_factory=lambda: HeirarchicalFitConfig())

    def make(self, data: Union[str, pd.DataFrame], pretrained_potentials: Dict[str, Union[str, TrainedPotential]] = None) -> Flow:

        trainers = []
        train_checkers = []
        job_list = []
        trained_potential = None

        if self.trainer_config.chemsys is None:
            raise ValueError("Chemical system must be provided in the trainer config.")
        
        if not isinstance(data, str):
            try:
                data_path = os.getcwd() + '/data.pckl.gzip'
                pd.to_pickle(data, data_path, compression='gzip', protocol=4)
            except:
                raise ValueError("Due to JobStore issues, data must be a path to a pickled dataframe in .pckl.gzip format OR an instance of a pd.DataFrame which is pickled in this call.")

        '''if pretrained_potential:
            if isinstance(pretrained_potential, str):
                trained_potential = TrainedPotential()
                trained_potential.read_potential(pretrained_potential)
                if os.path.isfile(pretrained_potential.replace(".yaml", ".asi")):
                    trained_potential.active_set_file = pretrained_potential.replace(".yaml", ".asi")

            if isinstance(pretrained_potential, TrainedPotential):
                trained_potential = pretrained_potential
            else:
                raise ValueError("Pretrained potential must be a path to a yaml file or an instance of the TrainedPotential class.")'''
        
        bbasis_order_map = {0: {"UNARY": UnaryBBasisOrder()}, 1: {"BINARY": BinaryBBasisOrder()}, 2: {"TERNARY": TernaryBBasisOrder()}, 3: {"QUARTERNARY": QuaternaryBBasisOrder()}, 4: {"QUINARY": QuinaryBBasisOrder()}}
        potential_shape_dict = {"UNARY": UnaryBBasisOrder(), "bonds": BBasisBonds(), "embedding": BBasisEmbedding()}

        self.hconfig.initial_potentials = pretrained_potentials

        for hiter in range(self.hconfig.start_order, self.hconfig.end_order):
            if hiter > len(self.trainer_config.chemsys):
                break
            potential_shape_dict.update(bbasis_order_map[hiter])
            self.trainer_config.bbasis_train_orders = list(np.arange(self.hconfig.start_order, hiter + 1))
            self.trainer_config.bbasis = potential_shape_dict
            if hiter > self.hconfig.start_order:
                self.trainer_config.max_steps = self.trainer_config.max_steps // 2
        
            for i, loss in enumerate(self.loss_weights):
                self.trainer_config.loss_weight = loss
                if i:
                    trained_potential = train_checkers[-1].output.trained_potential
                self.trainer_config.name = f"Step 0.{i} Trainer, Loss Weight: {self.loss_weights[i]}, Order: {hiter}"
                trainers.append(naive_train_ACE(data, trainer_config=self.trainer_config, trained_potential=trained_potential))
                trainers[-1].name = self.trainer_config.name
                train_checkers.append(check_training_output(trainers[-1].output, trainer_config=self.trainer_config))
                train_checkers[-1].name = self.trainer_config.name + " Checker"
            
            self.hconfig.initial_potentials.update({f"Order {hiter}": train_checkers[-1].output.trained_potential})

        self.trainer_config.bbasis = potential_shape_dict
        self.trainer_config.bbasis_train_orders = [range(self.hconfig.start_order, self.hconfig.end_order)]

        for i, loss in enumerate(self.loss_weights):
            self.trainer_config.loss_weight = loss
            if i:
                trained_potential = train_checkers[-1].output.trained_potential
            self.trainer_config.name = f"Step 0.{i} Trainer, Loss Weight: {self.loss_weights[i]}, Final Run"
            trainers.append(naive_train_ACE(data, trainer_config=self.trainer_config, trained_potential=trained_potential))
            trainers[-1].name = self.trainer_config.name
            train_checkers.append(check_training_output(trainers[-1].output, trainer_config=self.trainer_config))
            train_checkers[-1].name = self.trainer_config.name + " Checker"

        job_list.extend(trainers)
        job_list.extend(train_checkers)
        return Flow(job_list, output=train_checkers[-1].output, name=self.name)
