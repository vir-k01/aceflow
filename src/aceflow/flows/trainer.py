from dataclasses import dataclass, field
from jobflow import Maker, Flow
from atomate2.vasp.jobs.md import MDMaker
from aceflow.flows.data import DataGenFlowMaker, ActiveStructuresFlowMaker
from aceflow.utils.config import TrainConfig, DataGenConfig, ActiveLearningConfig
from aceflow.jobs.data import read_MD_outputs, consolidate_data, combine_potentials
from aceflow.jobs.train import naive_train_ACE, check_training_output
import pandas as pd
import os
import yaml
import itertools


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
    name : str = 'ACE Maker'
    trainer_config : TrainConfig = field(default_factory=lambda: TrainConfig())
    data_gen_config : DataGenConfig = field(default_factory=lambda: DataGenConfig())
    active_learning_config : ActiveLearningConfig = field(default_factory=lambda: ActiveLearningConfig())
    static_maker : Maker = None
    md_maker : Maker = None
    loss_weights = [0.99, 0.3]

    def make(self, compositions: list = None, precomputed_data: pd.DataFrame = None, structures: list = None, pretrained_potential: str = None):

        trainers = []
        train_checkers = []
        active_set_flows = []
        consolidate_data_jobs = []
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

        restart_dict = {}

        if pretrained_potential:
            if isinstance(pretrained_potential, str):
                with open(pretrained_potential, 'r') as f:
                    pretrained_potential = yaml.load(f, Loader=yaml.FullLoader)
                restart_dict.update({'potential': pretrained_potential})
                if os.path.isfile(pretrained_potential.replace(".yaml", ".asi")):
                    restart_dict.update({'active_set': pretrained_potential.replace(".yaml", ".asi")})
                prev_run_dict = restart_dict
            else:
                raise ValueError("Pretrained potential must be a path to a yaml file.")
        
        read_job = read_MD_outputs(md_outputs=data_output, precomputed_dataset=precomputed_data, step_skip=self.data_gen_config.step_skip)
        consolidate_data_jobs.append(consolidate_data([read_job.output]))
        job_list.append(read_job)

        for i, loss in enumerate(self.loss_weights):
            self.trainer_config.loss_weight = loss
            if i:
                prev_run_dict = train_checkers[-1].output
            #self.trainer_config.name = f"Step 0 Trainer, Loss Weight: {self.loss_weights[i]}"
            trainers.append(naive_train_ACE(consolidate_data_jobs[-1].output, trainer_config=self.trainer_config, prev_run_dict=prev_run_dict))
            train_checkers.append(check_training_output(trainers[-1].output))

        if self.active_learning_config.active_learning_loops:
            for i in range(self.active_learning_config.active_learning_loops):
                active_set_flow = ActiveStructuresFlowMaker(static_maker=self.static_maker, active_learning_config=self.active_learning_config).make(compositions, prev_run_dict=train_checkers[-1].output)
                active_set_flows.append(active_set_flow)
                consolidate_data_jobs.append(consolidate_data([consolidate_data_jobs[-1].output, active_set_flow.output]))
      
                for j, loss in enumerate(self.loss_weights):
                    self.trainer_config.loss_weight = loss
                    prev_run_dict = train_checkers[-1].output
                    #self.trainer_config.name = f"Active Step {i} Trainer, Loss Weight: {self.loss_weights[j]}"
                    trainers.append(naive_train_ACE(computed_data_set=consolidate_data_jobs[-1].output, prev_run_dict=prev_run_dict, trainer_config=self.trainer_config))
                    train_checkers.append(check_training_output(trainers[-1].output))

            job_list.extend(active_set_flows)
            job_list.extend(consolidate_data_jobs)

        job_list.extend(trainers)
        job_list.extend(train_checkers)
        return Flow(job_list, output=train_checkers[-1].output, name=self.name)
    
@dataclass
class HeiracricalACETrainer(ProductionACEMaker):

    name : str = 'Hierarchical ACE Trainer'
    pretrained_potential_dict : dict = None


    def make(self, compositions: list = None, precomputed_data: pd.DataFrame = None, structures: list = None):
        elements = set()
        for comp in compositions:
            elements.update(comp)
        for struct in structures:
            elements.update(struct.composition.elements)
        
        element_combinations = []
        for i in range(1, len(elements)+1):
            element_combinations.extend(itertools.combinations(elements, i))
            for j in range(1, len(elements)+1):
                element_combinations.extend(itertools.combinations(elements, j)) #abstract out to N-body potentials

        pretrained_potentials = {}
        if self.pretrained_potential_dict:
            for element_combination in element_combinations:
                if element_combination in self.pretrained_potential_dict:
                    pretrained_potentials[element_combination] = self.pretrained_potential_dict[element_combination]
                else:
                    pretrained_potentials[element_combination] = ProductionACEMaker(trainer_config=self.trainer_config, data_gen_config=self.data_gen_config, \
                                                                                    active_learning_config=self.active_learning_config, static_maker=self.static_maker, \
                                                                                    md_maker=self.md_maker, loss_weights=self.loss_weights).make(compositions=[element_combination]).output 
                    #take compositions as element_combinations from the hull of the elements in the combination

        pretrained_potential = combine_potentials(pretrained_potentials)

        with open("pretrained_potential.yaml", 'w') as f:
            yaml.dump(pretrained_potential, f)

        return ProductionACEMaker.make(compositions, precomputed_data, structures, pretrained_potential)
