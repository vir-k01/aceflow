from dataclasses import dataclass, field
from jobflow import Maker, Flow
from atomate2.vasp.jobs.md import MDMaker
from ace_jobflow.flows.data_gen import data_gen_flow
from ace_jobflow.jobs.data import read_outputs
from ace_jobflow.jobs.train import naive_train_ACE, check_training_output
import pandas as pd


@dataclass
class NaiveACEFlowMaker(Maker):
    '''
    Basic ACE trainer: Wrapper for pacemaker, also calls a series of md jobs on structures (both amorphous using packmol and crystalline, queried from MP). 
    Any precomputed data can be passed to the trainer too, as long as it is in the format required by pacemaker (pd.DataFrame with columns for energy, ase_atoms, forces and energy_corrected).
    The flow returns the directory with the output_potential.yaml, training log and reports.
    '''
    name : str = 'Naive ACE Trainer'
    md_maker : Maker = None #MDMaker = field(default_factory=lambda: MDMaker())
    num_points : int = 5
    temperature : float = 2000
    max_steps : int = 2000
    batch_size : int = 200
    step_skip : int = 1
    gpu_index : int = None

    def make(self, compositions: list = None, precomputed_data: pd.DataFrame = None, structures: list = None):
        if compositions is None and structures is None:
            read_job = read_outputs(precomputed_data=precomputed_data, step_skip=self.step_skip)
            trainer = naive_train_ACE(read_job.output, max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_checker = check_training_output(trainer.output)
            return Flow([read_job, trainer, train_checker], output=train_checker.output, name='ACE_wf_Test')
        else: 
            data = data_gen_flow(compositions, num_points=self.num_points, temperature=self.temperature, md_maker=self.md_maker, structures=structures)
            read_job = read_outputs(md_outputs = data.output, precomputed_data=precomputed_data, step_skip=self.step_skip)
            trainer = naive_train_ACE(read_job.output, pre_computed_dataset=precomputed_data, max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_checker = check_training_output(trainer.output)
            return Flow([data, read_job, trainer, train_checker], output=train_checker.output, name='ACE_wf_Test')
        
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
            read_job = read_outputs(precomputed_data=precomputed_data, step_skip=self.step_skip)
            train_step_1 = naive_train_ACE(read_job.output, loss_weight=self.loss_weights[0], max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_checker_1 = check_training_output(train_step_1.output)
            train_step_2 = naive_train_ACE(read_job.output, loss_weight=self.loss_weights[1], prev_run_dict=train_checker_1.output, max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_checker_2 = check_training_output(train_step_2.output)
            return Flow([read_job, train_step_1, train_checker_1, train_step_2, train_checker_2], output=train_checker_2.output, name='ACE_wf_Test')
        else: 
            data = data_gen_flow(compositions, num_points=self.num_points, temperature=self.temperature, md_maker=self.md_maker, structures=structures)
            read_job = read_outputs(md_outputs= data.output, precomputed_data=precomputed_data, step_skip=self.step_skip)
            train_step_1 = naive_train_ACE(read_job.output, loss_weight=self.loss_weights[0], max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_checker_1 = check_training_output(train_step_1.output)
            train_step_2 = naive_train_ACE(read_job.output, loss_weight=self.loss_weights[1], prev_run_dict=train_checker_1.output, max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_checker_2 = check_training_output(train_step_2.output)
            return Flow([read_job, train_step_1, train_checker_1, train_step_2, train_checker_2], output=train_checker_2.output, name='ACE_wf_Test')

'''
def naive_flow(compositions: list, num_points: int = 5, temperature: float = 2000, max_steps: int = 2000, batch_size: int = 200, gpu_index: int = None):
    data = data_gen_flow(compositions, num_points=num_points, temperature=temperature)
    read_job = read_outputs(data.output)
    trainer = naive_train_ACE(read_job.output, max_steps=max_steps, batch_size=batch_size, gpu_index=gpu_index)
    return Flow([data, read_job, trainer], output=trainer.output, name='ACE_wf_Test')

def naive_flow_two_step(compositions: list, num_points: int = 5, temperature: float = 2000, max_steps: int = 2000, batch_size: int = 200, gpu_index: int = None):
    data = data_gen_flow(compositions, num_points=num_points, temperature=temperature)
    read_job = read_outputs(data.output)
    train_step_1 = naive_train_ACE(read_job.output, loss_weight=0.99, max_steps=max_steps, batch_size=batch_size, gpu_index=gpu_index)
    train_step_2 = naive_train_ACE(read_job.output, loss_weight=0.3, prev_dir=train_step_1.output.dir_name, max_steps=max_steps, batch_size=batch_size, gpu_index=gpu_index)
    return Flow([data, read_job, train_step_1, train_step_2], output=train_step_2.output, name='ACE_wf_Test')'''