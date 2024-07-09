from dataclasses import dataclass, field
from jobflow import Maker, Flow
from atomate2.vasp.jobs.md import MDMaker
from ace_jobflow.flows.data import DataGenFlowMaker, ActiveStructuresFlowMaker
from ace_jobflow.jobs.data import read_MD_outputs
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
            read_job = read_MD_outputs(precomputed_dataset=precomputed_data, step_skip=self.step_skip)
            trainer = naive_train_ACE(read_job.output, max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_checker = check_training_output(trainer.output)
            return Flow([read_job, trainer, train_checker], output=train_checker.output, name=self.name)
        else: 
            data = DataGenFlowMaker(num_points=self.num_points, temperature=self.temperature, md_maker=self.md_maker).make(compositions, structures)
            read_job = read_MD_outputs(md_outputs = data.output, precomputed_dataset=precomputed_data, step_skip=self.step_skip)
            trainer = naive_train_ACE(read_job.output, max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
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
class ACEMaker(NaiveACEFlowMaker):
    '''
    Basic ACE trainer: Wrapper for pacemaker, also calls a series of md jobs on structures (both amorphous using packmol and crystalline, queried from MP). 
    Any precomputed data can be passed to the trainer too, as long as it is in the format required by pacemaker (pd.DataFrame with columns for energy, ase_atoms, forces and energy_corrected).
    The flow returns the directory with the output_potential.yaml, training log and reports.
    '''
    name : str = 'ACE Maker'
    md_maker : Maker = None
    static_maker : Maker = None
    max_structures : int = 10
    max_points : int = 1
    gamma_max : int = 1

    def make(self, compositions: list = None, precomputed_data: pd.DataFrame = None, structures: list = None):
        if compositions is None and structures is None:
            read_job = read_MD_outputs(precomputed_dataset=precomputed_data, step_skip=self.step_skip)
            trainer = naive_train_ACE(read_job.output, max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_checker = check_training_output(trainer.output)
            active_set_flow = ActiveStructuresFlowMaker(static_maker=self.static_maker, prev_dir=trainer.output, max_structures=self.max_structures, max_points=self.max_points, gamma_max=self.gamma_max).make(compositions)
            train_active = naive_train_ACE(computed_data_set=active_set_flow.output, prev_run_dict=train_checker.output, max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_active_checker = check_training_output(train_active.output)
            return Flow([read_job, trainer, train_checker, active_set_flow, train_active, train_active_checker], output=train_active_checker.output, name=self.name)
        else: 
            data = DataGenFlowMaker(num_points=self.num_points, temperature=self.temperature, md_maker=self.md_maker).make(compositions, structures)
            read_job = read_MD_outputs(md_outputs = data.output, precomputed_dataset=precomputed_data, step_skip=self.step_skip)
            trainer = naive_train_ACE(read_job.output, max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_checker = check_training_output(trainer.output)
            active_set_flow = ActiveStructuresFlowMaker(static_maker=self.static_maker, prev_dir=train_checker.output.dir_name, max_structures=self.max_structures, max_points=self.max_points, gamma_max=self.gamma_max).make(compositions)
            train_active = naive_train_ACE(computed_data_set=active_set_flow.output, prev_run_dict=train_checker.output.dir_name, max_steps=self.max_steps, batch_size=self.batch_size, gpu_index=self.gpu_index)
            train_active_checker = check_training_output(train_active.output)
            return Flow([data, read_job, trainer, train_checker, *active_set_flow, train_active, train_active_checker], output=train_active_checker.output, name=self.name)

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