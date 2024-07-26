from dataclasses import dataclass
from monty.json import MSONable
import yaml
import os
from aceflow.active_learning.active_learning import get_active_set
from aceflow.utils.config import TrainConfig
import pandas as pd

@dataclass
class TrainedPotential(MSONable):

    train_dir: str = None
    output_potential: dict = None
    interim_potential: dict = None
    active_set_file: dict = None
    status: str = None
    trainer_config: TrainConfig = None
    metadata: dict = None

    def read_potential(self, potential_file: str) -> dict:
        with open(potential_file, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
        
    @staticmethod
    def dump_potential(potential: dict, filename: str = 'output_potential.yaml'):
        with open(filename, 'w') as f:
            yaml.dump(potential, f, default_flow_style=False, sort_keys=False, Dumper=yaml.Dumper, default_style=None)
    
    def read_training_dir(self, dataset: pd.DataFrame = None):

        if dataset is None:
            try:
                dataset = pd.read_pickle(self.train_dir + '/data.pckl.gzip', compression='gzip')
            except:
                raise FileNotFoundError("No dataset found in the training directory.")
        
        if os.path.isfile(self.train_dir + '/output_potential.yaml'):
            self.output_potential = self.read_potential(self.train_dir + '/output_potential.yaml')
            self.status = 'complete'
            active_set_file = get_active_set(self.train_dir + '/output_potential.yaml', dataset=dataset, is_full=False)
        else:
            self.status = 'incomplete'
            active_set_file = get_active_set(self.train_dir + '/interim_potential_0.yaml', dataset=dataset, is_full=False)
        
        self.active_set_file = active_set_file
        self.interim_potential = self.read_potential(self.train_dir + '/interim_potential_0.yaml')
