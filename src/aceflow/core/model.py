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
    
    @classmethod
    def from_dir(cls, train_dir: str, dataset: pd.DataFrame = None):

        if not dataset:
            try:
                dataset = pd.read_pickle(train_dir + '/data.pckl.gzip', compression='gzip')
            except:
                raise FileNotFoundError("No dataset found in the training directory.")
        
        if os.path.isfile(train_dir + '/output_potential.yaml'):
            output_potential = cls.read_potential(cls, train_dir + '/output_potential.yaml')
            status = 'complete'
            if os.path.isfile(train_dir + '/output_potential.asi'):
                active_set_file = train_dir + '/output_potential.asi'
            else:   
                active_set_file = get_active_set(train_dir + '/output_potential.yaml', dataset=dataset, is_full=False)
        else:
            status = 'incomplete'
            if os.path.isfile(train_dir + '/interim_potential_0.asi'):
                active_set_file = train_dir + '/interim_potential_0.asi'
            else:
                active_set_file = get_active_set(train_dir + '/interim_potential_0.yaml', dataset=dataset, is_full=False)
        
        interim_potential = cls.read_potential(cls, train_dir + '/interim_potential_0.yaml')
    
        return cls(train_dir=train_dir, 
                   status=status, 
                   active_set_file=active_set_file, 
                   interim_potential=interim_potential, 
                   output_potential=output_potential if output_potential else None,
                   )