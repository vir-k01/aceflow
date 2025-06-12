from dataclasses import dataclass
from monty.json import MSONable
import yaml
import os
from aceflow.active_learning.active_learning import get_active_set
from aceflow.utils.config import TrainConfig, GraceConfig
import pandas as pd
import subprocess
from monty.serialization import loadfn

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
            output_potential = cls.read_potential(train_dir + '/output_potential.yaml')
            status = 'complete'
            active_set_file = get_active_set(train_dir + '/output_potential.yaml', dataset=dataset, is_full=False)
        else:
            status = 'incomplete'
            active_set_file = get_active_set(train_dir + '/interim_potential_0.yaml', dataset=dataset, is_full=False)
        
        interim_potential = cls.read_potential(train_dir + '/interim_potential_0.yaml')
    
        return cls(train_dir=train_dir, status=status, active_set_file=active_set_file, interim_potential=interim_potential)

@dataclass
class GraceModel(MSONable):
    train_dir: str = None
    model_yaml: str = None
    model_checkpoint: str = None
    active_set_file: str = None
    final_model: str = None
    status: str = None
    trainer_config: GraceConfig = None
    metadata: dict = None
    
    def read_model_yaml(self, model_yaml: str):
        with open(model_yaml, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    
    @classmethod
    def from_dir(cls, train_dir: str):
        
        if not os.path.isdir(os.path.join(train_dir, 'seed', '1')):
            raise FileNotFoundError("Training directory not found.")
            
        model = cls()
        model.train_dir = os.path.join(train_dir, 'seed', '1')
        
        model.model_yaml = os.path.join(model.train_dir, 'model.yaml')
        model.model_checkpoint = os.path.join(model.train_dir, 'checkpoints', 'checkpoint.best_test_loss.index')
        model.trainer_config = GraceConfig.from_dict(loadfn(os.path.join(train_dir, 'trainer_config.yaml')))
        
        if os.path.isdir(os.path.join(model.train_dir, 'final_model')):
            model.status = 'complete'
        else:
            model.status = 'incomplete'
                    
        if 'FS' in model.trainer_config.finetune_foundation_model or model.trainer_config.preset == 'FS':
            subprocess.run(f"gracemaker -r -s -sf")
            model.final_model = os.path.join(model.train_dir, 'FS_model.yaml')
            subprocess.run(f"pace_activeset -d {train_dir}/training_set.pkl.gz {model.final_model_yaml}")
            model.active_set_file = os.path.join(model.train_dir, 'FS_model.asi')
        else:
            subprocess.run(f"gracemaker -r -s")
            model.final_model = os.path.join(model.train_dir, 'saved_model')
        
        return model