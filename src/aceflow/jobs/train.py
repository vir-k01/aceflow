from jobflow import job
import pandas as pd
import subprocess
from aceflow.utils.util import write_input
from aceflow.core.active_learning import get_active_set
from aceflow.utils.config import TrainConfig
from aceflow.core.model import TrainedPotential
import os
import yaml
from typing import Union
from pymatgen.io.ase import AseAtomsAdaptor, MSONAtoms

@job
def naive_train_ACE(computed_data_set : Union[dict, pd.DataFrame] = None, trainer_config: TrainConfig = None, trained_potential: TrainedPotential = None) -> str:

    if isinstance(computed_data_set, dict):
        computed_data_set = pd.DataFrame.from_dict(computed_data_set)
    if computed_data_set.get('ase_atoms') is None:
        raise ValueError("Computed data set must contain ase_atoms column.")
    
    if isinstance(computed_data_set['ase_atoms'][0], MSONAtoms):
        processed_atoms = [AseAtomsAdaptor().get_atoms(AseAtomsAdaptor().get_structure(atoms), msonable=False) for atoms in computed_data_set['ase_atoms']]
        computed_data_set.drop(columns=['ase_atoms'], inplace=True)
        computed_data_set['ase_atoms'] = processed_atoms

    data_set = computed_data_set
    #data_set = pd.concat([computed_data_set, precomputed_dataset], axis=0, join="outer", ignore_index=False, keys=None)
    data_set.to_pickle("data.pckl.gzip", compression='gzip', protocol=4)
    write_input(trainer_config)

    if trained_potential is not None:
        potential = trained_potential.output_potential
        trained_potential.dump_potential(potential, 'continue.yaml')
        #prev_run_status = prev_run_dict['status']

        #with open("continue.yaml", 'w') as f:
        #    yaml.dump(potential, f, default_flow_style=False, sort_keys=False, Dumper=yaml.Dumper, default_style=None)

        #if prev_run_status == 'complete':
        subprocess.run("pacemaker -p continue.yaml input.yaml", shell=True)
        #else:
        #    naive_train_ACE(computed_data_set, num_basis, cutoff, loss_weight, max_steps, batch_size, gpu_index, prev_run_dict)
        #    write_input(num_basis, cutoff, loss_weight, batch_size, gpu_index, max_steps=100)
        #    subprocess.run("pacemaker -p continue.yaml input.yaml", shell=True)e)
    else:
        subprocess.run("pacemaker input.yaml", shell=True)
    
    return os.getcwd()

@job
def check_training_output(prev_run_dir: str) -> TrainedPotential:
    trained_potential = TrainedPotential(train_dir = prev_run_dir)
    trained_potential.read_training_dir()
    return trained_potential