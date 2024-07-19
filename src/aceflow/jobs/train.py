from jobflow import job
import pandas as pd
import subprocess
from aceflow.utils.util import write_input
from aceflow.schemas.core import ACETrainerTaskDoc
from aceflow.utils.config import TrainConfig
from aceflow.core.model import TrainedPotential
import os
from typing import Union
from pymatgen.io.ase import AseAtomsAdaptor, MSONAtoms
from tensorflow.config import list_physical_devices

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
    if list_physical_devices('GPU'):
        trainer_config.gpu_index = 0
    
    data_set = computed_data_set
    data_set.to_pickle("data.pckl.gzip", compression='gzip', protocol=4)
    write_input(trainer_config)

    

    if trained_potential is not None:
        if isinstance(trained_potential, dict):
            trained_potential = TrainedPotential.from_dict(trained_potential)
        potential = trained_potential.output_potential
        TrainedPotential().dump_potential(potential, 'continue.yaml')
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

@job(output_schema=ACETrainerTaskDoc)
def check_training_output(prev_run_dir: str, trainer_config: TrainConfig = None) -> TrainedPotential:
    trained_potential = TrainedPotential(train_dir = prev_run_dir, trainer_config = trainer_config)
    trained_potential.read_training_dir()

    #dataset = pd.read_pickle(prev_run_dir + '/data.pckl.gzip', compression='gzip')
    with open(prev_run_dir + '/log.txt') as f:
        log = f.readlines()
    doc_data = {'log_file': log,
                'trainer_config': trainer_config,
                'trained_potential': trained_potential,
                'train_dir': prev_run_dir}

    doc = ACETrainerTaskDoc(**doc_data)
    doc.task_label = trainer_config.name
    return doc