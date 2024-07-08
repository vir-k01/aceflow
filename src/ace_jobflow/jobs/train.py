from jobflow import job
import pandas as pd
import subprocess
from ace_jobflow.utils.util import write_input
from ace_jobflow.utils.active_learning import get_active_set
import os
import yaml

@job
def naive_train_ACE(computed_data_set : dict = None, num_basis : int = 10, cutoff : int = 7, loss_weight : float = 0.9, max_steps: int = 2000, batch_size: int = 200, gpu_index: int = None, prev_run_dict: dict = None) -> str:
    data_set = pd.DataFrame.from_dict(computed_data_set)
    #data_set = pd.concat([computed_data_set, precomputed_dataset], axis=0, join="outer", ignore_index=False, keys=None)
    data_set.to_pickle("data.pckl.gzip", compression='gzip', protocol=4)
    write_input(num_basis, cutoff, loss_weight, max_steps, batch_size, gpu_index)
    if prev_run_dict is not None:
        #data_set.to_pickle(prev_dir + "/data.pckl.gzip", compression='gzip', protocol=4)
        prev_dir = prev_run_dict['dir_name']
        potential = prev_run_dict['potential']
        prev_run_status = prev_run_dict['status']
        with open("continue.yaml", 'w') as f:
            yaml.dump(potential, f, default_flow_style=False, sort_keys=False, Dumper=yaml.Dumper, default_style=None)
        #if prev_run_status == 'complete':
        subprocess.run("pacemaker -p continue.yaml input.yaml", shell=True)
        #else:
        #    naive_train_ACE(computed_data_set, num_basis, cutoff, loss_weight, max_steps, batch_size, gpu_index, prev_run_dict)
        #    write_input(num_basis, cutoff, loss_weight, batch_size, gpu_index, max_steps=100)
        #    subprocess.run("pacemaker -p continue.yaml input.yaml", shell=True)e)
    else:
        subprocess.run("pacemaker input.yaml", shell=True)
        #subprocess.run("pace_activeset -d fitting_data_info.pckl.gzip output_potential.yaml")
    return os.getcwd()

@job
def check_training_output(prev_run_dir: str) -> dict:
    output_dict = {}
    if os.path.isfile(prev_run_dir + '/output_potential.yaml'):
        output_dict.update({'status': 'complete'})
        with open(prev_run_dir + '/output_potential.yaml', 'r') as f:
            output = yaml.load(f, Loader=yaml.FullLoader)
    else:
        output_dict.update({'status': 'incomplete'})
        with open(prev_run_dir + '/interim_potential_0.yaml', 'r') as f:
            output = yaml.load(f, Loader=yaml.FullLoader)
    df = pd.read_pickle(prev_run_dir + '/data.pckl.gzip', compression='gzip')
    active_set = get_active_set(output, df, is_full=False)
    output_dict.update({'potential': output, 'active_set': active_set,'dir_name': prev_run_dir})
    return output_dict