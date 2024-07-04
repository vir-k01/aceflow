from jobflow import job
import pandas as pd
import subprocess
from ..utils.util import write_input

@job
def naive_train_ACE(computed_data_set = None, pre_computed_dataset : pd.DataFrame = None, num_basis : int = 10, cutoff : int = 7, loss_weight : float = 0.9, max_steps: int = 2000, batch_size: int = 200, gpu_index: int = None, prev_dir: str = None):
    computed_data_set = pd.DataFrame.from_dict(computed_data_set)
    data_set = pd.concat([computed_data_set, pre_computed_dataset], axis=0, join="outer", ignore_index=False, keys=None)
    write_input(num_basis, cutoff, loss_weight, max_steps, batch_size, gpu_index)
    if prev_dir is not None:
        data_set.to_pickle(prev_dir + "/data.pckl.gzip", compression='gzip', protocol=4)
        try:
            subprocess.run("cp " + prev_dir + "/output_potential.yaml ./continue.yaml", shell=True)
        except:
            subprocess.run("cp " + prev_dir + "/interim_potential.yaml ./continue.yaml", shell=True)
        
        subprocess.run("pacemaker -p continue.yaml input.yaml", shell=True)
    else:
        data_set.to_pickle("data.pckl.gzip", compression='gzip', protocol=4)
        subprocess.run("pacemaker input.yaml", shell=True)
        #subprocess.run("pace_activeset -d fitting_data_info.pckl.gzip output_potential.yaml")