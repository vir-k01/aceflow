from jobflow import job, Flow, Response
import pandas as pd
import subprocess
from aceflow.utils.input_writer import write_input, flexible_input_writer, write_grace_input
from aceflow.schemas.core import ACETrainerTaskDoc
from aceflow.utils.config import TrainConfig, HeirarchicalFitConfig, GraceConfig
from aceflow.core.model import TrainedPotential, GraceModel
import os
from typing import Union
from pymatgen.io.ase import AseAtomsAdaptor, MSONAtoms

@job
def naive_train_ACE(computed_data_set : Union[dict, pd.DataFrame, str] = None, trainer_config: TrainConfig = None, trained_potential: TrainedPotential = None) -> str:

    if isinstance(computed_data_set, dict):
        if computed_data_set.get('ase_atoms') is None:
            raise ValueError("Computed data set must contain ase_atoms column.")
        computed_data_set = pd.DataFrame.from_dict(computed_data_set)

    if isinstance(computed_data_set, str):
        try:
            computed_data_set = pd.read_pickle(computed_data_set, compression='gzip')
        except:
            raise FileNotFoundError("No data found in the provided directory.")
    
    if isinstance(computed_data_set, pd.DataFrame):
        if isinstance(computed_data_set['ase_atoms'][0], MSONAtoms):
            processed_atoms = [AseAtomsAdaptor().get_atoms(AseAtomsAdaptor().get_structure(atoms), msonable=False) for atoms in computed_data_set['ase_atoms']]
            computed_data_set.drop(columns=['ase_atoms'], inplace=True)
            computed_data_set['ase_atoms'] = processed_atoms
        computed_data_set.to_pickle("data.pckl.gzip", compression='gzip', protocol=4)
    
    if trainer_config.gpu_index == -1:
        from tensorflow.config import list_physical_devices
        if list_physical_devices('GPU'):
            trainer_config.gpu_index = 0
        
    if trainer_config.ladder_type:
        trainer_config.upfit = True
    
    flexible_input_writer(trainer_config)
    init_control = '-ip' if trainer_config.upfit else '-p'

    if trained_potential is not None:
        if isinstance(trained_potential, dict):
            trained_potential = TrainedPotential.from_dict(trained_potential)
        potential = trained_potential.output_potential
        TrainedPotential().dump_potential(potential, 'continue.yaml')
        subprocess.run(f"pacemaker {init_control} continue.yaml input.yaml", shell=True)
    else:
        subprocess.run("pacemaker input.yaml", shell=True)
    
    return os.getcwd()

@job(output_schema=ACETrainerTaskDoc)
def check_training_output(prev_run_dir: str, 
                          trainer_config: TrainConfig | GraceConfig = None) -> TrainedPotential | GraceModel:
    
    if isinstance(trainer_config, TrainConfig):
        trained_potential = TrainedPotential.from_dir(prev_run_dir)
    else:
        trained_potential = GraceModel.from_dir(prev_run_dir)

    if trained_potential.status == 'incomplete':
        if trainer_config.restart_failed_runs:
            trainer_config.max_steps = trainer_config.max_steps // 2
            if isinstance(trainer_config, TrainConfig):
                return Response(addition=naive_train_ACE(computed_data_set=prev_run_dir, 
                                                         trainer_config=trainer_config, 
                                                         trained_potential=trained_potential))
            else:
                return Response(addition=naive_train_grace(computed_data_set=prev_run_dir, 
                                                           trainer_config=trainer_config, 
                                                           trained_potential=trained_potential))

    #dataset = pd.read_pickle(prev_run_dir + '/data.pckl.gzip', compression='gzip')
    
    with open(os.path.join(trained_potential.train_dir, 'log.txt')) as f:
        log = f.readlines()
    
    doc_data = {'log_file': log,
                'trainer_config': trained_potential.trainer_config,
                'trained_potential': trained_potential,
                'train_dir': trained_potential.train_dir}

    doc = ACETrainerTaskDoc(**doc_data)
    doc.task_label = trainer_config.name
    return doc

@job
def naive_train_hACE(computed_data_set : Union[dict, pd.DataFrame, str] = None, trainer_config: TrainConfig = None, trained_potential: TrainedPotential = None, initial_potentials: dict = None) -> str:

    if isinstance(computed_data_set, dict):
        if computed_data_set.get('ase_atoms') is None:
            raise ValueError("Computed data set must contain ase_atoms column.")
        computed_data_set = pd.DataFrame.from_dict(computed_data_set)

    if isinstance(computed_data_set, str):
        computed_data_set = pd.read_pickle(computed_data_set, compression='gzip')
        
    if isinstance(computed_data_set, pd.DataFrame):
        if isinstance(computed_data_set['ase_atoms'][0], MSONAtoms):
            processed_atoms = [AseAtomsAdaptor().get_atoms(AseAtomsAdaptor().get_structure(atoms), msonable=False) for atoms in computed_data_set['ase_atoms']]
            computed_data_set.drop(columns=['ase_atoms'], inplace=True)
            computed_data_set['ase_atoms'] = processed_atoms
        computed_data_set.to_pickle("data.pckl.gzip", compression='gzip', protocol=4)
    
    if trainer_config.gpu_index == -1:
        from tensorflow.config import list_physical_devices
        if list_physical_devices('GPU'):
            trainer_config.gpu_index = 0
    
    if trainer_config.ladder_type:
        trainer_config.upfit = True

    trainer_config.initial_potentials = initial_potentials if initial_potentials else None
    
    flexible_input_writer(trainer_config)
    init_control = '-ip' if trainer_config.upfit else '-p'

    if trained_potential is not None:
        if isinstance(trained_potential, dict):
            trained_potential = TrainedPotential.from_dict(trained_potential)
        potential = trained_potential.output_potential
        TrainedPotential().dump_potential(potential, 'continue.yaml')
        subprocess.run(f"pacemaker {init_control} continue.yaml input.yaml", shell=True)
    else:
        subprocess.run("pacemaker input.yaml", shell=True)
    
    return os.getcwd()

@job
def naive_train_grace(computed_data_set : Union[dict, pd.DataFrame, str] = None, 
                      trainer_config: GraceConfig = None, 
                      trained_potential: GraceModel = None, 
                      ) -> str:
    
    if isinstance(computed_data_set, dict):
        if computed_data_set.get('ase_atoms') is None:
            raise ValueError("Computed data set must contain ase_atoms column.")
        computed_data_set = pd.DataFrame.from_dict(computed_data_set)

    if isinstance(computed_data_set, str):
        try:
            computed_data_set = pd.read_pickle(computed_data_set, compression='gzip')
        except:
            raise FileNotFoundError("No data found in the provided directory.")
    
    if isinstance(computed_data_set, pd.DataFrame):
        if isinstance(computed_data_set['ase_atoms'][0], MSONAtoms):
            processed_atoms = [AseAtomsAdaptor().get_atoms(AseAtomsAdaptor().get_structure(atoms), msonable=False) for atoms in computed_data_set['ase_atoms']]
            computed_data_set.drop(columns=['ase_atoms'], inplace=True)
            computed_data_set['ase_atoms'] = processed_atoms
        if computed_data_set.get('stress') is not None:
            trainer_config.stress_control = ''
        computed_data_set.to_pickle("data.pckl.gzip", compression='gzip', protocol=4)
    
    write_grace_input(trainer_config)
    run_cmd = "gracemaker"
    
    if trained_potential:
        if isinstance(trained_potential, dict):
            trained_potential = GraceModel.from_dict(trained_potential)
        
        run_cmd += f" -r -p {trained_potential.model_yaml} -cn {trained_potential.model_checkpoint}"
    process = subprocess.run(run_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise RuntimeError(f"Error running gracemaker: {process.stderr.decode('utf-8')}")
    return os.getcwd()