from jobflow import job, Response, Flow
from typing import List, Union
from pymatgen.io.ase import AseAtomsAdaptor
from pyace import PyACECalculator
from aceflow.utils.structure_sampler import generate_test_points
from aceflow.utils.active_learning import psuedo_equilibrate_and_test, select_structures_with_active_set
from aceflow.utils.config import ActiveLearningConfig
import pandas as pd
import os

@job
def read_MD_outputs(md_outputs: List = None, precomputed_dataset: pd.DataFrame = None, step_skip: int= 1):
    energies = []
    forces = []
    structures = []
    if precomputed_dataset:
        energies = precomputed_dataset['energy']
        forces = precomputed_dataset['forces']
        structures = precomputed_dataset['ase_atoms']
    output = {}
    if md_outputs:
        for md_output in md_outputs:
            #trajectory = md_output.vasp_objects['trajectory']
            trajectory = md_output.forcefield_objects['trajectory']
            for frame_id in range(0, len(trajectory.frame_properties), step_skip):
                energies.append(trajectory.frame_properties[frame_id]['energy'])
                forces.append(trajectory.frame_properties[frame_id]['forces'])
                structures.append(AseAtomsAdaptor().get_atoms(trajectory.get_structure(frame_id)))
    output = {
            'energy': energies,
            'forces': forces,
            'ase_atoms': structures,
            'energy_corrected': energies,
            }
    return output

@job
def read_statics_outputs(statics: List = None):
    energies = []
    forces = []
    structures = []
    if statics:
        for static in statics:
            if static is not None:
                energies.append(static.output.energy)
                forces.append(static.output.forces)
                structures.append(AseAtomsAdaptor().get_atoms(static.output.structure))
    output = {
            'energy': energies,
            'forces': forces,
            'ase_atoms': structures,
            'energy_corrected': energies,
            }
    return output


@job
def consolidate_data(data: Union[List[dict], List[pd.DataFrame]]):
        
    energies = []
    forces = []
    structures = []
    for datum in data:
        if isinstance(datum, pd.DataFrame):
            datum = datum.to_dict()
        energies.extend(datum['energy'])
        forces.extend(datum['forces'])
        structures.extend(datum['ase_atoms'])
    return {'energy': energies, 'forces': forces, 'ase_atoms': structures}


@job
def read_pseudo_equilibration_outputs(outputs: pd.DataFrame):
    outputs = {'structures': AseAtomsAdaptor().get_structure(atoms) for atoms in outputs['ase_atoms']}
    return outputs

@job
def deferred_static_from_list(maker, structures):
    if type(structures) is list:
        static_jobs = [maker.make(structure) for structure in structures]
        static_outputs = [static_job.output for static_job in static_jobs]
        flow = Flow(static_jobs, output=static_outputs)
        return Response(replace=flow)
    else:
        return maker.make.original(maker, structures)

@job
def test_potential_in_restricted_space(prev_run_dict : dict, compositions: list, active_learning_config: ActiveLearningConfig):
    gamma_max = active_learning_config.gamma_max
    max_points = active_learning_config.max_points
    max_structures = active_learning_config.max_structures
    prev_dir = prev_run_dict['dir_name']
    if os.path.isfile(prev_dir + '/output_potential.yaml'):
        potential_file = prev_dir + "/output_potential.yaml"
    else:
        potential_file = prev_dir + '/interim_potential_0.yaml'
    if os.path.isfile(potential_file.replace(".yaml", ".asi")):
        active_set = potential_file.replace(".yaml", ".asi")
    else:
        active_set = potential_file.replace(".yaml", ".asi.nonlinear")
    active_set = potential_file.replace(".yaml", ".asi")
    base_calculator = PyACECalculator(potential_file)
    base_calculator.set_active_set(active_set)
    active_structures = []
    chemsys = [element.decode('utf-8') for element in list(base_calculator.elements_mapper_dict.keys())]
    test_points = generate_test_points(compositions, chemsys, iterations=3, max_points=max_points)
    for point in test_points:
        atoms, gamma = psuedo_equilibrate_and_test(base_calculator, point)
        if gamma > gamma_max:
            active_structures.append(atoms)
    df = pd.DataFrame({'ase_atoms': active_structures})
    df_selected = select_structures_with_active_set(potential_file, active_set, df, max_structures=max_structures)
    return [AseAtomsAdaptor().get_structure(structure) for structure in df_selected['ase_atoms']]