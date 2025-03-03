from jobflow import job, Response, Flow, Maker
from typing import List, Union
from pymatgen.io.ase import AseAtomsAdaptor, MSONAtoms
from pyace import PyACECalculator
from aceflow.utils.structure_sampler import get_random_packed_points
from aceflow.active_learning.active_learning import run_NVT_MD, select_structures_with_active_set
from aceflow.utils.config import ActiveLearningConfig
from aceflow.utils.cleaner import dataframe_to_ace_dict
from aceflow.core.model import TrainedPotential
from aceflow.schemas.core import ACEDataTaskDoc
from aceflow.active_learning.base import BaseActiveLearningStrategy
from pymatgen.core.structure import Structure
import pandas as pd
import os

@job(acedata='acedata', output_schema=ACEDataTaskDoc)
def read_MD_outputs(md_outputs: List = None, step_skip: int = 1):
    energies = []
    forces = []
    structures = []
    if md_outputs:
        for md_output in md_outputs:
            if md_output:
                trajectory = md_output.vasp_objects['trajectory']
                #trajectory = md_output.forcefield_objects['trajectory']
                for frame_id in range(0, len(trajectory.frame_properties), step_skip):
                    energies.append(trajectory.frame_properties[frame_id]['e_0_energy'])
                    forces.append(trajectory.frame_properties[frame_id]['forces'])
                    structures.append(trajectory.get_structure(frame_id))
    
    data = {
            'energy': energies,
            'forces': forces,
            'ase_atoms': [AseAtomsAdaptor().get_atoms(structure) for structure in structures],
            'energy_corrected': energies,
            }
    
    doc = ACEDataTaskDoc(**{'acedata': data})
    doc.task_label = 'MD Outputs'
    return doc


@job(acedata='acedata', output_schema=ACEDataTaskDoc)
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
    data = {
            'energy': energies,
            'forces': forces,
            'ase_atoms': structures,
            'energy_corrected': energies,
            }
    
    doc = ACEDataTaskDoc(**{'acedata': data})
    doc.task_label = 'Static Outputs'
    return doc


@job(acedata='acedata', output_schema=ACEDataTaskDoc)
def consolidate_data(data: List[Union[dict, pd.DataFrame, str]]):
        
    energies = []
    forces = []
    structures = []
    for datum in data:
        if isinstance(datum, pd.DataFrame):
            datum = datum.to_dict(orient='list')
        if isinstance(datum, str):
            datum = pd.read_pickle(datum, compression='gzip').to_dict(orient='list')
        if datum is None:
            continue
        if not isinstance(datum['ase_atoms'][0], MSONAtoms):
            processed_atoms = [AseAtomsAdaptor().get_atoms(AseAtomsAdaptor().get_structure(atoms), msonable=True) for atoms in datum['ase_atoms']]
            datum['ase_atoms'] = processed_atoms
        energies.extend(datum['energy'])
        forces.extend(datum['forces'])
        structures.extend(datum['ase_atoms'])
    data = {'energy': energies, 'forces': forces, 'ase_atoms': structures, 'energy_corrected': energies}

    doc = ACEDataTaskDoc(**{'acedata': data})
    doc.task_label = 'Consolidated Data for ACE Training'
    return doc

@job
def deferred_static_from_list(maker, structures : List[Union[dict, Structure, MSONAtoms, list]]):

    all_structures = []
    for structure in structures:
        if isinstance(structure, dict):
            if 'ase_atoms' in structure.keys():
                all_structures.extend([AseAtomsAdaptor().get_structure(structure_) for structure_ in structure['ase_atoms']]) # if structure is a dict of structures
            else:
                raise ValueError("Invalid structure format. Must be a dict with 'ase_atoms' key.")
        if isinstance(structure, MSONAtoms):
            all_structures.extend([AseAtomsAdaptor().get_structure(structure) for structure in structures])
        if isinstance(structure, Structure):
            all_structures.extend([structure])
        if isinstance(structure, list):
            all_structures.extend([AseAtomsAdaptor().get_structure(structure_) for structure_ in structure])

    '''if isinstance(structures, dict):
        structures = [AseAtomsAdaptor().get_structure(structure) for structure in structures['ase_atoms']]

    if isinstance(structures, list):
        if isinstance(structures[0], MSONAtoms):
            structures = [AseAtomsAdaptor().get_structure(structure) for structure in structures]
        if isinstance(structures[0], dict):
            structures = [AseAtomsAdaptor().get_structure(structure) for structure_dict in structures for structure in structure_dict['ase_atoms']]'''

    static_jobs = [maker.make(structure) for structure in all_structures]
    static_outputs = [static_job.output for static_job in static_jobs]
    flow = Flow(static_jobs, output=static_outputs)
    return Response(replace=flow)

@job(acedata='acedata', output_schema=ACEDataTaskDoc)
def test_potential_in_restricted_space(trained_potential: Union[TrainedPotential, str], compositions: list, sampling_strategy: BaseActiveLearningStrategy = None, active_set_file: str = None):

    prev_dir = None
    if isinstance(trained_potential, TrainedPotential):
        TrainedPotential().dump_potential(trained_potential.output_potential, 'output_potential.yaml')
        potential_file = 'output_potential.yaml'
        active_set = trained_potential.active_set_file
    if isinstance(trained_potential, dict):
        TrainedPotential().dump_potential(trained_potential['output_potential'], 'output_potential.yaml')
        potential_file = 'output_potential.yaml'
        active_set = trained_potential['active_set_file']
    
    if isinstance(trained_potential, str):
        potential_file = trained_potential
        if active_set_file is not None:
            active_set = active_set_file
        else:
            active_set = potential_file.replace(".yaml", ".asi")

    '''else:
        prev_dir = trained_potential
    if prev_dir:
        if os.path.isfile(prev_dir + '/output_potential.yaml'):
            potential_file = prev_dir + "/output_potential.yaml"
        else:
            potential_file = prev_dir + '/interim_potential_0.yaml'
        active_set = potential_file.replace(".yaml", ".asi")'''

    sampler = sampling_strategy
    sampler.base_calculator = PyACECalculator(potential_file)
    sampler.base_calculator.set_active_set(active_set)

    active_structures = sampler.sample_structures(compositions)
    print(len(active_structures))
    df = pd.DataFrame({'ase_atoms': active_structures})
    df_selected = select_structures_with_active_set(potential_file, active_set, df, max_structures=sampler.max_structures)

    data = {'ase_atoms': list(df_selected['ase_atoms'])}#[AseAtomsAdaptor().get_structure(structure) for structure in df_selected['ase_atoms']]}
    doc = ACEDataTaskDoc(**{'acedata': data})
    doc.task_label = 'Active Structures Generation'
    return doc