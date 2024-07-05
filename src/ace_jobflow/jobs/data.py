from jobflow import job
from typing import List
from pymatgen.io.ase import AseAtomsAdaptor
import pandas as pd

@job
def read_outputs(md_outputs: List, precomputed_dataset: pd.DataFrame = None, step_skip: int= 1):
    energies = []
    forces = []
    structures = []
    output = {}
    if md_outputs is None:
        output = {"energy": precomputed_dataset['energy'], "forces": precomputed_dataset['forces'], "ase_atoms": precomputed_dataset['ase_atoms'], "energy_corrected": precomputed_dataset['energy_corrected']}
    else:
        for md_output in md_outputs:
            #trajectory = md_output.vasp_objects['trajectory']
            trajectory = md_output.forcefield_objects['trajectory']
            for frame_id in range(0, len(trajectory.frame_properties), step_skip):
                energies.append(trajectory.frame_properties[frame_id]['energy'])
                forces.append(trajectory.frame_properties[frame_id]['forces'])
                structures.append(AseAtomsAdaptor().get_atoms(trajectory.get_structure(frame_id)))
        output = {
                'energy': energies.extend(precomputed_dataset['energy']),
                'forces': forces.extend(precomputed_dataset['forces']),
                'ase_atoms': structures.extend(precomputed_dataset['ase_atoms']),
                'energy_corrected': energies.extend(precomputed_dataset['energy_corrected']),
                }
    return output