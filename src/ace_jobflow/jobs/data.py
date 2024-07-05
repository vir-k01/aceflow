from jobflow import job
from typing import List
from pymatgen.io.ase import AseAtomsAdaptor


@job
def read_outputs(md_outputs: List, step_skip: int= 1):
    energies = []
    forces = []
    structures = []
    for md_output in md_outputs:
        #trajectory = md_output.vasp_objects['trajectory']
        trajectory = md_output.forcefield_objects['trajectory']
        for frame_id in range(0, len(trajectory.frame_properties), step_skip):
            energies.append(trajectory.frame_properties[frame_id]['energy'])
            forces.append(trajectory.frame_properties[frame_id]['forces'])
            structures.append(AseAtomsAdaptor().get_atoms(trajectory.get_structure(frame_id)))
    df = {
            'energy': energies,
            'forces': forces,
            'ase_atoms': structures,
            'energy_corrected': energies
            }
    return df