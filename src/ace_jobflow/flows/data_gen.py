from typing import List
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from jobflow import Flow
from mp_api.client import MPRester
from atomate2.common.jobs.eos import _apply_strain_to_structure
from atomate2.common.jobs.structure_gen import get_random_packed
from atomate2.forcefields.md import PyACEMDMaker
from atomate2.vasp.jobs.md import MDMaker


def data_gen_flow(compositions: List | None, num_points: int = 5, temperature: float = 2000):
    with MPRester() as mpr:
        entries = mpr.get_entries(compositions, inc_structure=True, additional_criteria={"is_stable": True, "energy_above_hull": (0, 0)})
    
    working_structures = [entry.structure for entry in entries]
    for composition in compositions:
        working_structures.append(get_random_packed(composition))

    '''md_maker = PyACEMDMaker(**{"time_step": 2,
                        "n_steps": 10,
                        "temperature": temperature,
                        "calculator_kwargs": {'basis_set':'/pscratch/sd/v/virkaran/ace_test/train/29-6/2/output_potential.yaml'},
                        "traj_file": "test-ACE.traj",
                        "traj_file_fmt": "pmg",
                        "traj_interval": 1
    })'''
    md_maker = MDMaker(temperature=temperature, end_temp=temperature, steps=200)
    linear_strain = np.linspace(-0.2, 0.2, num_points)
    deformation_matrices = [np.eye(3) * (1.0 + eps) for eps in linear_strain]
    md_jobs = []
    md_outputs = []
    for structure in working_structures:
        deformed_structures = _apply_strain_to_structure(structure, deformation_matrices)
        for deformed_structure in deformed_structures:
            md_job = md_maker.make(deformed_structure.final_structure)
            md_job.update_metadata({"mp-id": id})
            md_job.update_metadata({"Composition": structure.composition.reduced_formula})
            md_job.name = f"{structure.composition.reduced_formula}_MD"
            md_outputs.append(md_job.output)
            md_jobs.append(md_job)
    return Flow([*md_jobs], output=md_outputs)