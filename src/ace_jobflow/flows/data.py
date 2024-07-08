from typing import List
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from jobflow import Flow, Maker
from mp_api.client import MPRester
from atomate2.common.jobs.eos import _apply_strain_to_structure
from atomate2.common.jobs.structure_gen import get_random_packed
from atomate2.forcefields.md import PyACEMDMaker
from atomate2.vasp.jobs.md import MDMaker
from atomate2.vasp.jobs.core import StaticMaker
from dataclasses import dataclass
from ace_jobflow.jobs.data import test_potential_in_restricted_space, read_pseudo_equilibration_outputs


@dataclass
class DataGenFlowMaker(Maker):
    name = "Data Generation Flow"
    md_maker : Maker = None
    num_points : int = 5
    temperature : float = 2000
    md_steps = 10

    def make(self, compositions: list = None, structures : list = None):

        working_structures = []
        if structures:
            working_structures.extend(structures)
        if compositions:
            with MPRester() as mpr:
                entries = mpr.get_entries(compositions, inc_structure=True, additional_criteria={"is_stable": True, "energy_above_hull": (0, 0)})

            working_structures = [entry.structure for entry in entries]
            for composition in compositions:
                working_structures.append(get_random_packed(composition))
        if self.md_maker is None:
            self.md_maker = PyACEMDMaker(**{"time_step": 2,
                            "n_steps": self.md_steps,
                            "temperature": self.temperature,
                            "calculator_kwargs": {'basis_set':'/pscratch/sd/v/virkaran/ace_test/train/29-6/2/output_potential.yaml'},
                            "traj_file": "test-ACE.traj",
                            "traj_file_fmt": "pmg",
                            "traj_interval": 1
        })
            #self.md_maker = MDMaker(temperature=self.temperature, end_temp=self.temperature, steps=self.md_steps)
        
        linear_strain = np.linspace(-0.2, 0.2, self.num_points)
        deformation_matrices = [np.eye(3) * (1.0 + eps) for eps in linear_strain]
        md_jobs = []
        md_outputs = []
        if working_structures:
            for structure in working_structures:
                deformed_structures = _apply_strain_to_structure(structure, deformation_matrices)
                for deformed_structure in deformed_structures:
                    md_job = self.md_maker.make(deformed_structure.final_structure)
                    md_job.update_metadata({"mp-id": id})
                    md_job.update_metadata({"Composition": structure.composition.reduced_formula})
                    md_job.name = f"{structure.composition.reduced_formula}_DataGen_MD"
                    md_outputs.append(md_job.output)
                    md_jobs.append(md_job)
            return Flow([*md_jobs], output=md_outputs)
        else:
            return None

@dataclass
class ActiveStructuresFlowMaker(Maker):
    name = "Active Structures Flow"
    static_maker : Maker = None
    gamma_max : int = 10
    max_points : int = 500
    potential : str = None
    active_set : str = None
    
    
    def make(self, compositions: list):

        active_structures = test_potential_in_restricted_space(self.potential, self.active_set, compositions, gamma_max=self.gamma_max, max_points=self.max_points)
        read_active_structures = read_pseudo_equilibration_outputs(active_structures.output)
        structures = read_active_structures.output

        if self.static_maker is None:
            self.static_maker = StaticMaker()
        static_jobs = []
        static_energy = []
        static_forces = []
        if structures:
            for structure in structures:
                static_job = self.static_maker.make(structure)
                static_job.name = f"{structure.composition.reduced_formula}_Active_Structure"
                static_jobs.append(static_job)
                static_energy.append(static_job.output.output.energy)
                static_forces.append(static_job.output.output.forces)
            
            return Flow([active_structures, read_active_structures, *static_jobs], output={"energy": static_energy, "forces": static_forces, "ase_atoms": active_structures["ase_atoms"], "energy_corrected": static_energy})
        else:
            return None