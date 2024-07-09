from typing import List
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from jobflow import Flow, Maker, job
from mp_api.client import MPRester
from atomate2.common.jobs.eos import _apply_strain_to_structure
from atomate2.common.jobs.structure_gen import get_random_packed
from atomate2.forcefields.md import PyACEMDMaker
from atomate2.vasp.jobs.md import MDMaker
from atomate2.vasp.jobs.core import StaticMaker
from dataclasses import dataclass
from ace_jobflow.jobs.data import test_potential_in_restricted_space, deferred_static_from_list, read_statics_outputs


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
    gamma_max : int = 5
    max_points : int = 500
    potential : str = None
    active_set : str = None
    prev_dir : str = None
    max_structures : int = 200
    if prev_dir:
        potential = prev_dir + "/output_potential.yaml"
        active_set = potential.replace(".yaml", ".asi")

    def make(self, compositions: list):

        active_structures = test_potential_in_restricted_space(self.potential, self.active_set, compositions, gamma_max=self.gamma_max, max_points=self.max_points, max_structures=self.max_structures)
        structures = active_structures.output
        statics_outputs = []
        statics = []
        if self.static_maker is None:
            self.static_maker = StaticMaker()
        for i in range(self.max_structures):
            statics.append(deferred_static_from_list(self.static_maker, structures, i))
            statics_outputs.append(statics[-1].output)
        output_reader = read_statics_outputs(statics_outputs)
        return Flow([active_structures, *statics, output_reader], output=output_reader.output)