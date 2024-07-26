import numpy as np
from jobflow import Flow, Maker, OnMissing
from mp_api.client import MPRester
from atomate2.common.jobs.eos import _apply_strain_to_structure
from atomate2.common.jobs.structure_gen import get_random_packed
from atomate2.forcefields.md import PyACEMDMaker
from atomate2.vasp.jobs.md import MDMaker
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.powerups import update_user_incar_settings, update_user_kpoints_settings
from dataclasses import dataclass, field
from aceflow.jobs.data import test_potential_in_restricted_space, deferred_static_from_list, read_statics_outputs, read_MD_outputs
from aceflow.utils.config import DataGenConfig, ActiveLearningConfig
from aceflow.core.model import TrainedPotential
from aceflow.active_learning.core import RandomPackedSampler, HighTempMDSampler
from aceflow.active_learning.base import BaseActiveLearningStrategy
from pymatgen.io.ase import AseAtomsAdaptor
from typing import List

@dataclass
class DataGenFlowMaker(Maker):
    name = "Data Generation Flow"
    data_gen_config : DataGenConfig = field(default_factory=lambda: DataGenConfig())
    md_maker : MDMaker = None
    static_maker : StaticMaker = None

    def make(self, compositions: list = None, structures : list = None):

        working_structures = []
        if structures:
            working_structures.extend(structures)
        if compositions:
            with MPRester() as mpr:
                entries = mpr.get_entries(compositions, inc_structure=True, additional_criteria={"is_stable": True, "energy_above_hull": (0, self.data_gen_config.max_energy_above_hull)})

            working_structures = [entry.structure.make_supercell(2, 2, 2) if len(entry.structure) < 50 else entry.structure for entry in entries]
            for composition in compositions:
                working_structures.append(get_random_packed(composition, vol_exp=1.2))
        if self.md_maker is None:
            self.md_maker = PyACEMDMaker(**{"time_step": 2,
                            "n_steps": self.data_gen_config.md_steps,
                            "temperature": self.data_gen_config.temperature,
                            "calculator_kwargs": {'basis_set':'/pscratch/sd/v/virkaran/ace_test/train/29-6/2/output_potential.yaml'},
                            "traj_file": "test-ACE.traj",
                            "traj_file_fmt": "pmg",
                            "traj_interval": 1
        })
            #self.md_maker = MDMaker(temperature=self.data_gen_config.temperature, end_temp=self.data_gen_config.temperature, steps=self.data_gen_config.md_steps)
            #self.maker = update_user_incar_settings(self.maker, self.data_gen_config.incar_updates)
            #self.maker = update_user_kpoints_settings(self.maker, self.data_gen_config.kpoints)
        
        if self.static_maker is None:
            self.static_maker = StaticMaker()
            #if self.data_gen_config.data_generator == 'MD':
            #    self.static_maker = update_user_incar_settings(self.static_maker, self.data_gen_config.incar_updates)
            #    self.static_maker = update_user_kpoints_settings(self.static_maker, self.data_gen_config.kpoints)
        
        linear_strain = np.linspace(-0.25, 0.25, self.data_gen_config.num_points)
        deformation_matrices = [np.eye(3) * (1.0 + eps) for eps in linear_strain]
        perturbation_magnitudes = np.linspace(0.01, 0.05, self.data_gen_config.num_points)
        jobs_list = []
        md_outputs = []
        static_outputs = []

        if working_structures:
            for structure in working_structures:
                deformed_structures = _apply_strain_to_structure(structure, deformation_matrices)
                perturbed_structures = [structure.copy() for structure in deformed_structures]
                for i, perturbation_magnitude in enumerate(perturbation_magnitudes):
                    for j in range(self.data_gen_config.num_points):
                        perturbed_structures[i].perturb(perturbation_magnitude)

                for i, deformed_structure in enumerate(deformed_structures):
                    md_job = self.md_maker.make(deformed_structure.final_structure)
                    md_job.update_metadata({"Type": "AIMD"})
                    md_job.update_metadata({"Composition": structure.composition.reduced_formula})
                    md_job.update_metadata({"Strain": linear_strain[i]})
                    md_job.name = f"{structure.composition.reduced_formula}_DataGen_MD"
                    md_outputs.append(md_job.output)
                    jobs_list.append(md_job)

            MD_output_reader = read_MD_outputs(md_outputs, step_skip=self.data_gen_config.step_skip)
            MD_output_reader.config.on_missing_references = OnMissing.NONE

            jobs_list.append(MD_output_reader)

            if self.data_gen_config.data_generator in ['Static', 'Static_Defect']:
                static_jobs = deferred_static_from_list(maker=self.static_maker, structures=[MD_output_reader.output.acedata, perturbed_structures])
                output_reader = read_statics_outputs(static_jobs.output)
                output_reader.config.on_missing_references = OnMissing.NONE
                jobs_list.append(static_jobs)
                jobs_list.append(output_reader)

            return Flow([*jobs_list], output=jobs_list[-1].output.acedata)

@dataclass
class ActiveStructuresFlowMaker(Maker):
    name = "Active Structures Flow"
    static_maker : Maker = None
    data_gen_config : DataGenConfig = None
    active_learning_config : ActiveLearningConfig = field(default_factory=lambda: ActiveLearningConfig())
  
    def make(self, compositions: list, trained_potential: TrainedPotential):

        if self.active_learning_config.sampling_strategy is None:
            self.active_learning_config.sampling_strategy = RandomPackedSampler()
    
        
        if self.active_learning_config.sampling_strategy.gamma_high is None:
            self.active_learning_config.sampling_strategy.gamma_high = self.active_learning_config.gamma_high
        
        if self.active_learning_config.sampling_strategy.gamma_low is None:
            self.active_learning_config.sampling_strategy.gamma_low = self.active_learning_config.gamma_low
        
        structures = []

        if isinstance(self.active_learning_config.sampling_strategy, BaseActiveLearningStrategy):
            self.active_learning_config.sampling_strategy = [self.active_learning_config.sampling_strategy]
        
        if isinstance(self.active_learning_config.sampling_strategy, List):
            for strategy in self.active_learning_config.sampling_strategy:
                active_structures = test_potential_in_restricted_space(trained_potential, compositions, sampling_strategy=strategy)
                structures.append(active_structures.output.acedata)
        
        if self.static_maker is None:
            self.static_maker = StaticMaker()
            #self.static_maker = update_user_incar_settings(self.static_maker, self.data_gen_config.incar_updates)
            #self.static_maker = update_user_kpoints_settings(self.static_maker, self.data_gen_config.kpoints)
        static_jobs = deferred_static_from_list(self.static_maker, structures)
        output_reader = read_statics_outputs(static_jobs.output)
        output_reader.config.on_missing_references = OnMissing.NONE
        return Flow([active_structures, static_jobs, output_reader], output=output_reader.output.acedata)