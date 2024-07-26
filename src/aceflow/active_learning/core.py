from aceflow.active_learning.base import BaseActiveLearningStrategy
from dataclasses import dataclass, field
from aceflow.utils.structure_sampler import get_random_packed_points
from atomate2.common.jobs.structure_gen import get_random_packed
from pymatgen.io.ase import AseAtomsAdaptor
from mp_api.client import MPRester
from aceflow.active_learning.active_learning import run_NVT_MD
from pyace.activeexploration import ActiveExploration
from pyace.basis import BBasisConfiguration


@dataclass
class RandomPackedSampler(BaseActiveLearningStrategy):
    name : str = 'Random Packed Active Learning'
    sampling_frequency : int = 5
    num_atoms : int = 100
    iterations : int = 3

    def sample_structures(self, compositions):
        chemsys = [element.decode('utf-8') for element in list(self.base_calculator.elements_mapper_dict.keys())]
        active_structures = []
        test_points = get_random_packed_points(compositions, chemsys = chemsys, iterations=self.iterations, sampling_frequency=self.sampling_frequency, max_points=self.max_structures)
        for point in test_points:
            atoms, gamma = run_NVT_MD(self.base_calculator, point)
        if gamma > self.gamma_low and gamma < self.gamma_high:
            active_structures.append(atoms)
        return active_structures


@dataclass
class HighTempMDSampler(BaseActiveLearningStrategy):
    name : str = 'High Temperature MD Active Learning'
    temperature : int = 5000
    md_steps : int = 20000
    sampling_frequency : int = 1000

    def sample_structures(self, compositions):

        structures = [get_random_packed(composition, vol_exp=1.2, target_atoms=self.target_atoms) for composition in compositions]
        with MPRester() as mpr:
            entries = mpr.get_entries(compositions, inc_structure=True, additional_criteria={"is_stable": True, "energy_above_hull": (0, 1)})
        structures.extend([entry.structure.make_supercell(2, 2, 2) if len(entry.structure) < 50 else entry.structure for entry in entries])

        atoms = [AseAtomsAdaptor().get_atoms(structure) for structure in structures]

        sample_steps = self.md_steps/self.sampling_frequency
        active_set = []

        for atom in atoms:
            for i in range(int(sample_steps)):
                atom, gamma = run_NVT_MD(calculator=self.base_calculator, atoms=atom, temperature=self.temperature, num_steps=self.sampling_frequency)
                if gamma > self.gamma_low and gamma < self.gamma_high:
                    active_set.append(atom)
                if gamma > self.gamma_high:
                    break
        
        return active_set
    

@dataclass
class ActiveExplorationSampler(BaseActiveLearningStrategy):
    name : str = 'Active Exploration Active Learning'
    bconf_file : str = None
    asi_file : str = None
    shake_max_attemps : int = 100
    shake_amplitude : float = 0.1

    def sample_structures(self, compositions):

        bconf = BBasisConfiguration(self.bconf_file)
        ae = ActiveExploration(bconf, self.asi_file)

        structures = [get_random_packed(composition, vol_exp=1.2, target_atoms=self.target_atoms) for composition in compositions]
        with MPRester() as mpr:
            entries = mpr.get_entries(compositions, inc_structure=True, additional_criteria={"is_stable": True, "energy_above_hull": (0, 1)})
        structures.extend([entry.structure.make_supercell(2, 2, 2) if len(entry.structure) < 50 else entry.structure for entry in entries])

        atoms = [AseAtomsAdaptor().get_atoms(structure) for structure in structures]

        active_set = []
        for atom in atoms:
            ae = ae.active_exploration(atoms=atom, min_dist=1.1, gamma_hi=self.gamma_high, gamma_lo=self.gamma_low, gamma_tol=1, n_atom_shake_max_attempts=self.shake_max_attemps, shake_amplitude=self.shake_amplitude)
            active_set.append(ae.extrapolative_structures)
        
        return active_set


@dataclass
class CHGNetGenSampler(BaseActiveLearningStrategy):
    name : str = 'CHGNet Generation Active Learning'
    temperature : int = 5000
    md_steps : int = 20000
    sampling_frequency : int = 1000

    def sample_structures(self, compositions):
        pass



            






        
        
