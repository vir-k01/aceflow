import numpy as np
from pymatgen.core import Composition
from atomate2.common.jobs.structure_gen import get_random_packed
from pymatgen.io.ase import AseAtomsAdaptor
import re

def linear(x,m,b):
    return m*x + b

def formula_to_dict(formula, chemsys):
    pattern = r'([A-Z][a-z]*)(\d*)'
    matches = re.findall(pattern, formula)

    formula_dict = {element: 0 for element in chemsys}
    for element, count in matches:
        if count == '':
            count = 1
        else:
            count = int(count)
        formula_dict[element] = count

    return formula_dict

def get_composition_along_branches(alpha : list[float], branches : list, chemsys: list):

    comps = {}
    for branch, sl_icp_pairs in branches.items():
        comps[branch] = []
        for a in alpha:
            comps[branch].append(
                Composition({
                    chemsys[isli]: linear(a,*sl_icp)
                    for isli, sl_icp in enumerate(sl_icp_pairs)
                })
            )
    return comps

def get_branches_from_endpoints(composition_1, composition_2, chemsys):
    if isinstance(composition_1, str):
        composition_1 = formula_to_dict(composition_1, chemsys=chemsys)
    
    if isinstance(composition_2, str):
        composition_2 = formula_to_dict(composition_2, chemsys=chemsys)

    if isinstance(composition_1, Composition):
        composition_1 = composition_1.as_dict()
    composition_1 = {k: v/sum(composition_1.values()) for k, v in composition_1.items()}

        
    if isinstance(composition_2, Composition):
        composition_2 = composition_2.as_dict()
    composition_2 = {k: v/sum(composition_2.values()) for k, v in composition_2.items()}
    
    branch_name = Composition(composition_1).reduced_formula+'->'+Composition(composition_2).reduced_formula
    
    branch =  {branch_name: [(composition_2[ele]-frac, frac) for  ele, frac in composition_1.items()]}
    
    return branch


def get_boundary_branches(compositions: list, chemsys):
    main_branch = {}
    for i in range(len(compositions)-1):
        for j in range(i+1, len(compositions)):
            branch = get_branches_from_endpoints(compositions[i], compositions[j], chemsys=chemsys)
            main_branch = {**main_branch, **branch}
    return main_branch

def generate_test_points(compositions : list, chemsys : list, iterations : int = 3, max_points : int = 500):
    test_points = []
    num_sampled = 0
    base_branches = get_boundary_branches(compositions=compositions, chemsys=chemsys)
    num_edges = len(list(base_branches.keys()))
    dist = int((max_points // (num_edges)**2)**(1/iterations))
    for i in range(iterations):
        points = list(get_composition_along_branches(alpha=np.linspace(0, 1, dist), branches=base_branches, chemsys=chemsys).values())
        test_points.append([composition for sublist in points for composition in sublist])
        num_sampled += len(test_points[-1])
        dist = dist // 2
        if num_sampled > max_points:
            break
    
    test_compositions = [composition for sublist in test_points for composition in sublist]
    test_compositions = list(set(test_compositions))

    test_structures = [get_random_packed(composition, vol_exp=1.2) for composition in test_compositions for i in range(5)]

    test_structures_with_oxi = []
    test_atoms = []
    for i, test_structure in enumerate(test_structures):
        structure = test_structure.copy()
        try:
            test_structures_with_oxi.append(structure.add_oxidation_state_by_guess())
        except:
            test_structures.remove(test_structure)
        if not np.isclose(test_structures_with_oxi[-1].charge, 0):
            test_structures.remove(test_structure)
        test_atoms.append(AseAtomsAdaptor().get_atoms(test_structure))
        
    #TODO: add functionality to change num atoms in structure
    return test_atoms