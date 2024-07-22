import numpy as np
import pandas as pd

from pyace import BBasisConfiguration, ACEBBasisSet, aseatoms_to_atomicenvironment, PyACECalculator
from pyace.activelearning import compute_B_projections, compute_active_set, compute_active_set_by_batches, \
    compute_A_active_inverse, compute_number_of_functions, \
    count_number_total_atoms_per_species_type, save_active_inverse_set

from pyace.aceselect import compute_mem_limit, compute_batch_size, compute_required_memory, select_structures_maxvol

from ase.md.langevin import Langevin
from ase import units


def get_active_set(potential_file: str, dataset: pd.DataFrame, batch_size_option: str = 'auto', is_full: bool = False, memory_limit: str = 'auto') -> str:
    gamma_tolerance = 1.01
    maxvol_iters = 300
    maxvol_refinement = 5
    memory_limit='auto'
    verbose=True

    mem_lim = compute_mem_limit(memory_limit)

    bconf = BBasisConfiguration(potential_file)
    bbasis = ACEBBasisSet(bconf)
    nfuncs = compute_number_of_functions(bbasis)

    if is_full:
        n_projections = [p * bbasis.map_embedding_specifications[st].ndensity for st, p in enumerate(nfuncs)]
    else:
        n_projections = nfuncs
    
    elements_to_index_map = bbasis.elements_to_index_map
    elements_name = bbasis.elements_name
    cutoffmax = bbasis.cutoffmax

    ATOMIC_ENV_COLUMN = "atomic_env"

    dataset[ATOMIC_ENV_COLUMN] = dataset["ase_atoms"].apply(aseatoms_to_atomicenvironment,
                                                      cutoff=cutoffmax, elements_mapper_dict=elements_to_index_map)
    
    atomic_env_list = dataset[ATOMIC_ENV_COLUMN]
    structure_ind_list = dataset.index
    total_number_of_atoms_per_species_type = count_number_total_atoms_per_species_type(atomic_env_list)

    required_active_set_memory, required_projections_memory = compute_required_memory(
        total_number_of_atoms_per_species_type, elements_name, nfuncs, n_projections, verbose)

    
    num_structures = len(atomic_env_list)
    batch_size = compute_batch_size(batch_size_option, mem_lim, num_structures, required_active_set_memory,
                                    required_projections_memory, verbose)

    if is_full:
        active_set_inv_filename = potential_file.replace(".yaml", ".asi.nonlinear")
    else:
        active_set_inv_filename = potential_file.replace(".yaml", ".asi")
    
    if batch_size is None:
        # single shot MaxVol
        A0_proj_dict, forces_dict = compute_B_projections(bbasis, atomic_env_list, is_full=is_full,
                                                          compute_forces_dict=True, verbose=verbose)
        A_active_set_dict = compute_active_set(A0_proj_dict, tol=gamma_tolerance, max_iters=maxvol_iters,
                                               verbose=verbose)
        A_active_inverse_set = compute_A_active_inverse(A_active_set_dict)

        with open(active_set_inv_filename, "wb") as f:
            np.savez(f, **{elements_name[st]: v for st, v in A_active_inverse_set.items()})

    else:
        # multiple round maxvol
        n_batches = len(atomic_env_list) // batch_size
        (best_gamma, best_active_sets_dict, _) = \
            compute_active_set_by_batches(
                bbasis,
                atomic_env_list=atomic_env_list,
                structure_ind_list=structure_ind_list,
                n_batches=n_batches,
                gamma_tolerance=gamma_tolerance,
                maxvol_iters=maxvol_iters,
                n_refinement_iter=maxvol_refinement,
                save_interim_active_set=True,
                is_full=is_full,
                verbose=verbose
            )
        A_active_inverse_set = compute_A_active_inverse(best_active_sets_dict)
        save_active_inverse_set(active_set_inv_filename, A_active_inverse_set, elements_name=elements_name)

    return active_set_inv_filename


def select_structures_with_active_set(potential_file: str, active_set: str, dataset: pd.DataFrame, max_structures: int = -1):

    asi_data = np.load(active_set)
    elements = sorted(asi_data.keys())
    asi_dict = {i: asi_data[el] for i, el in enumerate(elements)}

    bconf = BBasisConfiguration(potential_file)
    extra_A0_projections_dict = compute_A_active_inverse(asi_dict)
    df_selected = select_structures_maxvol(dataset, bconf, extra_A0_projections_dict, max_structures=max_structures)
    return df_selected


def psuedo_equilibrate_and_test(calculator: PyACECalculator, atoms):
    atoms.set_calculator(calculator)
    T=5000
    dyn = Langevin(atoms, 1 * units.fs, T * units.kB, 0.002)
    dyn.run(100)
    if atoms.get_kinetic_energy()/(1.5 * units.kB * T) > 1000:
        return [atoms, 1000]
    return [atoms, np.max(calculator.results['gamma'])]