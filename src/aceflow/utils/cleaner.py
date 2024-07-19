import pandas as pd


def dataframe_to_ace_dict(df: pd.DataFrame) -> dict:
    """
    Convert a dataframe to an ACE training compatible dictionary.
    """
    energies = list(df['energy'])
    forces = list(df['forces'])
    structures = list(df['ase_atoms'])
    energy_corrected = energies
    return {'energy': energies, 'forces': forces, 'ase_atoms': structures, 'energy_corrected': energy_corrected}