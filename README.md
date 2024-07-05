# ACE_jobflow

(WIP) Wrapper for the pacemaker python package to use with jobflow for high-throughput training of ACE Machine Learning Interatomic Potentials.

Requirements:
- jobflow
- pymatgen == 2024.3.1
- pandas
- numpy == 1.26.4
- python-ace (from github repo)
- TensotPotential (from github repo)
- tensorflow >= 2.8.0
- ase (from github repo)

The flows implememted are "naive", in the sense that they simply run a data generation step, write out the approprite inputs and then call pacemaker to do the actual training. A more streamlined training process entirely in python has to be implemented, along with the active learning steps from pacemaker. 

Example usage:

Say we want to train an ACE potential for the Ba-Ti-O composition space, and we already have some precomputed data, stored in a dataframe and certain structures (say of an interface or a defect) we want the potential to be trained on for whatever downstream usecase for the potential. To train with jobflow:
```
from ace_jobflow.flows.trainer import NaiveACEFlowMaker, NaiveACETwoStepFlowMaker
from jobflow.managers.local import run_locally

compositions = ["BaTiO3"] #add more compositions of interest here! The Maker generates amorphous and a series of deformed crystalline structures of these compositions and runs AIMD on them to generate training data.
precomputed_data = ... #DFT data already computed for problem of interest, in the format ACE expects (pd.DataFrame with energy,ase_atoms,forces columns)
structures = ... #
md_maker = ... #atomate2 MD Maker to run the MD steps. If not given, the default VASPMDSet will be run for 200 steps of 2fs MD.

flow_kwargs = {'num_points': 1, 
               'batch_size': 1,
               'max_steps': 1,
               'md_maker': md_maker
               'gpu_index': -1} #Minimal working example, uses no gpus. 
            
flow = NaiveACEFlowMaker(**flow_kwargs).make(compositions, precomputed_data, structures)
output = run_locally(flow)
```
THEN, once the flow has run, the trained potential can be accessed as:
```
potential = out[list(out.keys())[-1]][1].output['potential']  #use yaml.dump(potential, output_file, default_flow_style=False, sort_keys=False, Dumper=yaml.Dumper, default_style=None) to write it out for use with ASE or LAMMPS
```
Any additional data generated in the flow can be accessed using the outputs of the read_outputs job in the flow:
```
computed_data = out[list(out.keys())[-3]][1].output
```

The NaiveACETwoStepFlowMaker does two steps of training, one with an emphasis on forces, followed by a more balanced weighting to energy and forces. 
This helps with the quality of potential trained, esp for MLMD. The loss weights for each step can be provided in the flow_kwargs as a list, and they default to [0.99, 0.3].

Finally, for more advanced training configurations (for example using ladder fitting), simply modify the write_inputs function:
```
from ace_jobflow.utils.util import write_inputs
write_inputs() #modidy the body of text in this function to change the input.yaml written for pacemaker.
```

