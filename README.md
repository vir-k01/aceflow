# aceflow

(WIP) Wrapper for the pacemaker python package to use with jobflow for high-throughput training of ACE Machine Learning Interatomic Potentials.

Requirements:
- jobflow
- pymatgen == 2024.3.1
- pandas
- numpy == 1.26.4
- atomate2 (from https://github.com/BryantLi-BLI/atomate2.git)
- python-ace (from https://github.com/ICAMS/python-ace.git)
- TensotPotential (from https://github.com/ICAMS/TensorPotential.git)
- tensorflow >= 2.8.0
- ase (using ```pip install --upgrade git+https://gitlab.com/ase/ase.git@master```)

Most of the code here is built on top of the functionality offerred by the pacemaker python package. The flows implememted are "naive", in the sense that they run a data generation step, write out the approprite inputs and then call pacemaker to do the actual training, followed by a series of active learning steps. A more streamlined training process entirely in python has to be implemented. For now, the goal of these flows is to provide a full end-to-end training pipeline for ACE potentials for bulk systems, requiring little to no user intervention. The trained potentials can then be fine-tuned to the problem of interest by providing appropriate data.

Example usage:

Refer to the Jupyter Notebooks in the examples folder of this repo. 

Quick Start:
Say we want to train an ACE potential for the Ba-Ti-O composition space, and we already have some precomputed data, stored in a dataframe or a dict and certain structures (say of an interface or a defect) we want the potential to be trained on for whatever downstream usecase for the potential. To train with jobflow:
```
from aceflow.flows.trainer import ProductionACEMaker
from jobflow.managers.local import run_locally

compositions = ["BaTiO3"] #add more compositions of interest here! The Maker generates amorphous and a series of deformed crystalline structures of these compositions and runs AIMD on them to generate training data.
precomputed_data = ... #DFT data already computed for problem of interest, in the format ACE expects (pd.DataFrame or dict with keys: energy,ase_atoms,forces)
structures = ... # can be None, since the flow generates crystalline structures from MP too.
md_maker = ... #atomate2 MD Maker to run the MD steps. If not given, the default VASPMDMaker with the MPMorphMDSetGenerator will be run for 200 steps of 2fs MD.
static_maker = ... #atomate2 StaticMaker to compute the energies and forces for structures in the active learning loop. If none, the default StaticMaker with an input set consistent with the md_maker will be used.

flow = ProductionACEMaker().make(compositions, precomputed_data, structures)
output = run_locally(flow) #or on a workflow runner, such as FireWorks.
```
Then, once the flow has run, the trained potential can be accessed as:
```
potential = output[list(out.keys())[-1]][1].output.potential #or using the corresponding job UUID instead of the .keys() attribute.
# use this to write it out for use with ASE or LAMMPS
potential.dump('dump_file_name.yaml')
```
Any additional data generated in the flow can be accessed using the outputs of the consolidate_data job in the flow:
```
computed_data = out[list(out.keys())[-3]][1].output.acedata #or use the right UUID to get jobs with name "consildate_data"
```

The ProductionACEMaker does two steps of training, one with an emphasis on forces, followed by a more balanced weighting to energy and forces. This helps with the quality of potential trained, esp. for use in MLMD. The loss weights for each step can be provided when defining the flow as a list, and they default to [0.99, 0.3] for step 1 and 2 respectively. For example, to run the training thrice, with loss weights 0.99, 0.9 and then 0.3:
```
flow = ProductionACEMaker(loss_weights=[0.99, 0.9, 0.3]).make(compositions)
```

For more control over the flow, use the Config class objects from:
```
from acemaker.utils.config import TrainConfig, ActiveLearningConfig, DataGenConfig
```

