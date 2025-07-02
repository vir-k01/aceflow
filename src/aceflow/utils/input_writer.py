from aceflow.utils.config import TrainConfig, GraceConfig
from aceflow.reference_objects.BBasis_classes import *
from aceflow.core.model import TrainedPotential
from string import Template
from monty.serialization import dumpfn
import os

CWD = os.path.dirname(os.path.abspath(__file__))

def write_input(trainer_config : TrainConfig, reference_energy_dict: dict = None):
    num_basis = trainer_config.num_basis
    cutoff = trainer_config.cutoff
    loss_weight = trainer_config.loss_weight
    max_steps = trainer_config.max_steps
    batch_size = trainer_config.batch_size
    gpu_index = trainer_config.gpu_index
    ladder_step = trainer_config.ladder_step
    ladder_type = trainer_config.ladder_type
    test_size = trainer_config.test_size
    chemsys = trainer_config.chemsys
  

    if isinstance(ladder_step, int):
        ladder_step = [ladder_step]
      
    if isinstance(chemsys, dict):
        reference_energy_dict = chemsys.copy()
        chemsys = list(chemsys.keys())

    gpu_index_str = '-1' if gpu_index is None else str(gpu_index)

    ladder_control = '#'
    if ladder_type:
        ladder_control = ''
        ladder_step = [str(step) for step in ladder_step]

    if reference_energy_dict is None:
        reference_energy_dict = {'Ba': -0.13385613, 'Ti': -1.21095265, 'O': -0.05486302, 'Zr': -1.31795132, 'Cl': -0.04836128, 'N': -0.05012608, 'Ca': -0.06538611, 'Li': -0.04367102, 'Al': -0.02012669, 'Y': -1.96075795}
    
    content = f"""
  cutoff: {cutoff} # cutoff for neighbour list construction
  seed: 42  # random seed

  #################################################################
  ## Metadata section
  #################################################################
  metadata:
    origin: "Automatically generated input via ACEMaker"

  #################################################################
  ## Potential definition section
  #################################################################
  potential:
    #filename: continue.yaml
    deltaSplineBins: 0.001
    elements: {chemsys}

    embeddings:
      ALL: {{
        npot: 'FinnisSinclairShiftedScaled',
        fs_parameters: [ 1, 1, 1, 0.5, 1, 0.75, 1, 0.25, 1, 0.125, 1, 0.375, 1, 0.875, 1, 2],
        ndensity: 8,
      }}

    bonds:
      ALL: {{
        radbase: SBessel,
        radparameters: [ 5.25 ],
        rcut: 5.0,
        dcut: 0.01,
        NameOfCutoffFunction: cos,
      }}

    functions:
      number_of_functions_per_element: {num_basis}
      UNARY:   {{ nradmax_by_orders: [ 15, 6, 4, 3, 2, 2 ], lmax_by_orders: [ 0 , 3, 3, 2, 2, 1 ]}}
      BINARY:  {{ nradmax_by_orders: [ 15, 6, 3, 2, 2, 1 ], lmax_by_orders: [ 0 , 3, 2, 1, 1, 0 ]}}
      TERNARY: {{ nradmax_by_orders: [ 15, 3, 3, 2, 1 ],    lmax_by_orders: [ 0 , 2, 2, 1, 1 ], }}
      ALL:     {{ nradmax_by_orders: [ 15, 3, 2, 1, 1 ],    lmax_by_orders: [ 0 , 2, 2, 1, 1 ] }}

  #################################################################
  ## Dataset specification section
  #################################################################
  data:
    filename: data.pckl.gzip       # force to read reference pickled dataframe from given file
    test_size: {test_size}
    #  aug_factor: 1e-4 # common prefactor for weights of augmented structures
    reference_energy: {reference_energy_dict}

  #################################################################
  ## Fit specification section
  #################################################################
  fit:
    loss: {{ kappa: {loss_weight}, L1_coeffs: 1e-8,  L2_coeffs: 1e-8}}
    # if kappa: auto, then it will be determined from the variation of energy per atom and forces norms in train set

    optimizer: BFGS # or L-BFGS-B

    ## maximum number of minimize iterations
    maxiter: {max_steps}

    ## additional options for scipy.minimize
    #  options: {{maxcor: 100}}

    ## Automatically find the smallest interatomic distance in dataset  and set inner cutoff for ZBL to it
    repulsion: auto

    {ladder_control}ladder_step: {ladder_step}
    {ladder_control}ladder_type: {ladder_type}

    # Early stopping
    min_relative_train_loss_per_iter: 5e-5
    min_relative_test_loss_per_iter: 1e-5
    early_stopping_patience: 150

  #################################################################
  ## Backend specification section
  #################################################################
  backend:
    evaluator: tensorpot
    batch_size: {batch_size}
    batch_size_reduction: True
    batch_size_reduction_factor: 1.618
    display_step: 50
    gpu_config: {{gpu_ind: {gpu_index_str}, mem_limit: 0}}
  """

    with open("input.yaml", 'w') as file:
        file.write(content)


def flexible_input_writer(trainer_config : TrainConfig, reference_energy_dict: dict = None):
    num_basis = trainer_config.num_basis
    cutoff = trainer_config.cutoff
    loss_weight = trainer_config.loss_weight
    max_steps = trainer_config.max_steps
    batch_size = trainer_config.batch_size
    gpu_index = trainer_config.gpu_index
    ladder_step = trainer_config.ladder_step
    ladder_type = trainer_config.ladder_type
    test_size = trainer_config.test_size
    chemsys = trainer_config.chemsys
    bbasis_train_orders = trainer_config.bbasis_train_orders
    initial_potentials = trainer_config.initial_potentials


    embedding, bonds, unary, binary, ternary, quaternary, quinary, all_basis = [None]*8
    initial_potentials_paths = []

    initial_potentials_control = '#'
    
    if initial_potentials:
        initial_potentials_control = ''

        if isinstance(initial_potentials, dict):
            for k,p in initial_potentials.items():
                if isinstance(p, TrainedPotential):
                    TrainedPotential().dump_potential(p.interim_potential, 'initial_potential_'+k+'.yaml')
                if isinstance(p, dict):
                    TrainedPotential().dump_potential(p['interim_potential'], 'initial_potential_'+k+'.yaml')
                initial_potentials_paths.append('initial_potential_'+k+'.yaml')

    if isinstance(ladder_step, int):
        ladder_step = [ladder_step]
      
    if isinstance(chemsys, dict):
        reference_energy_dict = chemsys.copy()
        chemsys = list(chemsys.keys())

    basis_order_mapping = {}
    for name, bbasis in trainer_config.bbasis.items():
      if isinstance(bbasis, UnaryBBasisOrder):
        unary = bbasis
      if isinstance(bbasis, BinaryBBasisOrder):
        binary = bbasis
      if isinstance(bbasis, TernaryBBasisOrder):
        ternary = bbasis
      if isinstance(bbasis, QuaternaryBBasisOrder):
        quaternary = bbasis
      if isinstance(bbasis, QuinaryBBasisOrder):
        quinary = bbasis
      if isinstance(bbasis, AllBBasisOrder):
        all_basis = bbasis
      if isinstance(bbasis, BBasisBonds):
        bonds = bbasis
      if isinstance(bbasis, BBasisEmbedding):
        embedding = bbasis
      if isinstance(bbasis, FlowBBasisOrder):
        basis_order_mapping[bbasis.order] = bbasis

    func_order_control = {0: '#', 1: '#', 2: '#', 3: '#', 4: '#', -1: '#'}
    for order in basis_order_mapping.keys():
      if order < len(chemsys):
        func_order_control[order] = ''

    
    trainable_parameters_control = '' if bbasis_train_orders else '#'
    trainable_parameters = []
    if bbasis_train_orders:
      for order in bbasis_train_orders:
        if order < len(chemsys):
          trainable_parameters.append(basis_order_mapping[order].name)

    if embedding is None:
        embedding = BBasisEmbedding()
    if bonds is None:
        bonds = BBasisBonds()
    if unary is None:
        unary = UnaryBBasisOrder()
        func_order_control[0] = ''
    if binary is None:
        func_order_control[1] = '#'
        binary = BinaryBBasisOrder()
    if ternary is None:
        func_order_control[2] = '#'
        ternary = TernaryBBasisOrder()
    if quaternary is None:
        func_order_control[3] = '#'
        quaternary = QuaternaryBBasisOrder()
    if quinary is None:
        func_order_control[4] = '#'
        quinary = QuinaryBBasisOrder()
    if all_basis is None:
        func_order_control[-1] = '#'
        all_basis = AllBBasisOrder()
  
    gpu_index_str = '-1' if gpu_index is None else str(gpu_index)

    ladder_control = '#'
    if ladder_type:
        ladder_control = ''
        ladder_step = [str(step) for step in ladder_step]

    if reference_energy_dict is None:
        reference_energy_dict = {'Ba': -0.13385613, 'Ti': -1.21095265, 'O': -0.05486302, 'Zr': -1.31795132, 'Cl': -0.04836128, 'N': -0.05012608, 'Ca': -0.06538611}
    

    content = f"""
  cutoff: {cutoff} # cutoff for neighbour list construction
  seed: 42

  #################################################################
  ## Metadata section
  #################################################################
  metadata:
    origin: "Automatically generated input via ACEMaker"
  
  #################################################################
  ## Potential definition section
  #################################################################
  
  potential:

    deltaSplineBins: 0.001
    elements: {chemsys}

    embeddings:
      ALL: {{
        npot: {embedding.npot},
        fs_parameters: {embedding.fs_parameters},
        ndensity: {embedding.ndensity},
      }}
    
    bonds:
      ALL: {{
        radbase: {bonds.radbase},
        radparameters: {bonds.radparameters},
        rcut: {bonds.rcut},
        dcut: {bonds.dcut},
        NameOfCutoffFunction: {bonds.NameOfCutoffFunction},
      }}

    functions:
      number_of_functions_per_element: {num_basis}
      {func_order_control[0]}UNARY:   {{ nradmax_by_orders: {unary.nradmax_by_orders}, lmax_by_orders: {unary.lmax_by_orders} }}
      {func_order_control[1]}BINARY:  {{ nradmax_by_orders: {binary.nradmax_by_orders}, lmax_by_orders: {binary.lmax_by_orders} }}
      {func_order_control[2]}TERNARY: {{ nradmax_by_orders: {ternary.nradmax_by_orders}, lmax_by_orders: {ternary.lmax_by_orders} }}
      {func_order_control[3]}QUATERNARY: {{ nradmax_by_orders: {quaternary.nradmax_by_orders}, lmax_by_orders: {quaternary.lmax_by_orders} }}
      {func_order_control[4]}QUINARY: {{ nradmax_by_orders: {quinary.nradmax_by_orders}, lmax_by_orders: {quinary.lmax_by_orders} }}
      {func_order_control[-1]}ALL:     {{ nradmax_by_orders: {all_basis.nradmax_by_orders}, lmax_by_orders: {all_basis.lmax_by_orders} }}

    {initial_potentials_control}initial_potentials: {initial_potentials_paths}
    
  #################################################################
  ## Dataset specification section
  #################################################################
  data:
    filename: data.pckl.gzip
    test_size: {test_size}
    reference_energy: {reference_energy_dict}

  #################################################################
  ## Fit specification section
  #################################################################

  fit:
    loss: {{ kappa: {loss_weight}, L1_coeffs: 1e-8,  L2_coeffs: 1e-8}}
    optimizer: BFGS
    maxiter: {max_steps}
    repulsion: auto
    {trainable_parameters_control}trainable_parameters: {trainable_parameters}
    {ladder_control}ladder_step: {ladder_step}
    {ladder_control}ladder_type: {ladder_type}
    min_relative_train_loss_per_iter: 5e-5
    min_relative_test_loss_per_iter: 1e-5
    early_stopping_patience: 150

  #################################################################
  ## Backend specification section
  #################################################################

  backend:
    evaluator: tensorpot
    batch_size: {batch_size}
    batch_size_reduction: True
    batch_size_reduction_factor: 1.618
    display_step: 50
    gpu_config: {{gpu_ind: {gpu_index_str}, mem_limit: 0}}
  """

    with open("input.yaml", 'w') as file:
        file.write(content)


def write_grace_input(trainer_config : GraceConfig, reference_energy_dict: dict = None):
  if reference_energy_dict:
    trainer_config.reference_energy = reference_energy_dict
  else:
    trainer_config.reference_energy = 0
  
  subs_dict = trainer_config.__dict__.copy()
  
  subs_dict.update({f'{k}_control': '#' for k in ['stress', 'kwargs', 'custom', 'filename', 'checkpoint_name']})
  
  if trainer_config.finetune_foundation_model:
    template = Template(open(os.path.join(CWD, "reference_objects/grace_finetune_template.yaml")).read())
  else:
    trainer_config.preset_kwargs = trainer_config.preset_kwargs if trainer_config.preset_kwargs else {}
    subs_dict.update({'kwargs_control' : ''})
    
    if trainer_config.cutoff == 'n/a':
      if trainer_config.preset == 'FS':
        trainer_config.cutoff = 6
      elif trainer_config.preset == 'GRACE_1LAYER':
        trainer_config.cutoff = "DEFAULT_CUTOFF_1L"
      elif trainer_config.preset == 'GRACE_2LAYER':
        trainer_config.cutoff = "DEFAULT_CUTOFF_2L"
      else:
        raise ValueError("Cutoff must be provided for the chosen preset.")
      
    subs_dict.update({'cutoff': trainer_config.cutoff})
    subs_dict.update({'kwargs': trainer_config.preset_kwargs})
    if trainer_config.preset == 'custom':
      if trainer_config.filename and trainer_config.checkpoint_name:
        subs_dict.update({'filename_control': '', 'checkpoint_name_control': '', 'custom_control': ''})
      else:
        raise ValueError("Filename and checkpoint name must be provided for custom model.")
      
    template = Template(open(os.path.join(CWD, "reference_objects/grace_template.yaml")).read())
  
  dumpfn(trainer_config.as_dict(), 'trainer_config.yaml')
  content = template.substitute(subs_dict)
  with open("input.yaml", 'w') as file:
      file.write(content)
    
    