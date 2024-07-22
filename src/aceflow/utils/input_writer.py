from aceflow.utils.config import TrainConfig


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
        reference_energy_dict = {'Ba': -0.13385613, 'Ti': -1.21095265, 'O': -0.05486302, 'Zr': -1.31795132, 'Cl': -0.04836128, 'N': -0.05012608, 'Ca': -0.06538611}
    
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