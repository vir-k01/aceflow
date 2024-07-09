#from ace_jobflow.temp_scripts import train_ACE, read_outputs, write_input
from aceflow.jobs.train import naive_train_ACE
from aceflow.jobs.data import read_MD_outputs, test_potential_in_restricted_space
from aceflow.utils.util import write_input
from aceflow.utils.structure_sampler import generate_test_points
from aceflow.utils.active_learning import get_active_set, select_structures_with_active_set
from aceflow.flows.data import DataGenFlowMaker, ActiveStructuresFlowMaker
from aceflow.flows.trainer import NaiveACEFlowMaker, NaiveACETwoStepFlowMaker, ACEMaker