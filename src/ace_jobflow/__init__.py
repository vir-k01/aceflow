#from ace_jobflow.temp_scripts import train_ACE, read_outputs, write_input
from ace_jobflow.jobs.train import naive_train_ACE
from ace_jobflow.jobs.data import read_outputs
from ace_jobflow.utils.util import write_input
from ace_jobflow.utils.active_set_generator import get_active_set
from ace_jobflow.flows.data_gen import data_gen_flow
from ace_jobflow.flows.trainer import NaiveACEFlowMaker, NaiveACETwoStepFlowMaker