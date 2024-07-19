from typing import Any, Optional, List
from pydantic import BaseModel, Field

class ACETrainerTaskDoc(BaseModel):
    '''
    Calculation output after ACE training. Contains the trained potential and other relevant information, can optionally include the dataset used for training.
    '''
    task_label: Optional[str] = Field(None, description='task_label')
    computed_data_set: Optional[Any] = Field(None, description='computed_data_set')
    trainer_config: Optional[Any] = Field(None, description='trainer_config')
    trained_potential: Optional[Any] = Field(None, description='trained_potential')
    log_file: Optional[List[str]] = Field(None, description='log_file')


class ACEDataTaskDoc(BaseModel):
    '''
    Calculation output after every data generation job in the ACE workflow. Contains the energy, forces, and structure of the system. Can optionally include the corrected energy.
    '''
    task_label : Optional[str] = Field(None, description='task_label')
    acedata: Optional[Any] = Field(None, description='acedata')