import yaml
from aceflow.utils.config import TrainConfig, HeirarchyConfig
from dataclasses import dataclass
from montyt.json import MSONable
import itertools

@dataclass
class TrainedPotential(MSONable): #WIP
    name : str = 'Trained Potential'
    pretrained_potential_dict : dict = None
    heirarchy_config : HeirarchyConfig = None
    train_config : TrainConfig = None

    def make(self, chemsys):

        body_order = self.train_config.body_order
        elements = set()
        for comp in chemsys:
            elements.update(comp)
        
        element_combinations = []
        for i in range(1, body_order):
            element_combinations.extend(itertools.combinations(elements, i))
            for j in range(len(elements)):
                element_combinations.extend(itertools.combinations(elements[j], i))



    