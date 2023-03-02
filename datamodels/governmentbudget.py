import os
import pickle

from dataclasses import dataclass
import datetime as dt
from datetime import timedelta
from dataclasses import field

from typing import List

from datamodels.healthindicators import HealthIndicator
import csv
import yaml
from pandas import to_timedelta

@dataclass
class BudgetItem():
 
    name: int
    code: int
    final_date: dt.date
    period: timedelta

    description:str = None
    group_code: str = None
    group_name: str = None

    connetcted_health_indicators: List[HealthIndicator] = field(default_factory=list)
    
    connected_budget_items: List = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.final_date, str):
            
            args = list(reversed(self.final_date.split('-')))
            self.final_date = dt.date( *map(int,args) )


        if not isinstance(self.period, timedelta):
            
            self.period =  to_timedelta(self.period).to_numpy()
    
    def __equal__(self, budget_item):
        
        return self.name == budget_item.name

class GovernmentBudget():
    """
        Represents a local/regional/national govt's expenditure as the 
        different budget items that it can spend on.

        A 'method' can be used to draw 'connections' between budget items 
        and health indicators, this will 
    """
    info = {}

    def __init__(self, li_budget_item: List[BudgetItem] ):
        self.li_budget_items = li_budget_item

    def init_from_csv(fp:str) -> None:
        raise NotImplementedError
            
    def to_csv(self, fp:str):

        # Get the max number of health indicators related to a budget item

        # create a csv with this many columns - named H.I. 1, HI2, HI3 etc..
        # Each column will just contain the name of the health indicator

        max_count_hi = max( len(bi.connetcted_health_indicators) for bi in self.li_budget_items )

        with open(fp, 'w') as csvfile:
            fieldnames = ['budget'] +  [ f'HI {idx}' for idx in range(max_count_hi) ]
            writer = csv.DictWriter(csvfile, fieldnames = fieldnames, restval="",  )
            writer.writeheader()

            for budget_item in self.li_budget_items:
                dict_budget = {'budget': budget_item.name}
                dict_health_indicators = { f'HI {idx}':hi.name  for idx, hi in enumerate(budget_item.connetcted_health_indicators) }

                writer.writerow(
                    {**dict_budget, **dict_health_indicators}
                )

    @staticmethod
    def load_from_pickled(fp:str) -> None:
        return pickle.load(open(fp,"rb"))

    def to_file(self, fp):

        dir = os.path.dirname(fp)
        os.makedirs(dir, exist_ok=True)

        pickle.dump(self, open(fp,"wb"))

    def to_yaml(self, fp):
        """
            Saves a yaml file containing info about the dataset and
            method used to produce this matching of
            government budget spending to health indicators
        """
        dir = os.path.dirname(fp)
        os.makedirs(dir, exist_ok=True)

        yaml.dump( self.info, open(fp,'w'))

        
    
    
