from dataclasses import dataclass
from datetime import timedelta
import datetime as dt
from pandas import to_timedelta

@dataclass
class HealthIndicator():
    code:str
    name: str
    final_date: dt.date
    period: timedelta

    definition_short: str = None
    definition_long: str = None

    group_code:str = None
    group_name:str = None

    def __post_init__(self):
        if isinstance(self.final_date, str):
            
            args = reversed(self.final_date.split('-'))
            self.final_date = dt.date( *map(int,args) )


        if not isinstance(self.period, timedelta):
            
            self.period =  to_timedelta(self.period).to_numpy()


@dataclass
class GovernmentHealthIndicatorSet():
    
    name: str # Name of Local govt
    li_health_indicators:list = None

    def get_health_indicators_by_group(self, group_code:str=None, group_name:str=None ):

        assert not group_code or not group_name, "Please only specify one of group_code or group_name"

        attr, value = ( 'group_code', group_code ) if group_code else ( 'group_name', group_name)

        li_group_hi = [ hi for hi in self.li_health_indicators if getattr(hi, attr)==value ]

        return li_group_hi
        
            


