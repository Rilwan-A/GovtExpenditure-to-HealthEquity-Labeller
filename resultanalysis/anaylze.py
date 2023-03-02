import pandas
import yaml

from typing import List

from datamodels.governmentbudget import GovernmentBudget

# class Analyze():

#     def __init__( govt_budget: GovernmentBudget):


class CrossAnalyzeMatching():
    """
        Provides the following method to compare different matchings 
            between government budget items and health indicators

    """



    def __init__( self, baseline_govt_budget:GovernmentBudget, li_govt_budget:List[GovernmentBudget] ):
        """ The first govt budget item will be assumed to the base for evaluations

            Each govt budget must have had its budget items matched to specific health indicators

        Args:
            baseline_govt_budget (GovernmentBudget): Baseline govt budget for all evaluations
            li_govt_budget (List[GovernmentBudget]): List of GovernmentBudget Containing the matchings to evaluate. 
        """
        self.base_govt_budget = li_govt_budget
        self.li_govt_budget = li_govt_budget
        

    def store_differences(self, ):
        

    def summary_comparison(self, ):

        # Accuracy, Precision, Recall, F1 score