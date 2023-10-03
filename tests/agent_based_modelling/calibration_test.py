import unittest
import numpy as np
import pandas as pd
from agent_based_modelling.calibration import get_b2i_network, get_i2i_network, calibrate
import itertools

class TestCalibration(unittest.TestCase):
    
    def setUp(self):
        self.model_size = '7bn'
        self.budget_item_count = 25
        self.indic_count = 100
        self.start_year = 2013
        self.end_year = 2017
        self.T = int(5*(self.end_year - self.start_year))

        self.threshold = 0.1
        self.low_precision_counts = 5
        self.increment = 10
        self.parallel_processes = 1
        self.verbose = True
        self.time_experiments = True

        
        # Dummy data
        self.indic_start = np.random.rand(self.indic_count)/5 # shape: (indic_count)
        self.indic_final = self.indic_start + np.random.rand(self.indic_count)/5

        self.success_rates = np.random.rand(self.indic_count) # shape: (indic_count)
        self.R = np.ones(self.indic_count ) # shape: (indic_count)
        self.qm = np.random.rand(self.indic_count) # shape: (indic_count)
        self.rl = np.random.rand(self.indic_count ) # shape: (indic_count)
        self.Bs = np.random.rand( self.budget_item_count, self.T ) # shape: (budget_item_count, years)
        
        
        self.i2i_network = np.random.rand(self.indic_count, self.indic_count)
        # Create an neverending iterable of integers between 0 and self.budget_item_count
        bi_idx_iter = itertools.cycle( list( range(self.budget_item_count) ))
        # B_dict creates a mapping from indicators (keys) to a list of budget items (values)
        # all budget items must be included in the mapping
        self.B_dict = { k: sorted( list(itertools.islice(bi_idx_iter, 2)))  
            for k in range(self.indic_count) }
        

    def test_get_b2i_network(self):
        for b2i_method in ['ea', 'verbalize', 'CPUQ_binomial']:
            with self.subTest(b2i_method=b2i_method):
                B_dict = get_b2i_network(b2i_method, self.model_size)
                self.assertIsInstance(B_dict, dict)
                self.assertEqual(len(B_dict), 415)
                        
    def test_get_i2i_network(self):
        for i2i_method in [ 'CPUQ_multinomial', 'ccdr', 'zero', 'verbalize', 'entropy']:
            with self.subTest(i2i_method=i2i_method):
                i2i_network = get_i2i_network(i2i_method, self.indic_count, self.model_size)
                self.assertIsInstance(i2i_network, np.ndarray)
                self.assertEqual(i2i_network.shape, (self.indic_count, self.indic_count))
                if i2i_method in ['verbalize', 'entropy', 'CPUQ_multinomial', 'ccdr' ]:
                    self.assertTrue(np.any(i2i_network != 0))
                    
    def test_calibrate(self):
        for b2i_method, i2i_method, parrallel_processes, time_experiments in [
            # ('ea', 'zero', None, False),
            #  ('ea','verbalize', 2, True),
        #  ('ea', 'entropy', None, False),
        #   ('ea','CPUQ_multinomial', None, False),
         ('verbalize','verbalize', 4, False), 
        #  ('CPUQ_binomial','CPUQ_multinomial', None, False),
        #  ('ea','ccdr', None, False)
         ]:
            
            with self.subTest(b2i_method=b2i_method, i2i_method=i2i_method, 
                parrallel_processes=parrallel_processes, time_experiments=time_experiments):
                                
                df_output = calibrate(
                    indic_start = self.indic_start,
                    indic_final = self.indic_final,

                    success_rates=self.success_rates,
                    R=self.R,
                    qm=self.qm,
                    rl=self.rl,
                    Bs=self.Bs,
                    B_dict=self.B_dict,
                    T=self.T,
                    i2i_network=self.i2i_network,

                    parallel_processes=parrallel_processes,
                    threshold = self.threshold,
                    low_precision_counts=self.low_precision_counts,
                    increment=self.increment,
                    verbose=self.verbose,
                    time_experiments=self.time_experiments,

                    mc_simulations=1
                )

                self.assertIsInstance(df_output, dict)
                
                self.assertIn('parameters', df_output)
                self.assertEqual( df_output['parameters'].shape[1], 8)
                
                if time_experiments:
                    self.assertIn('time_elapsed', df_output)
                    self.assertIsInstance(df_output['time_elapsed'], float)


                    self.assertIn('iterations', df_output)
                    self.assertIsInstance(df_output['iterations'], int)
    
        
if __name__ == '__main__':
    unittest.main()
