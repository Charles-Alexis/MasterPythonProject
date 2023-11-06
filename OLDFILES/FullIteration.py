# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:08:05 2022

@author: Charles-Alexis
"""

###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import time
###############################################################################
import valueiteration
import discretizer
import system
import costfunction
from pyro.control  import controller
###############################################################################

###############################################################################
#%%

temps_debut = time.time()
sys  = system.LongitudinalFrontWheelDriveCarWithDriverModel()


for r in sys.roads:
    for d in sys.drivers10:
        sys.driver = sys.drivers10[d]
        sys.road = sys.roads[r]
        print(r+'_'+d)
        
        grid_sys = discretizer.GridDynamicSystem(sys, (101,101), (21,2), 0.1)
        cf2 = costfunction.DriverModelCostFunction.from_sys(sys)
        cf2.xbar = np.array( [0, 0] ) # target
        
        vi = valueiteration.ValueIteration_ND( grid_sys , cf2 )
        vi.threshold = 2
        vi.uselookuptable = False
        vi.initialize()
        vi.compute_steps(500,False)
        vi.save_data('Models_analytic_10_2/' + r + '_' + d)
        print(r+'_'+d)

temps_fin = time.time()
print('Temps total: {}'.format(temps_fin-temps_debut))


