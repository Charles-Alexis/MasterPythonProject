# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:05:22 2022

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
import BaselineController

import simulationv2 as s
import CustomCostFunctionSimulation as cf_sim
from pyro.control  import controller

import cost2go
###############################################################################

###############################################################################
#%%
temps_debut = time.time()

sys  = system.LongitudinalFrontWheelDriveCarWithDriverModel()
sys.driver = sys.drivers10['Good']
sys.road = sys.roads['AsphalteWet']
sys.obs_dist = 100 + sys.lenght/2

#%%
grid_sys = discretizer.GridDynamicSystem(sys, (281,181), (2,2), 0.1)
cf2 = costfunction.DriverModelCostFunction.from_sys(sys)
cf2.xbar = np.array( [0, 0] ) # target

sys.cost_function = cf2

#%%

temp = BaselineController.TTCController(sys.n, sys.m, sys.p, sys.human_model, sys.f)
temp.ttc_min = 5

cl_sys = controller.ClosedLoopSystem( sys , temp )
cl_sys.cost_function = sys.cost_function

cl_sys.x0 = np.array([50,10])
traj = cl_sys.compute_trajectory(20, 10001, 'euler')
#cl_sys.plot_trajectory('xuj')

#%%
c = cost2go.cost2go(grid_sys, cl_sys)
c.compute_steps()