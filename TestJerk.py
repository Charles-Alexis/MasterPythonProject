# -*- coding: utf-8 -*-

###############################################################################
import numpy as np
import time
import matplotlib.pyplot as plt
###############################################################################
import discretizer2
import system 
import costfunction
import BaselineController 
import controller
import simulationv2 as s
import dynamic_programming as dprog
import cost2go2
###############################################################################
from datetime import date
import os
###############################################################################

sys = system.LongitudinalFrontWheelDriveCarWithDriverModel_withjerk()
sys.dmax = sys.return_max_mu()

### DRIVER MODEL CONFIGURATION
sys.tm_dot = -0.4
sys.tf = 1.75
sys.tm_coef = 0.8

slip_data = sys.return_max_mu()
sys.dmax = sys.f([0,sys.x_ub[1],0],[slip_data[1],1])
print(sys.dmax)

dt = 0.2
x_dim = [200,200,20]
u_dim = [20,2]
grid_sys = discretizer2.GridDynamicSystem(sys, x_dim, u_dim, dt, esperance = sys.driver[0], print_data=True, lookup = True)
#%%
### COST FUNCTION CONFIGURATION
cf = costfunction.DriverModelCostFunction_jerk.from_sys(sys) 

cf.confort_coef = 1
cf.override_coef = 10
cf.security_coef = 1
cf.security_slope = 10
cf.security_distance = 3
sys.cost_function = cf

dp = dprog.DynamicProgrammingWithLookUpTable(grid_sys, cf, esperance = sys.driver[0])
dp.compute_steps(5000,  treshhold=0.001, animate_policy=False, animate_cost2go=False, jmax = 1000)