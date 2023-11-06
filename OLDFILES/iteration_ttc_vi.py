#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
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
from pyro.control import controller
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
grid_sys = discretizer.GridDynamicSystem(sys, (81,81), (16,2), 0.1)
cf2 = costfunction.DriverModelCostFunction.from_sys(sys)
cf2.xbar = np.array( [0, 0] ) # target

sys.cost_function = cf2
#%%
vi2 = valueiteration.ValueIteration_ND( grid_sys , cf2 )
vi2.threshold = 2
vi2.uselookuptable = False
vi2.initialize()
vi2.load_data('VI_CONTROLLER_FOR_TEST')

#vi2.compute_steps(500,False)

#%%
vi2.assign_interpol_controller()
vi2.ctl.vi_law = vi2.vi_law

#%%

sys.driver = sys.drivers10['Good']
sys.road = sys.roads['AsphalteDry']
temp = BaselineController.TTCController(sys.n, sys.m, sys.p, sys.human_model, sys.f)
temp.ttc_min = 2

cl_sys = controller.ClosedLoopSystem( sys , temp )
cl_sys.cost_function = sys.cost_function
cl_sys2 = controller.ClosedLoopSystem( sys , vi2.ctl )
cl_sys2.cost_function = sys.cost_function
#%%
x0   = np.array([0,10])
cl_sys.x0 = x0
cl_sys2.x0 = x0
sim_s = s.SimulatorV2(cl_sys)
traj_s = sim_s.compute()
sim2_s = s.SimulatorV2(cl_sys2)
traj2_s = sim2_s.compute()
cf_custom1_s = cf_sim.CustomCostFunctionSimulationSim(traj_s, traj2_s, sys.cost_function)
cf_custom1_s.compute_cost_function()
cf_custom1_s.plot_multiple_g_add()

c = cost2go.cost2go(grid_sys, cl_sys, 'TTC')
c.compute_steps()

c2 = cost2go.cost2go(grid_sys, cl_sys2, 'VI')
c2.compute_steps()

c.plot_commands()
c2.plot_commands()

plt.imshow((c.J-c2.J).transpose(),origin='lower')
