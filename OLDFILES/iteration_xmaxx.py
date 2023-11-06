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

sys = system.LongitudinalFrontWheelDriveCarWithDriverModel()
sys.lenght = 0.6
sys.xc = 0.3
sys.yc = 0.175
sys.mass = 25
sys.cdA = 0.3 * 0.105
sys.x_ub = np.array([+20.0, 6.0])
sys.x_lb = np.array([0, 0])
sys.u_ub = np.array([0.0, 1])
sys.u_lb = np.array([-0.3, 0])
#GRIDSYS
grid_sys = discretizer.GridDynamicSystem(sys, (101, 101), (4, 2), 0.1)

sys.driver = sys.driver_xmaxx_fort['Good']
sys.road = sys.roads['Ice']
sys.obs_dist = sys.x_ub[0]

#%%
cf2 = costfunction.DriverModelCostFunction.from_sys(sys)
cf2.confort_coef = 1
cf2.override_coef = 1
cf2.security_coef = 1
cf2.xbar = np.array( [0, 0] ) # target
sys.cost_function = cf2
#%%
vi2 = valueiteration.ValueIteration_ND( grid_sys , cf2 )
vi2.threshold = 0.3
vi2.uselookuptable = False
vi2.initialize()
#vi2.load_data('test_19octobre')
vi2.compute_steps(500,True)

#%%
vi2.assign_interpol_controller()
vi2.ctl.vi_law = vi2.vi_law
vi2.plot_policy()
#%%
sys.driver = sys.driver_xmaxx_fort['Good']
#sys.road = sys.roads['AsphalteDry']
temp = BaselineController.TTCController(sys.n, sys.m, sys.p, sys.human_model, sys.f, ttc_ref= 5.8, position_obs=19.5, slip_cmd=-0.116)

cl_sys = controller.ClosedLoopSystem( sys , temp )
cl_sys.cost_function = sys.cost_function
cl_sys2 = controller.ClosedLoopSystem( sys , vi2.ctl )
cl_sys2.cost_function = sys.cost_function
x0   = np.array([0,5])
cl_sys.x0 = x0
cl_sys2.x0 = x0

sim_test = s.SimulatorV2_vi_vs_ttc(cl_sys2, cl_sys,  x0_end = 20, x1_end = 0.03)
sim_test.plot_trajectories(name = sys.road[-1] + '_' + sys.driver[-1])


#%%
#cf_custom1_s = cf_sim.CustomCostFunctionSimulationSim(traj_s, traj2_s, sys.cost_function)
#cf_custom1_s.compute_cost_function()
#cf_custom1_s.plot_multiple_g_add()

#c = cost2go.cost2go(grid_sys, cl_sys, 'TTC')
#c.compute_steps()


c2 = cost2go.cost2go(grid_sys, cl_sys2, 'VI')
c2.compute_steps()

#c.plot_commands()
#c2.plot_commands()

# fig, axs = plt.subplots(6, 1)
# axs[0].set_title('X')
# axs[0].plot(traj_s.t, traj_s.x[:,0])
# axs[1].set_title('Xp')
# axs[1].plot(traj_s.t, traj_s.x[:,1])
# axs[2].set_title('Xpp')
# axs[2].plot(traj_s.t, traj_s.dx[:,1])
# axs[3].set_title('Slip')
# axs[3].plot(traj_s.t, traj_s.u[:,0])
# axs[4].set_title('Override')
# axs[4].plot(traj_s.t, traj_s.u[:,1])
# axs[5].set_title('J')
# axs[5].plot(traj_s.t, traj_s.J)

