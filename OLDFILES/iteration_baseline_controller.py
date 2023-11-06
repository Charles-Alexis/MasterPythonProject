#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 21:49:27 2022

@author: Charles-Alexis
a"""


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
from pyro.control  import controller
###############################################################################

###############################################################################
#%%
temps_debut = time.time()

sys  = system.LongitudinalFrontWheelDriveCarWithDriverModel()
sys.driver = sys.drivers10['Good']
sys.road = sys.roads['AsphalteDry']
sys.obs_dist = 100 + sys.lenght/2

#%%
grid_sys = discretizer.GridDynamicSystem(sys, (81,81), (16,2), 0.1)
cf2 = costfunction.DriverModelCostFunction.from_sys(sys)
cf2.xbar = np.array( [0, 0] ) # target

sys.cost_function = cf2


#%%

temp = BaselineController.TTCController(sys.n, sys.m, sys.p, sys.human_model, sys.f)


cl_sys2 = controller.ClosedLoopSystem( sys , temp)
cl_sys2.cost_function = sys.cost_function

###############################################################################
## Simulation and animation
x0   = np.array([0,10])
tf   = 30

cl_sys2.x0 = x0
traj = cl_sys2.compute_trajectory(tf, 10001, 'euler')
cl_sys2.plot_trajectory('xuj')
cl_sys2.animate_simulation( time_factor_video = 3 )


plt.figure()
ax1 = plt.subplot(411)
ax1.title.set_text('Slip')
plt.plot(traj.t, traj.u[:,0])
ax2 = plt.subplot(412)
ax2.title.set_text('Acceleration')
plt.plot(traj.t, traj.dx[:,1])
ax3 = plt.subplot(413)
ax3.title.set_text('Position')
plt.plot(traj.t, traj.x[:,0])
ax3 = plt.subplot(414)
ax3.title.set_text('Vitesse')
plt.plot(traj.t, traj.x[:,1])
plt.show()


#%%
temps_fin = time.time()
print('Temps total: {}'.format(temps_fin-temps_debut))


