#!/usr/bin/env python3
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
#%%
temps_debut = time.time()

### ROAD AND DRIVER SETUP
roads = ['CobblestoneWet']

### SYSTEM SETUP
sys = system.LongitudinalFrontWheelDriveCarWithDriverModel()
sys.mass = 20
sys.lenght = 0.48
sys.xc = 0.24
sys.yc = 0.15 
sys.mass = 20
sys.cdA = 0.3 * 0.105
sys.x_ub = np.array([0 , 4.5])
sys.x_lb = np.array([-10., 0])
sys.u_ub = np.array([0.0, 1])
sys.u_lb = np.array([-0.20, 0])
sys.obs_dist = sys.x_ub[0]

sys.road = sys.roads[roads[0]]
slip_data = sys.return_max_mu()
sys.dmax = sys.f([0,sys.x_ub[1]],[-slip_data[1],1])

### DRIVER SETUP
sys.tf = 1.75
sys.tm = 2.6
sys.tm_dot = 0.75

u_dim = [10,2]
x_dim = [300,150]
sys.x_grid = x_dim
dt = 0.02
sys.use_human_model = True

#COSTFUNCTION
cf = costfunction.DriverModelCostFunction.from_sys(sys)
cf.confort_coef = 100
cf.override_coef = 10
cf.security_coef = 500
cf.xbar = np.array( [(sys.x_ub[0]-1), 0] ) # target
sys.cost_function = cf

def calc_ttc(state):
    px = 0-state[0]
    vx = 0-state[1]
    ax = 0-state[2] +0.0000001
    axmax = 0-state[3] +0.0000001
    ax = axmax
    if ((vx**2)-(2*px*axmax)) <= 0.0:
        ttc = 1000
    else:
         ttc_minus = -(vx/ax) - (np.sqrt((vx**2)-(2*px*ax))/ax)
         ttc_plus = -(vx/ax) + (np.sqrt((vx**2)-(2*px*ax))/ax)
         if ttc_minus <= ttc_plus:
              ttc = ttc_minus
         else:
              ttc = ttc_plus
    ttc = np.clip(ttc,0,10)
    return ttc

def calc_ttc_treshhold(state):
    px = 0-state[0]
    vx = 0-state[1]
    ax = 0-state[2]
    ax = 0.0000001
    axmax = 0-state[3]
    if (vx**2) - (ax*px*axmax) < 0:
        ttc_treshhold = 0
    else:
        ttc_treshhold = (-(vx)/axmax) + (np.sqrt((vx**2) - (ax*px*axmax))/axmax)
    return ttc_treshhold

def calc_msd(state):
    px = 0-state[0]
    vx = 0-state[1]
    ax = 0-state[2] +0.0000001
    axmax = 0-state[3] +0.0000001
    if ((vx**2)-(2*px*axmax)) <= 0.0:
        ttc = 1000
    else:
         ttc_minus = -(vx/ax) - (np.sqrt((vx**2)-(2*px*ax))/ax)
         ttc_plus = -(vx/ax) + (np.sqrt((vx**2)-(2*px*ax))/ax)
         if ttc_minus <= ttc_plus:
              ttc = ttc_minus
         else:
              ttc = ttc_plus
    ttc = np.clip(ttc,0,10)
    return ttc

def calc_msd_treshhold(state):
    px = 0-state[0]
    vx = 0-state[1]
    ax = 0-state[2]
    axmax = 0-state[3]
    
    Sa = (vx**2)/(2*axmax)
    if Sa > px:
        return 1
    else:
        return 0 

sys.plot_human_model_ttc()
#%%
E = [[+0.0, 1.0]]
grid_sys = discretizer2.GridDynamicSystem(sys, sys.x_grid, u_dim, dt, esperance = E)

pos = grid_sys.x_level[0]
vit = grid_sys.x_level[1]

res_bool = np.zeros([len(pos), len(vit)])
res_ttc = np.zeros([len(pos), len(vit)])
msd_bool = np.zeros([len(pos), len(vit)])
res_treshhold = np.zeros([len(pos), len(vit)])

for i_p in range(len(pos)):
    for i_v in range(len(vit)):
        state = [pos[i_p], vit[i_v], sys.f([pos[i_p],vit[i_v]],[sys.human_model_time_margin(pos[i_p],vit[i_v]),1])[1], sys.dmax[1]]
        res_ttc[i_p][i_v] = calc_ttc(state)
        res_treshhold[i_p][i_v] = calc_ttc_treshhold(state)
        msd_bool[i_p][i_v] = calc_msd_treshhold(state)
        if res_ttc[i_p][i_v] <= res_treshhold[i_p][i_v]:
            res_bool[i_p][i_v] = 1
        else:
            res_bool[i_p][i_v] = 0

#%%
            
fig, axs = plt.subplots(2, 1)
plt.ion()
xname = sys.state_label[0] + ' ' + sys.state_units[0]
yname = sys.state_label[1] + ' ' + sys.state_units[1]           

axs[0].set_title('TTC')
i = axs[0].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], res_bool.T, shading='gouraud')
axs[0].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
fig.colorbar(i, ax=axs[0])

axs[1].set_title('MSD')
i = axs[1].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], msd_bool.T, shading='gouraud')
axs[1].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
fig.colorbar(i, ax=axs[1])

#%%    
fig, axs = plt.subplots(1, 4)
plt.ion()
fig.suptitle('TTC')
xname = sys.state_label[0] + ' ' + sys.state_units[0]
yname = sys.state_label[1] + ' ' + sys.state_units[1] 
 
axs[0].set_title('TTC')
i = axs[0].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], res_ttc.T, shading='gouraud')
axs[0].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
fig.colorbar(i, ax=axs[0])

axs[1].set_title('Threshold')
i = axs[1].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], res_treshhold.T, shading='gouraud')
axs[1].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
fig.colorbar(i, ax=axs[1])

axs[2].set_title('Baseline')
i = axs[2].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], res_bool.T, shading='gouraud')
axs[2].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])

axs[3].set_title('Human Model')
i = axs[3].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], sys.plot_human_model_ttc(plot=False).T, shading='gouraud')
axs[3].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
