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

time_debut = time.time()
roads = 'AsphalteWet'
drivers = 'Null'
comparaison = 'NoDiff'

plot_full_diff = False
plot_full_nodiff = False
compute_c2g = False

x_dim = [500,500]
u_dim = [30,2]
x_lb = [-100., 0]
controler = ['Vi','Ttc','Msd', 'Human']

print(roads+drivers+comparaison)
### SYSTEM CONFIGURATION
sys = system.LongitudinalFrontWheelDriveCarWithDriverModel()

sys.mass = 760
sys.lenght = 3.35
sys.xc = sys.lenght/2
sys.yc = 1.74/2 
sys.cdA = 0.3 * (1.84*1.74)
sys.x_ub = np.array([0 , 20.0])
sys.x_lb = np.array(x_lb)
sys.u_ub = np.array([0.0, 1])
sys.u_lb = np.array([-0.3, 0])

sys.u_dim = u_dim
sys.m = len(sys.u_dim)
sys.obs_dist = sys.x_ub[0]
sys.x_grid = x_dim
sys.road = sys.roads[roads]
sys.timing = 0.5

sys.driver = sys.drivers[drivers]

sys.mu_coef = -sys.return_max_mu()[0]

slip_data = sys.return_max_mu()
sys.dmax = sys.f([0,sys.x_ub[1]],[slip_data[1],1])
sys.best_slip = sys.return_max_mu()[1]

### DRIVER MODEL CONFIGURATION
sys.tm_dot = -0.4
sys.tf = 1.75
sys.tm_coef = 0.8

### COST FUNCTION CONFIGURATION
cf = costfunction.DriverModelCostFunction.from_sys(sys) 
cf.confort_coef = 0.01
cf.override_coef = 100
cf.security_coef = 10

cf.security_slope = 10
cf.security_distance = 3
sys.cost_function = cf
cf_list = list((sys.cost_function.g, sys.cost_function.g_confort, sys.cost_function.g_security, sys.cost_function.g_override))

### GRID SYSTEM CONFIGURATION
dt = 0.1
controlers_dim = x_dim
       

grid_sys_vi = discretizer2.GridDynamicSystem(sys, controlers_dim, u_dim, dt, esperance = sys.driver[0], print_data=False, lookup = True)
grid_sys_controller = discretizer2.GridDynamicSystem(sys, controlers_dim, u_dim, dt, esperance = sys.driver[0], print_data=False, lookup = False)
print('It took: '+str(time.time() - time_debut)+ ' to compute Grid')
#%%
### VALUE ITERATION
time_debut = time.time()
dp = dprog.DynamicProgrammingWithLookUpTable(grid_sys_vi, cf, esperance = sys.driver[0])
dp.compute_steps(5000,  treshhold=0.0001, animate_policy=False, animate_cost2go=False, jmax = 1000)
vi_controller = dprog.LookUpTableController(grid_sys_vi , dp.pi )
vi_controller.k = 2
cl_sys_vi = controller.ClosedLoopSystem( sys , vi_controller ) 
cl_sys_vi.cost_function = sys.cost_function 
vi_cmap = grid_sys_vi.get_grid_from_array(grid_sys_vi.get_input_from_policy(vi_controller.pi, 0)).T
#
#    
### TIME TO COLLISION
#ttc_controller = BaselineController.TTCController(sys, grid_sys_controller)
#cl_sys_ttc = controller.ClosedLoopSystem( sys , ttc_controller )  
#cl_sys_ttc.cost_function = sys.cost_function
#ttc_cmap = ttc_controller.c_array().T
#         
### MINIMAL STOPPING DISTANCE    
#msd_controller = BaselineController.MSDController(sys, grid_sys_controller)
#cl_sys_msd = controller.ClosedLoopSystem( sys , msd_controller )  
#cl_sys_msd.cost_function = sys.cost_function
#msd_cmap = msd_controller.c_array().T
#                
human_controller = BaselineController.humanController(sys, grid_sys_controller)
cl_sys_human = controller.ClosedLoopSystem( sys , human_controller ) 
cl_sys_human.cost_function = sys.cost_function
human_cmap = human_controller.c_array().T

print('It took: '+str(time.time() - time_debut)+ ' to compute controlers')
#%% Cost 2 Go
time_debut = time.time()
if compute_c2g:
    c2g_vi = cost2go2.cost2go_list_2(grid_sys_vi, sys, cf, vi_controller.c)
    c2g_ttc = cost2go2.cost2go_list_2(grid_sys_vi, sys, cf, ttc_controller.c)
    c2g_msd = cost2go2.cost2go_list_2(grid_sys_vi, sys, cf, msd_controller.c)
    c2g_human = cost2go2.cost2go_list_2(grid_sys_vi, sys, cf, human_controller.c)

print('It took: '+str(time.time() - time_debut)+ ' to compute cost 2 go')

#%% Simulation
# time_debut = time.time()

# sim_list_vi = list()
# sim_list_ttc = list()
# sim_list_msd = list()
# sim_list_human = list()

# x0 = np.array([-100.0,10.0])
# x1 = np.array([-100.0,15.0])
# x2 = np.array([-100.0,20.0])
# x3 = np.array([-80.0,15.0])
# x4 = np.array([-60.0,15.0])
# x5 = np.array([-20.0,15.0])

# x0_test = [x0,x1,x2,x3,x4,x5,x2]

# for x0_to_test in x0_test:
#      cl_sys_vi.x0 = x0_to_test
#      sim_vi = s.SimulatorV2(cl_sys_vi)
#      sim_list_vi.append(sim_vi.traj_to_args(sim_vi.traj))
         
#      cl_sys_ttc.x0 = x0_to_test
#      sim_ttc = s.SimulatorV2(cl_sys_ttc)
#      sim_list_ttc.append(sim_ttc.traj_to_args(sim_ttc.traj))
 
#      cl_sys_msd.x0 = x0_to_test
#      sim_msd = s.SimulatorV2(cl_sys_msd)
#      sim_list_msd.append(sim_msd.traj_to_args(sim_msd.traj))
         
#      cl_sys_human.x0 = x0_to_test
#      sim_human = s.SimulatorV2(cl_sys_human)
#      sim_list_human.append(sim_human.traj_to_args(sim_human.traj))
     
# print('It took: '+str(time.time() - time_debut)+ ' to compute simulations')

#%% Controler law map
fig, axs = plt.subplots(1)
plt.ion()
fig.suptitle('VI')
xname = sys.state_label[0] + ' ' + sys.state_units[0]
yname = sys.state_label[1] + ' ' + sys.state_units[1] 
i = axs.pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], vi_cmap, shading='gouraud', cmap = 'plasma')
axs.axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
fig.colorbar(i, ax=axs)

#fig, axs = plt.subplots(1)
#plt.ion()
#fig.suptitle('TTC')
#xname = sys.state_label[0] + ' ' + sys.state_units[0]
#yname = sys.state_label[1] + ' ' + sys.state_units[1] 
#i = axs.pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttc_cmap, shading='gouraud', cmap = 'plasma')
#axs.axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs)
#
#fig, axs = plt.subplots(1)
#plt.ion()
#fig.suptitle('MSD')
#xname = sys.state_label[0] + ' ' + sys.state_units[0]
#yname = sys.state_label[1] + ' ' + sys.state_units[1] 
#i = axs.pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], msd_cmap, shading='gouraud', cmap = 'plasma')
#axs.axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs)

# fig, axs = plt.subplots(1)
# plt.ion()
# fig.suptitle('Carte des commandes en ratio de glissement du modèle humain')
# xname = sys.state_label[0] + ' ' + sys.state_units[0]
# yname = sys.state_label[1] + ' ' + sys.state_units[1] 
# i = axs.pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], human_cmap, shading='gouraud', cmap = 'plasma')
# axs.axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs)

#%% Cost to go
# fig, axs = plt.subplots(1)
# plt.ion()
# fig.suptitle('VI')
# xname = sys.state_label[0] + ' ' + sys.state_units[0]
# yname = sys.state_label[1] + ' ' + sys.state_units[1] 
# i = axs.pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_vi.J_total.T, shading='gouraud', cmap = 'plasma')
# axs.axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs)

# fig, axs = plt.subplots(1)
# plt.ion()
# fig.suptitle('TTC')
# xname = sys.state_label[0] + ' ' + sys.state_units[0]
# yname = sys.state_label[1] + ' ' + sys.state_units[1] 
# i = axs.pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_ttc.J_total.T, shading='gouraud', cmap = 'plasma')
# axs.axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs)

# fig, axs = plt.subplots(1)
# plt.ion()
# fig.suptitle('MSD')
# xname = sys.state_label[0] + ' ' + sys.state_units[0]
# yname = sys.state_label[1] + ' ' + sys.state_units[1] 
# i = axs.pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_msd.J_total.T, shading='gouraud', cmap = 'plasma')
# axs.axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs)

# fig, axs = plt.subplots(1)
# plt.ion()
# fig.suptitle('HUMAN')
# xname = sys.state_label[0] + ' ' + sys.state_units[0]
# yname = sys.state_label[1] + ' ' + sys.state_units[1] 
# i = axs.pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_human.J_total.T, shading='gouraud', cmap = 'plasma')
# axs.axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs)

#%% Cost to go VI
#
#c2g_vi = cost2go2.cost2go_list_2(grid_sys_vi, sys, cf, vi_controller.c)
#fig, axs = plt.subplots(1,5)
#plt.ion()
#fig.suptitle('VI')
#xname = sys.state_label[0] + ' ' + sys.state_units[0]
#yname = sys.state_label[1] + ' ' + sys.state_units[1] 
#i = axs[0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], vi_cmap, shading='gouraud', cmap = 'plasma')
#axs[0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs[0])
#i = axs[1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_vi.J_total.T, shading='gouraud', cmap = 'plasma')
#axs[1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs[1])
#i = axs[2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1],c2g_vi.J_confort.T, shading='gouraud', cmap = 'plasma')
#axs[2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs[2])
#i = axs[3].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_vi.J_security.T, shading='gouraud', cmap = 'plasma')
#axs[3].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs[3])
#i = axs[4].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_vi.J_override.T, shading='gouraud', cmap = 'plasma')
#axs[4].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs[4])
#axs[0].set_title('Commandes')
#axs[1].set_title('Cost to Go Totale')
#axs[2].set_title('Cost to Go Confort: '+ str(cf.confort_coef))
#axs[3].set_title('Cost to Go Sécurité: '+ str(cf.security_coef))
#axs[4].set_title('Cost to Go Liberté: '+ str(cf.override_coef))
#axs[0].set_ylabel('VI')
#    
#
#fig, axs = plt.subplots(1)
#plt.ion()
#fig.suptitle('Carte des commandes en ratio de glissement du modèle humain')
#xname = sys.state_label[0] + ' ' + sys.state_units[0]
#yname = sys.state_label[1] + ' ' + sys.state_units[1] 
#i = axs.pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], human_cmap, shading='gouraud', cmap = 'plasma')
#axs.axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs)



#%% Cost to go
# fig, axs = plt.subplots(1,5)
# plt.ion()
# fig.suptitle('VI')
# xname = sys.state_label[0] + ' ' + sys.state_units[0]
# yname = sys.state_label[1] + ' ' + sys.state_units[1] 
# i = axs[0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], vi_cmap, shading='gouraud', cmap = 'plasma')
# axs[0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[0])
# i = axs[1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_vi.J_total.T, shading='gouraud', cmap = 'plasma')
# axs[1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[1])
# i = axs[2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1],c2g_vi.J_confort.T, shading='gouraud', cmap = 'plasma')
# axs[2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[2])
# i = axs[3].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_vi.J_security.T, shading='gouraud', cmap = 'plasma')
# axs[3].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[3])
# i = axs[4].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_vi.J_override.T, shading='gouraud', cmap = 'plasma')
# axs[4].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[4])
# axs[0].set_title('Commandes')
# axs[1].set_title('Cost to Go Totale')
# axs[2].set_title('Cost to Go Confort: '+ str(cf.confort_coef))
# axs[3].set_title('Cost to Go Sécurité: '+ str(cf.security_coef))
# axs[4].set_title('Cost to Go Liberté: '+ str(cf.override_coef))

# fig, axs = plt.subplots(1,5)
# plt.ion()
# fig.suptitle('ttc')
# xname = sys.state_label[0] + ' ' + sys.state_units[0]
# yname = sys.state_label[1] + ' ' + sys.state_units[1] 
# i = axs[0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttc_cmap, shading='gouraud', cmap = 'plasma')
# axs[0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[0])
# i = axs[1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_ttc.J_total.T, shading='gouraud', cmap = 'plasma')
# axs[1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[1])
# i = axs[2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1],c2g_ttc.J_confort.T, shading='gouraud', cmap = 'plasma')
# axs[2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[2])
# i = axs[3].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_ttc.J_security.T, shading='gouraud', cmap = 'plasma')
# axs[3].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[3])
# i = axs[4].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_ttc.J_override.T, shading='gouraud', cmap = 'plasma')
# axs[4].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[4])
# axs[0].set_title('Commandes')
# axs[1].set_title('Cost to Go Totale')
# axs[2].set_title('Cost to Go Confort: '+ str(cf.confort_coef))
# axs[3].set_title('Cost to Go Sécurité: '+ str(cf.security_coef))
# axs[4].set_title('Cost to Go Liberté: '+ str(cf.override_coef))

# fig, axs = plt.subplots(1,5)
# plt.ion()
# fig.suptitle('msd')
# xname = sys.state_label[0] + ' ' + sys.state_units[0]
# yname = sys.state_label[1] + ' ' + sys.state_units[1] 
# i = axs[0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], msd_cmap, shading='gouraud', cmap = 'plasma')
# axs[0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[0])
# i = axs[1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_msd.J_total.T, shading='gouraud', cmap = 'plasma')
# axs[1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[1])
# i = axs[2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1],c2g_msd.J_confort.T, shading='gouraud', cmap = 'plasma')
# axs[2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[2])
# i = axs[3].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_msd.J_security.T, shading='gouraud', cmap = 'plasma')
# axs[3].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[3])
# i = axs[4].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_msd.J_override.T, shading='gouraud', cmap = 'plasma')
# axs[4].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[4])
# axs[0].set_title('Commandes')
# axs[1].set_title('Cost to Go Totale')
# axs[2].set_title('Cost to Go Confort: '+ str(cf.confort_coef))
# axs[3].set_title('Cost to Go Sécurité: '+ str(cf.security_coef))
# axs[4].set_title('Cost to Go Liberté: '+ str(cf.override_coef))

# fig, axs = plt.subplots(1,5)
# plt.ion()
# fig.suptitle('human')
# xname = sys.state_label[0] + ' ' + sys.state_units[0]
# yname = sys.state_label[1] + ' ' + sys.state_units[1] 
# i = axs[0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], human_cmap, shading='gouraud', cmap = 'plasma')
# axs[0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[0])
# i = axs[1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_human.J_total.T, shading='gouraud', cmap = 'plasma')
# axs[1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[1])
# i = axs[2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1],c2g_human.J_confort.T, shading='gouraud', cmap = 'plasma')
# axs[2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[2])
# i = axs[3].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_human.J_security.T, shading='gouraud', cmap = 'plasma')
# axs[3].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[3])
# i = axs[4].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_human.J_override.T, shading='gouraud', cmap = 'plasma')
# axs[4].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
# fig.colorbar(i, ax=axs[4])
# axs[0].set_title('Commandes')
# axs[1].set_title('Cost to Go Totale')
# axs[2].set_title('Cost to Go Confort: '+ str(cf.confort_coef))
# axs[3].set_title('Cost to Go Sécurité: '+ str(cf.security_coef))
# axs[4].set_title('Cost to Go Liberté: '+ str(cf.override_coef))

#%% FULL PLOTTING DIFF
if plot_full_diff:
    fig, axs = plt.subplots(4,5)
    plt.ion()
    fig.suptitle(roads+drivers+comparaison)
    xname = sys.state_label[0] + ' ' + sys.state_units[0]
    yname = sys.state_label[1] + ' ' + sys.state_units[1] 
    i = axs[0][0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], vi_cmap, shading='gouraud', cmap = 'plasma')
    axs[0][0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[0][0])
    i = axs[0][1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_vi.J_total.T, shading='gouraud', cmap = 'plasma')
    axs[0][1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[0][1])
    i = axs[0][2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1],c2g_vi.J_confort.T, shading='gouraud', cmap = 'plasma')
    axs[0][2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[0][2])
    i = axs[0][3].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_vi.J_security.T, shading='gouraud', cmap = 'plasma')
    axs[0][3].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[0][3])
    i = axs[0][4].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_vi.J_override.T, shading='gouraud', cmap = 'plasma')
    axs[0][4].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[0][4])
    axs[0][0].set_title('Commandes')
    axs[0][1].set_title('Cost to Go Totale')
    axs[0][2].set_title('Cost to Go Confort: '+ str(cf.confort_coef))
    axs[0][3].set_title('Cost to Go Sécurité: '+ str(cf.security_coef))
    axs[0][4].set_title('Cost to Go Liberté: '+ str(cf.override_coef))
    axs[0][0].set_ylabel('VI')
    
    i = axs[1][0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttc_cmap, shading='gouraud', cmap = 'plasma')
    axs[1][0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[1][0])
    i = axs[1][1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_ttc.J_total.T-c2g_vi.J_total.T, shading='gouraud', cmap = 'plasma')
    axs[1][1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[1][1])
    i = axs[1][2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1],c2g_ttc.J_confort.T-c2g_vi.J_confort.T, shading='gouraud', cmap = 'plasma')
    axs[1][2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[1][2])
    i = axs[1][3].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_ttc.J_security.T-c2g_vi.J_security.T, shading='gouraud', cmap = 'plasma')
    axs[1][3].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[1][3])
    i = axs[1][4].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_ttc.J_override.T-c2g_vi.J_override.T, shading='gouraud', cmap = 'plasma')
    axs[1][4].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[1][4])
    axs[1][0].set_ylabel('TTC')
    
    i = axs[2][0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], msd_cmap, shading='gouraud', cmap = 'plasma')
    axs[2][0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[2][0])
    i = axs[2][1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_msd.J_total.T-c2g_vi.J_total.T, shading='gouraud', cmap = 'plasma')
    axs[2][1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[2][1])
    i = axs[2][2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1],c2g_msd.J_confort.T-c2g_vi.J_confort.T, shading='gouraud', cmap = 'plasma')
    axs[2][2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[2][2])
    i = axs[2][3].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_msd.J_security.T-c2g_vi.J_security.T, shading='gouraud', cmap = 'plasma')
    axs[2][3].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[2][3])
    i = axs[2][4].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_msd.J_override.T-c2g_vi.J_override.T, shading='gouraud', cmap = 'plasma')
    axs[2][4].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[2][4])
    axs[2][0].set_ylabel('MSD')
    
     
    i = axs[3][0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], human_cmap, shading='gouraud', cmap = 'plasma')
    axs[3][0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[3][0])
    i = axs[3][1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_human.J_total.T-c2g_vi.J_total.T, shading='gouraud', cmap = 'plasma')
    axs[3][1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[3][1])
    i = axs[3][2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1],c2g_human.J_confort.T-c2g_vi.J_confort.T, shading='gouraud', cmap = 'plasma')
    axs[3][2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[3][2])
    i = axs[3][3].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_human.J_security.T-c2g_vi.J_security.T, shading='gouraud', cmap = 'plasma')
    axs[3][3].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[3][3])
    i = axs[3][4].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_human.J_override.T-c2g_vi.J_override.T, shading='gouraud', cmap = 'plasma')
    axs[3][4].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[3][4])
    axs[3][0].set_ylabel('HUMAN')

#%% NO DIFF PLOTTING

if plot_full_nodiff:
    fig, axs = plt.subplots(4,5)
    plt.ion()
    fig.suptitle(roads+drivers+comparaison)
    
    xname = sys.state_label[0] + ' ' + sys.state_units[0]
    yname = sys.state_label[1] + ' ' + sys.state_units[1] 
    i = axs[0][0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], vi_cmap, shading='gouraud', cmap = 'plasma')
    axs[0][0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[0][0])
    i = axs[0][1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_vi.J_total.T, shading='gouraud', cmap = 'plasma')
    axs[0][1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[0][1])
    i = axs[0][2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1],c2g_vi.J_confort.T, shading='gouraud', cmap = 'plasma')
    axs[0][2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[0][2])
    i = axs[0][3].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_vi.J_security.T, shading='gouraud', cmap = 'plasma')
    axs[0][3].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[0][3])
    i = axs[0][4].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_vi.J_override.T, shading='gouraud', cmap = 'plasma')
    axs[0][4].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[0][4])
    axs[0][0].set_title('Commandes')
    axs[0][1].set_title('Cost to Go Totale')
    axs[0][2].set_title('Cost to Go Confort: '+ str(cf.confort_coef))
    axs[0][3].set_title('Cost to Go Sécurité: '+ str(cf.security_coef))
    axs[0][4].set_title('Cost to Go Liberté: '+ str(cf.override_coef))
    axs[0][0].set_ylabel('VI')
    
    i = axs[1][0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttc_cmap, shading='gouraud', cmap = 'plasma')
    axs[1][0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[1][0])
    i = axs[1][1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_ttc.J_total.T, shading='gouraud', cmap = 'plasma')
    axs[1][1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[1][1])
    i = axs[1][2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1],c2g_ttc.J_confort.T, shading='gouraud', cmap = 'plasma')
    axs[1][2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[1][2])
    i = axs[1][3].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_ttc.J_security.T, shading='gouraud', cmap = 'plasma')
    axs[1][3].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[1][3])
    i = axs[1][4].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_ttc.J_override.T, shading='gouraud', cmap = 'plasma')
    axs[1][4].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[1][4])
    axs[1][0].set_ylabel('TTC')
    
    i = axs[2][0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], msd_cmap, shading='gouraud', cmap = 'plasma')
    axs[2][0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[2][0])
    i = axs[2][1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_msd.J_total.T, shading='gouraud', cmap = 'plasma')
    axs[2][1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[2][1])
    i = axs[2][2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1],c2g_msd.J_confort.T, shading='gouraud', cmap = 'plasma')
    axs[2][2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[2][2])
    i = axs[2][3].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_msd.J_security.T, shading='gouraud', cmap = 'plasma')
    axs[2][3].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[2][3])
    i = axs[2][4].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_msd.J_override.T, shading='gouraud', cmap = 'plasma')
    axs[2][4].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[2][4])
    axs[2][0].set_ylabel('MSD')
    
     
    i = axs[3][0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], human_cmap, shading='gouraud', cmap = 'plasma')
    axs[3][0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[3][0])
    i = axs[3][1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_human.J_total.T, shading='gouraud', cmap = 'plasma')
    axs[3][1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[3][1])
    i = axs[3][2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1],c2g_human.J_confort.T, shading='gouraud', cmap = 'plasma')
    axs[3][2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[3][2])
    i = axs[3][3].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_human.J_security.T, shading='gouraud', cmap = 'plasma')
    axs[3][3].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[3][3])
    i = axs[3][4].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], c2g_human.J_override.T, shading='gouraud', cmap = 'plasma')
    axs[3][4].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    fig.colorbar(i, ax=axs[3][4])
    axs[3][0].set_ylabel('HUMAN')
