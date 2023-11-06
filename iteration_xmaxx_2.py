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
###############################################################################

###############################################################################
#%%
temps_debut = time.time()

### ROAD AND DRIVER SETUP
road_nbr = 0

roads = ['AsphalteDry', 'AsphalteWet', 'CobblestoneWet', 'Snow', 'Ice']
road = roads[road_nbr]

### SYSTEM SETUP
sys = system.LongitudinalFrontWheelDriveCarWithDriverModel()
sys.mass = 20
sys.lenght = 0.48
sys.xc = 0.24
sys.yc = 0.15 
sys.mass = 20
sys.cdA = 0.3 * 0.105

sys.x_ub = np.array([0 , 4.5])

sys.x_lb = np.array([-15., 0])
sys.u_ub = np.array([0.0, 1])
sys.u_lb = np.array([-0.20, 0])
sys.road = sys.roads[road]
sys.obs_dist = sys.x_ub[0]


tm_roads = [2.2, 2.5, 2.8, 3.8, 4.2]
tm_dot_driver = [0.1, 0.3, 0.5, 0.7, 0.9]

timing_conservateur = -0.2
timing_normal = +0.0
timing_aggressif = +0.2
timing_sleep = +10.0
E_mauvais = [[timing_conservateur, 0.10], [timing_normal, 0.30], [timing_aggressif, 0.59], [timing_sleep, 0.01]]
E_normal = [[timing_conservateur, 0.33], [timing_normal, 0.33], [timing_aggressif, 0.33], [timing_sleep, 0.01]]
E_bon = [[timing_conservateur, 0.30], [timing_normal, 0.59], [timing_aggressif, 0.10], [timing_sleep, 0.01]]
E_sleep = [[timing_conservateur, 0.01], [timing_normal, 0.01], [timing_aggressif, 0.01], [timing_sleep, 0.97]]
E_null = [[+0.0, 1.0]]

E_arr = [E_mauvais, E_normal, E_bon, E_sleep, E_null]
driver_nbr = 4
E = E_arr[driver_nbr]

sys.tm = tm_roads[road_nbr]
sys.tf = 1.75
sys.tm_dot = tm_dot_driver[road_nbr]
sys.tm_dot = 0.5
sys.x_grid = [550,150]

slip_data = sys.return_max_mu()
sys.dmax = sys.f([0,sys.x_ub[1]],[-slip_data[1],1])

test_name = 'Tmargin: '+str(sys.tm)+' Treaction: '+str(sys.tf)+' Tdmargin: '+str(sys.tm_dot)

# sys.plot_human_model_ttc(name = test_name)


sys.use_human_model = True

#%%
#COSTFUNCTION
cf = costfunction.DriverModelCostFunction.from_sys(sys)

cf.confort_coef = 50
cf.override_coef = 1000
cf.security_coef = 25

cf.xbar = np.array( [(sys.x_ub[0]-1), 0] ) # target
sys.cost_function = cf

#%%
#GRIDSYS
grid_sys = discretizer2.GridDynamicSystem(sys, sys.x_grid, [10, 2], 0.02, esperance = E)


#%%
#DYNAMICPROGRAMMING
import dynamic_programming as dprog
dp = dprog.DynamicProgrammingWithLookUpTable( grid_sys, cf, esperance = E)

dp.compute_steps(5000,  treshhold=0.0005, animate_policy=False, animate_cost2go=False, jmax = 100)
vi_controller = dprog.LookUpTableController( grid_sys , dp.pi )
vi_controller.k = 2

#%% CLOSE LOOP SYSTEM
### CONTROLLER FOR TTC AND MSD
ttc_security_factor = 1
ttc_max_time = np.array([0.5075, 0.8671, 2.1370, 4.4723, 16.7718])
ttc_slip = np.array([-0.143, -0.112, -0.116, -0.052, -0.022])

ttc_controller = BaselineController.TTCController(sys, grid_sys, sys.human_model_time_margin, ttc_ref=ttc_max_time[road_nbr], position_obs=-0.1, slip_cmd=ttc_slip[road_nbr])
msd_controller = BaselineController.MSDController(sys, grid_sys)

### CLOSE LOOP SYSTEM FOR TTC AND MSD
cl_sys_vi = controller.ClosedLoopSystem( sys , vi_controller ) 
cl_sys_vi.cost_function = sys.cost_function
cl_sys_ttc = controller.ClosedLoopSystem(sys , ttc_controller)
cl_sys_ttc.cost_function = sys.cost_function 
cl_sys_msd = controller.ClosedLoopSystem(sys , msd_controller)
cl_sys_msd.cost_function = sys.cost_function 

#%% HUMAN CONTROLLER

human_controler = BaselineController.humanController(sys, grid_sys, sys.human_model_time_margin)
cl_sys_human = controller.ClosedLoopSystem(sys , human_controler)
cl_sys_human.cost_function = sys.cost_function

#%% SIMULATION

x0 = np.array([-15.,4.5])
cl_sys_vi.x0 = x0
cl_sys_ttc.x0 = x0
cl_sys_msd.x0 = x0
cl_sys_human.x0 = x0

sim_vi = s.SimulatorV2(cl_sys_vi, x0_end=sys.x_ub[0])
sim_ttc = s.SimulatorV2(cl_sys_ttc, x0_end=sys.x_ub[0])
sim_msd = s.SimulatorV2(cl_sys_msd, x0_end=sys.x_ub[0])
sim_human = s.SimulatorV2(cl_sys_human, x0_end=sys.x_ub[0])

args_vi = sim_vi.traj_to_args(sim_vi.traj)
args_ttc = sim_ttc.traj_to_args(sim_ttc.traj)
args_msd = sim_msd.traj_to_args(sim_msd.traj)
args_human = sim_human.traj_to_args(sim_human.traj)

#%%
import cost2go2

cf_list = list((sys.cost_function.g, sys.cost_function.g_confort, sys.cost_function.g_security, sys.cost_function.g_override))

c_vi = cost2go2.cost2go_list(grid_sys, sys, cl_sys_vi, cf_list)
c_vi.compute_steps(print_iteration=False)
c_vi.plot_cost2go_map()
# c_vi.plot_bugged_states()


#c_ttc = cost2go2.cost2go_list(grid_sys, sys, cl_sys_ttc, cf_list)
# c_ttc.compute_steps(print_iteration=False)

#c_msd = cost2go2.cost2go_list(grid_sys, sys, cl_sys_msd, cf_list)
# c_msd.compute_steps(print_iteration=False)

c_human = cost2go2.cost2go_list(grid_sys, sys, cl_sys_human, cf_list)
# c_human.compute_steps(print_iteration=False)

#%%  FULL PLOTTING

temps_fin = time.time()
print(temps_fin-temps_debut)

uk_0 = grid_sys.get_input_from_policy(dp.pi, 0)
u0 = grid_sys.get_grid_from_array(uk_0)
u0_ttc = ttc_controller.c_array()
u0_msd = msd_controller.c_array()
sys.x_grid = grid_sys.x_grid_dim
human_model = sys.plot_human_model_ttc(plot=False)


#fig, axs = plt.subplots(2, 4)
#plt.ion()
##fig.suptitle('Optimale baseline for Driver: ' + driv + ' On road: ' + road)
#fig.suptitle('Confort Coef: '+str(cf.confort_coef)+' Security Coef: ' + str(cf.security_coef) + ' Override Coef: '+str(cf.override_coef))
#xname = sys.state_label[0] + ' ' + sys.state_units[0]
#yname = sys.state_label[1] + ' ' + sys.state_units[1]  
#
#axs[0][0].set_title('VI OPTIMAL U')
## axs[0][0].set(xlabel=xname, ylabel=yname)
#i1 = axs[0][0].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], u0.T, shading='gouraud')
#axs[0][0].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
#fig.colorbar(i1, ax=axs[0, 0])
#axs[0][0].grid(True)
#axs[0][0].plot(args_vi[:,2],args_vi[:,3])
#axs[0][0].plot(args_vi[:,0],args_vi[:,1])
#   
#axs[0][1].set_title('TTC U')
## axs[0][1].set(xlabel=xname, ylabel=yname)
#i2 = axs[0][1].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], u0_ttc.T, shading='gouraud')
#axs[0][1].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
#fig.colorbar(i2, ax=axs[0, 1])
#axs[0][1].grid(True)
#axs[0][1].plot(args_ttc[:,2],args_ttc[:,3])
#axs[0][1].plot(args_ttc[:,0],args_ttc[:,1])
#
#axs[0][2].set_title('MSD U')
## axs[0][2].set(xlabel=xname, ylabel=yname)
#i3 = axs[0][2].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], u0_msd.T, shading='gouraud')
#axs[0][2].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
#fig.colorbar(i3, ax=axs[0, 2])
#axs[0][2].grid(True)
#axs[0][2].plot(args_msd[:,2],args_msd[:,3])
#axs[0][2].plot(args_msd[:,0],args_msd[:,1])
#   
#axs[0][3].set_title('HUMAN MODEL COMMANDS')
## axs[0][3].set(xlabel=xname, ylabel=yname)
#i4 = axs[0][3].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], human_model.T, shading='gouraud')
#axs[0][3].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
#fig.colorbar(i4, ax=axs[0, 3])
#axs[0][3].grid(True)
#axs[0][3].plot(args_human[:,2],args_human[:,3])
#axs[0][3].plot(args_human[:,0],args_human[:,1])
#
#axs[1][0].set_title('VI cost2go')
#axs[1][0].set(xlabel=xname, ylabel=yname)
#i5 = axs[1][0].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], np.clip(c_vi.cost2go_map_list[0].T,0,10000), shading='gouraud')
#axs[1][0].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
#fig.colorbar(i5, ax=axs[1, 0])
#axs[1][0].grid(True)
#axs[1][0].plot(args_vi[:,2],args_vi[:,3])
#axs[1][0].plot(args_vi[:,0],args_vi[:,1])
#
#axs[1][1].set_title('TTC cost2go')
#axs[1][1].set(xlabel=xname, ylabel=yname)
#i6 = axs[1][1].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], np.clip(c_ttc.cost2go_map_list[0].T,0,10000), shading='gouraud')
#axs[1][1].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
#fig.colorbar(i6, ax=axs[1, 1])
#axs[1][1].grid(True)
#axs[1][1].plot(args_ttc[:,2],args_ttc[:,3])
#axs[1][1].plot(args_ttc[:,0],args_ttc[:,1])
#
#axs[1][2].set_title('MSD cost2go')
#axs[1][2].set(xlabel=xname, ylabel=yname)
#i7 = axs[1][2].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], np.clip(c_msd.cost2go_map_list[0].T,0,10000), shading='gouraud')
#axs[1][2].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
#fig.colorbar(i7, ax=axs[1, 2])
#axs[1][2].grid(True)
#axs[1][2].plot(args_msd[:,2],args_msd[:,3])
#axs[1][2].plot(args_msd[:,0],args_msd[:,1])
#
#axs[1][3].set_title('TTC cost2go')
#axs[1][3].set(xlabel=xname, ylabel=yname)
#i8 = axs[1][3].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], np.clip(c_human.cost2go_map_list[0].T,0,10000), shading='gouraud')
#axs[1][3].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
#fig.colorbar(i8, ax=axs[1, 3])
#axs[1][3].grid(True)
#axs[1][3].plot(args_vi[:,2],args_vi[:,3])
#axs[1][3].plot(args_vi[:,0],args_vi[:,1])
#axs[1][3].plot(args_ttc[:,2],args_ttc[:,3])
#axs[1][3].plot(args_ttc[:,0],args_ttc[:,1])
#axs[1][3].plot(args_msd[:,2],args_msd[:,3])
#axs[1][3].plot(args_msd[:,0],args_msd[:,1])
#axs[1][3].plot(args_human[:,2],args_human[:,3])
#axs[1][3].plot(args_human[:,0],args_human[:,1])

fig, axs = plt.subplots(2, 2)
plt.ion()
#fig.suptitle('Optimale baseline for Driver: ' + driv + ' On road: ' + road)
fig.suptitle('Confort Coef: '+str(cf.confort_coef)+' Security Coef: ' + str(cf.security_coef) + ' Override Coef: '+str(cf.override_coef))
xname = sys.state_label[0] + ' ' + sys.state_units[0]
yname = sys.state_label[1] + ' ' + sys.state_units[1]  

axs[0][0].set_title('VI OPTIMAL U')
# axs[0][0].set(xlabel=xname, ylabel=yname)
i1 = axs[0][0].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], u0.T, shading='gouraud')
axs[0][0].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
fig.colorbar(i1, ax=axs[0, 0])
axs[0][0].grid(True)
axs[0][0].plot(args_vi[:,2],args_vi[:,3])
axs[0][0].plot(args_vi[:,0],args_vi[:,1])
   
axs[0][1].set_title('HUMAN MODEL COMMANDS')
# axs[0][3].set(xlabel=xname, ylabel=yname)
i4 = axs[0][1].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], human_model.T, shading='gouraud')
axs[0][1].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
fig.colorbar(i4, ax=axs[0, 1])
axs[0][1].grid(True)
axs[0][1].plot(args_human[:,2],args_human[:,3])
axs[0][1].plot(args_human[:,0],args_human[:,1])

axs[1][0].set_title('VI cost2go')
axs[1][0].set(xlabel=xname, ylabel=yname)
i5 = axs[1][0].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], np.clip(c_vi.cost2go_map_list[0].T,0,10000), shading='gouraud')
axs[1][0].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
fig.colorbar(i5, ax=axs[1, 0])
axs[1][0].grid(True)
axs[1][0].plot(args_vi[:,2],args_vi[:,3])
axs[1][0].plot(args_vi[:,0],args_vi[:,1])

axs[1][1].set_title('TTC cost2go')
axs[1][1].set(xlabel=xname, ylabel=yname)
i8 = axs[1][1].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], np.clip(c_human.cost2go_map_list[0].T,0,10000), shading='gouraud')
axs[1][1].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
fig.colorbar(i8, ax=axs[1, 1])
axs[1][1].grid(True)
axs[1][1].plot(args_vi[:,2],args_vi[:,3])
axs[1][1].plot(args_vi[:,0],args_vi[:,1])
axs[1][1].plot(args_human[:,2],args_human[:,3])
axs[1][1].plot(args_human[:,0],args_human[:,1])

# fig2, axs2 = plt.subplots(9, 2)
# plt.ion()
# fig2.suptitle('Simulation parameters for ' + ' On road: ' + road)
# sim_vi.plot_trajectories_new_figure('VI', axs2[:,0], print_label=True)
# sim_human.plot_trajectories_new_figure('HUMAN', axs2[:,1], print_label=True)

# fig2, axs2 = plt.subplots(9, 4)
# plt.ion()
# fig2.suptitle('Simulation parameters for ' + ' On road: ' + road)
# sim_vi.plot_trajectories_new_figure('VI', axs2[:,0], print_label=True)
# sim_ttc.plot_trajectories_new_figure('TTC', axs2[:,1], print_label=True)
# sim_msd.plot_trajectories_new_figure('MSD', axs2[:,2], print_label=True)
# sim_human.plot_trajectories_new_figure('HUMAN', axs2[:,3], print_label=True)

## SAVING OPTIMAL HUMAN MODEl
#dp.save_latest(name='optimal_human_model_1_0_1/'+roads[road_nbr])
#
#pos = grid_sys.x_level[0]
#vit = grid_sys.x_level[1]
#full_brake_array = np.zeros(grid_sys.x_grid_dim)
#full_brake_array_pos = np.zeros(grid_sys.x_grid_dim)
#full_brake_array_vit = np.zeros(grid_sys.x_grid_dim)
#no_brake_array = np.zeros(grid_sys.x_grid_dim)
#no_brake_array_pos = np.zeros(grid_sys.x_grid_dim)
#no_brake_array_vit = np.zeros(grid_sys.x_grid_dim)
#
#slip_data = sys.return_max_mu()
#
#def find_nearest_arg( value, array):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx
#
#for p in range(len(pos)):
#     for v in range(len(vit)):
#          a_min = sys.f([-10,vit[v]],[0,0])[1]
#          a_max = sys.f([-10,vit[v]],[-slip_data[1],1])[1]
#
#          dx_min = np.array([vit[v], 0])
#          dx_max = np.array([vit[v], a_max])
#          
#          x_min = np.array([pos[p],vit[v]]) + dx_min * grid_sys.dt
#          x_max = np.array([pos[p],vit[v]]) + dx_max * grid_sys.dt
#          
#          nx_pos_min_arg = find_nearest_arg(x_min[0], pos)
#          nx_vit_min_arg = find_nearest_arg(x_min[1], vit)
#
#          nx_pos_max_arg = find_nearest_arg(x_max[0], pos)
#          nx_vit_max_arg = find_nearest_arg(x_max[1], vit)
#          
#          if nx_pos_min_arg == p and nx_vit_min_arg == v:
#               no_brake_array[p][v] = 10
#          else:
#               no_brake_array[p][v] = 0
#          
#          if nx_pos_min_arg == p:
#               no_brake_array_pos[p][v] = 10
#          else:
#                no_brake_array_pos[p][v] = 0
#               
#          if nx_vit_min_arg == v:
#               no_brake_array_vit[p][v] = 10
#          else:
#               no_brake_array_vit[p][v] = 0
#
#
#
#
#          if nx_pos_max_arg == p and nx_vit_max_arg == v:
#               full_brake_array[p][v] = 10
#          else:
#               full_brake_array[p][v] = 0
#               
#          if nx_pos_max_arg == p:
#               full_brake_array_pos[p][v] = 10
#          else:
#               full_brake_array_pos[p][v] = 0
#            
#          if nx_vit_max_arg == v:
#               full_brake_array_vit[p][v] = 10
#          else:
#               full_brake_array_vit[p][v] = 0 
#
#fig, axs = plt.subplots(2, 3)
#plt.ion()    
#axs[0][0].contourf(no_brake_array.T)
#axs[0][0].set_title('No Brake')    
#axs[0][1].contourf(no_brake_array_vit.T)
#axs[0][1].set_title('No Brake Vit')
#axs[0][2].contourf(no_brake_array_pos.T)
#axs[0][2].set_title('No Brake Pos')    
#axs[1][0].contourf(full_brake_array.T)
#axs[1][0].set_title('Full Brake')    
#axs[1][1].contourf(full_brake_array_vit.T)
#axs[1][1].set_title('Full Brake Vit')    
#axs[1][2].contourf(full_brake_array_pos.T)
#axs[1][2].set_title('Full Brake Pos')