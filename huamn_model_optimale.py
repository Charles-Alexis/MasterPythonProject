# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 19:03:12 2023

@author: Charles-Alexis
"""

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
import controller
import dynamic_programming as dprog
import simulationv2 as s
###############################################################################

###############################################################################
#%%
temps_debut = time.time()

### ROAD AND DRIVER SETUP
road_nbr = 1
roads = ['AsphalteDry', 'AsphalteWet', 'CobblestoneWet', 'Snow', 'Ice']
road = roads[road_nbr]

### SYSTEM SETUP
sys = system.LongitudinalFrontWheelDriveCarWithHumanModel()

sys.x_ub = np.array([0., 10])
sys.x_lb = np.array([-50.0, 0.0])
sys.u_lb = np.array([-0.2])


sys.road = sys.roads[road]
sys.use_human_model = False

#%%
#COSTFUNCTION
slip_data = sys.return_max_mu()
dx_max = sys.f([0,sys.x_ub[1]],[-slip_data[1],1])
cf = costfunction.HumanModelCostFunction.from_sys(sys)

cf.a_max = dx_max[1]
cf.vit_max = sys.x_ub[1]

cf.velocity_coef = 1
cf.acceleration_coef = 1
cf.collision_coef = 1
cf.risk_coef = 1
cf.dt = 0.02
sys.cost_function = cf

#%%
#GRIDSYS
grid_sys = discretizer2.GridDynamicSystem(sys, [601, 401], [10], 0.003)
      
 #%%         
#DYNAMICPROGRAMMING
dp = dprog.DynamicProgrammingWithLookUpTable( grid_sys, cf)
#dp.compute_steps(1500,  treshhold=0.0001, animate_cost2go = False , animate_policy = False)

optimale_human_controller = dprog.LookUpTableController( grid_sys , dp.pi )
optimale_human_controller.k = 2

temps_fin = time.time()
print(temps_fin-temps_debut)

cl_sys_optimal_human = controller.ClosedLoopSystem( sys , optimale_human_controller ) 
cl_sys_optimal_human.cost_function = sys.cost_function

#%% SIMULATION

x0 = np.array([-10,4.5])
cl_sys_optimal_human.x0 = x0

sim_optimal_human = s.SimulatorV2(cl_sys_optimal_human, x0_end=sys.x_ub[0], vi_controller_flag=False)

args_optimal_human = sim_optimal_human.traj_to_args(sim_optimal_human.traj)


#%%
import cost2go2

cf_list = list((sys.cost_function.g_velocity, sys.cost_function.g_acceleration, sys.cost_function.g_risk, sys.cost_function.g_collision))
cf_list_name = list(('Velocity','Acceleration','Risk','Collision'))
c_optimal_human = cost2go2.cost2go_list(grid_sys, sys, cl_sys_optimal_human, cf_list, cf_list_name = cf_list_name)
c_optimal_human.compute_steps(print_iteration=False)
# c_optimal_human.plot_cost2go_map()

#%%  FULL PLOTTING

uk_0 = grid_sys.get_input_from_policy(dp.pi, 0)
u0 = grid_sys.get_grid_from_array(uk_0)

fig, axs = plt.subplots(1, 2)
plt.ion()
xname = sys.state_label[0] + ' ' + sys.state_units[0]
yname = sys.state_label[1] + ' ' + sys.state_units[1]  

axs[0].set_title('HUMAN U velocity: ' + str(cf.velocity_coef) + ' acceleration: ' + str(cf.acceleration_coef) + ' Risk: ' + str(cf.risk_coef) + ' collision: '+ str(cf.collision_coef))
axs[0].set(xlabel=xname, ylabel=yname)
i1 = axs[0].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], u0.T, shading='gouraud')
axs[0].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
fig.colorbar(i1, ax=axs[0])
axs[0].grid(True)
axs[0].plot(args_optimal_human[:,2],args_optimal_human[:,3])
axs[0].plot(args_optimal_human[:,0],args_optimal_human[:,1])

axs[1].set_title('human cost2go')
axs[1].set(xlabel=xname, ylabel=yname)
i5 = axs[1].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], np.clip(c_optimal_human.cost2go_map_list[0].T,0,500), shading='gouraud')
axs[1].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
fig.colorbar(i5, ax=axs[1])
axs[1].grid(True)
axs[1].plot(args_optimal_human[:,2],args_optimal_human[:,3])
axs[1].plot(args_optimal_human[:,0],args_optimal_human[:,1])

#fig2, axs2 = plt.subplots(9, 1)
#plt.ion()
#sim_optimal_human.plot_trajectories_human('VI', axs2, print_label=True)


# pos = grid_sys.x_level[0]
# vit = grid_sys.x_level[1]
# full_brake_array = np.zeros(grid_sys.x_grid_dim)
# full_brake_array_pos = np.zeros(grid_sys.x_grid_dim)
# full_brake_array_vit = np.zeros(grid_sys.x_grid_dim)
# no_brake_array = np.zeros(grid_sys.x_grid_dim)
# no_brake_array_pos = np.zeros(grid_sys.x_grid_dim)
# no_brake_array_vit = np.zeros(grid_sys.x_grid_dim)

# slip_data = sys.return_max_mu()

# def find_nearest_arg( value, array):
#         array = np.asarray(array)
#         idx = (np.abs(array - value)).argmin()
#         return idx

# for p in range(len(pos)):
#      for v in range(len(vit)):
#           a_min = sys.f([-10,vit[v]],[0,0])[1]
#           a_max = sys.f([-10,vit[v]],[-slip_data[1],1])[1]

#           dx_min = np.array([vit[v], 0])
#           dx_max = np.array([vit[v], a_max])
          
#           x_min = np.array([pos[p],vit[v]]) + dx_min * grid_sys.dt
#           x_max = np.array([pos[p],vit[v]]) + dx_max * grid_sys.dt
          
#           nx_pos_min_arg = find_nearest_arg(x_min[0], pos)
#           nx_vit_min_arg = find_nearest_arg(x_min[1], vit)

#           nx_pos_max_arg = find_nearest_arg(x_max[0], pos)
#           nx_vit_max_arg = find_nearest_arg(x_max[1], vit)
          
#           if nx_pos_min_arg == p and nx_vit_min_arg == v:
#                no_brake_array[p][v] = 10
#           else:
#                no_brake_array[p][v] = 0
          
#           if nx_pos_min_arg == p:
#                no_brake_array_pos[p][v] = 10
#           else:
#                no_brake_array_pos[p][v] = 0
               
#           if nx_vit_min_arg == v:
#                no_brake_array_vit[p][v] = 10
#           else:
#                no_brake_array_vit[p][v] = 0




#           if nx_pos_max_arg == p and nx_vit_max_arg == v:
#                full_brake_array[p][v] = 10
#           else:
#                full_brake_array[p][v] = 0
               
#           if nx_pos_max_arg == p:
#                full_brake_array_pos[p][v] = 10
#           else:
#                full_brake_array_pos[p][v] = 0
            
#           if nx_vit_max_arg == v:
#                full_brake_array_vit[p][v] = 10
#           else:
#                full_brake_array_vit[p][v] = 0 

# fig, axs = plt.subplots(2, 3)
# plt.ion()    
# axs[0][0].contourf(no_brake_array.T)
# axs[0][0].set_title('No Brake')    
# axs[0][1].contourf(no_brake_array_vit.T)
# axs[0][1].set_title('No Brake Vit')
# axs[0][2].contourf(no_brake_array_pos.T)
# axs[0][2].set_title('No Brake Pos')    

# axs[1][0].contourf(full_brake_array.T)
# axs[1][0].set_title('Full Brake')    
# axs[1][1].contourf(full_brake_array_vit.T)
# axs[1][1].set_title('Full Brake Vit')    
# axs[1][2].contourf(full_brake_array_pos.T)
# axs[1][2].set_title('Full Brake Pos')