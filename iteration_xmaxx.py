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
roads = ['AsphalteDry', 'AsphalteWet', 'CobblestoneWet', 'Snow', 'Ice']

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
sys.obs_dist = sys.x_ub[0]

### DRIVER SETUP
tm_roads = [2.2, 2.5, 2.8, 3.8, 4.2]
tm_dot_driver = [0.1, 0.3, 0.5, 0.7, 0.9]
timing_conservateur = -0.4
timing_normal = +0.0
timing_aggressif = +0.4
timing_sleep = +100.0
E_mauvais = [[timing_conservateur, 0.10], [timing_normal, 0.20], [timing_aggressif, 0.70]]
E_normal = [[timing_conservateur, 0.23], [timing_normal, 0.54], [timing_aggressif, 0.23]]
E_bon = [[timing_conservateur, 0.10], [timing_normal, 0.80], [timing_aggressif, 0.10]]
E_sleep = [[timing_normal, 0.01], [timing_sleep, 0.99]]
E_null = [[+0.0, 1.0]]

roads_array_flag = False
drivers_array_flag = True

E_arr_name = ['E_null', 'E_mauvais', 'E_normal', 'E_bon', 'E_sleep']

if drivers_array_flag: E_arr = [E_null, E_mauvais, E_normal, E_bon, E_sleep]
else: E_arr = [E_bon]

if roads_array_flag: roads = ['AsphalteDry', 'AsphalteWet', 'CobblestoneWet', 'Snow', 'Ice']
else: roads = ['AsphalteWet']

dp_list = list()
grid_sys_list = list()
vi_list = list()
name_list = list()
cost2go_list = list()

u_dim = [10,2]
x_dim = [550,150]
dt = 0.02

for r in range(len(roads)):
    for d in range(len(E_arr)):
        road = roads[r]
        E = E_arr[d]
        
        sys.road = sys.roads[road]
        sys.tm = tm_roads[r]
        sys.tf = 1.75
        sys.tm_dot = tm_dot_driver[r]
        sys.tm_dot = 0.5
        sys.x_grid = [550,150]
        slip_data = sys.return_max_mu()
        sys.dmax = sys.f([0,sys.x_ub[1]],[-slip_data[1],1])
        sys.use_human_model = True
        
        #COSTFUNCTION
        cf = costfunction.DriverModelCostFunction.from_sys(sys)
        cf.confort_coef = 5
        cf.override_coef = 2500
        cf.security_coef = 25
        cf.xbar = np.array( [(sys.x_ub[0]-1), 0] ) # target
        sys.cost_function = cf
        
        #GRIDSYS
        grid_sys = discretizer2.GridDynamicSystem(sys, sys.x_grid, [10, 2], dt, esperance = E)

        #DYNAMICPROGRAMMING
        dp = dprog.DynamicProgrammingWithLookUpTable( grid_sys, cf, esperance = E)
        dp.compute_steps(5000,  treshhold=0.001, animate_policy=False, animate_cost2go=False, jmax = 100)
        vi_controller = dprog.LookUpTableController( grid_sys , dp.pi )
        vi_controller.k = 2
        
        cf_list = list((sys.cost_function.g, sys.cost_function.g_confort, sys.cost_function.g_security, sys.cost_function.g_override))
        
        cl_sys_vi = controller.ClosedLoopSystem( sys , vi_controller ) 
        cl_sys_vi.cost_function = sys.cost_function
        c_vi = cost2go2.cost2go_list(grid_sys, sys, cl_sys_vi, cf_list)
        c_vi.compute_steps(print_iteration=False)
        
        dp_list.append(dp)
        grid_sys_list.append(grid_sys)
        vi_list.append(vi_controller)
        name_list.append(roads[r]+'_'+E_arr_name[d])
        cost2go_list.append(c_vi)
        

#%% Flag

plotting_flag = True
saving_flag = False

#%% PLOTTING
if plotting_flag:
    if drivers_array_flag and roads_array_flag:
        fig, axs = plt.subplots(len(roads), len(E_arr))
    else:
        fig, axs = plt.subplots(len(E_arr)*2, len(roads))
    plt.ion()
    fig.suptitle('Confort Coef: '+str(cf.confort_coef)+' Security Coef: ' + str(cf.security_coef) + ' Override Coef: '+str(cf.override_coef))
    xname = sys.state_label[0] + ' ' + sys.state_units[0]
    yname = sys.state_label[1] + ' ' + sys.state_units[1]  
    
    obstacle_array = np.zeros(len(grid_sys.x_level[1]))-0.5
    index = 0
    for r in range(len(roads)):
        for d in range(len(E_arr)):
            road = roads[r]
            E = E_arr[d]

            sys.road = sys.roads[road]
            sys.tm = tm_roads[r]
            sys.tm_dot = tm_dot_driver[r]
            sys.dmax = sys.f([0,sys.x_ub[1]],[-slip_data[1],1])
            sys.use_human_model = True
            
            u0 = grid_sys.get_grid_from_array(grid_sys.get_input_from_policy(dp_list[index].pi, 0))
    
            cl_sys_vi = controller.ClosedLoopSystem( sys , vi_list[index] ) 
            cl_sys_vi.cost_function = sys.cost_function
    
            x0 = np.array([-15.,4.5])
            cl_sys_vi.x0 = x0
            sim_vi = s.SimulatorV2(cl_sys_vi, x0_end=sys.x_ub[0])
            args_vi = sim_vi.traj_to_args(sim_vi.traj)
            if drivers_array_flag and roads_array_flag: 
                if r == 0:
                    axs[r][d].set_title(name_list[index])
                i = axs[r][d].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], u0.T, shading='gouraud')
                # axs[r][d].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
                axs[r][d].plot(obstacle_array,grid_sys.x_level[1])
                # fig.colorbar(i, ax=axs[r, d])
                axs[r][d].grid(True)
                axs[r][d].plot(args_vi[:,2],args_vi[:,3])
                axs[r][d].plot(args_vi[:,0],args_vi[:,1])
            elif drivers_array_flag is False and roads_array_flag is False: 
                axs[0].set_title(name_list[index])
                i = axs[0].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], u0.T, shading='gouraud')
                i = axs[1].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], np.clip(cost2go_list[index].cost2go_map_list[0].T,0,10000), shading='gouraud')
                axs[0].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
                axs[0].plot(obstacle_array,grid_sys.x_level[1])
                axs[1].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
                axs[1].plot(obstacle_array,grid_sys.x_level[1])
                # fig.colorbar(i, ax=axs[0, d + r])
                axs[0].grid(True)
                axs[1].grid(True)
                axs[0].plot(args_vi[:,2],args_vi[:,3])
                axs[0].plot(args_vi[:,0],args_vi[:,1]) 
                axs[1].plot(args_vi[:,2],args_vi[:,3])
                axs[1].plot(args_vi[:,0],args_vi[:,1])
            else:
                axs[0][d+r].set_title(name_list[index])
                i = axs[0][d+r].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], u0.T, shading='gouraud')
                i = axs[1][d+r].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], np.clip(cost2go_list[index].cost2go_map_list[0].T,0,10000), shading='gouraud')
                axs[0][d+r].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
                axs[0][d+r].plot(obstacle_array,grid_sys.x_level[1])
                axs[1][d+r].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
                axs[1][d+r].plot(obstacle_array,grid_sys.x_level[1])
                # fig.colorbar(i, ax=axs[0, d + r])
                axs[0][d+r].grid(True)
                axs[1][d+r].grid(True)
                axs[0][d+r].plot(args_vi[:,2],args_vi[:,3])
                axs[0][d+r].plot(args_vi[:,0],args_vi[:,1]) 
                axs[1][d+r].plot(args_vi[:,2],args_vi[:,3])
                axs[1][d+r].plot(args_vi[:,0],args_vi[:,1]) 
                
            index = index + 1 
            
#%% SAVING

if saving_flag:
    directory = 'xmaxx_policymap_Esp/'+str(date.today().day)+'_'+str(date.today().month)+'_'+str(date.today().year)+'_'+str(time.localtime().tm_hour)+'h'+str(time.localtime().tm_min)
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, directory)
    if not os.path.exists(final_directory):
       os.makedirs(final_directory)
    
    os.chdir(final_directory)
    with open('cf_config.txt', 'w') as f:
        f.write('Confort Coefficient: ' + str(cf.confort_coef)+'\n')  
        f.write('Override Coefficient: ' + str(cf.override_coef)+'\n')  
        f.write('Security Coefficient: ' + str(cf.security_coef)+'\n') 
        f.write(str(sys.roads)+'\n')    
        f.write('pos_dim: '+str(x_dim[0])+'\n')      
        f.write('vit_dim: '+str(x_dim[1])+'\n')    
        f.write('slip: '+str(u_dim[0])+'\n')        
        f.write('dt: '+str(dt)+'\n')
        f.write('time margin: '+str(tm_roads)+'\n')
        f.write('time margin dot: '+str(sys.tm_dot)+'\n')
        f.write('E_mauvais: '+str(E_mauvais)+'\n')
        f.write('E_normal: '+str(E_normal)+'\n')
        f.write('E_bon: '+str(E_bon)+'\n')
        f.write('E_sleep: '+str(E_sleep)+'\n')
        f.write('E_null: '+str(E_null)+'\n')
    os.chdir(current_directory)
    for index in range(len(name_list)):
        os.chdir(final_directory)
        dp_list[index].save_latest('xmaxx_' + name_list[index])
        os.chdir(current_directory)
        