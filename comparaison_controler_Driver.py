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
sys.tm = 2.8
sys.tm_dot = 0.5
sys.tf = 1.75

timing_conservateur = -0.4
timing_normal = +0.0
timing_aggressif = +0.4
timing_sleep = +100.0
E_mauvais = [[timing_conservateur, 0.10], [timing_normal, 0.20], [timing_aggressif, 0.70]]
E_normal = [[timing_conservateur, 0.25], [timing_normal, 0.50], [timing_aggressif, 0.25]]
E_bon = [[timing_conservateur, 0.10], [timing_normal, 0.80], [timing_aggressif, 0.10]]
E_sleep = [[timing_normal, 0.01], [timing_sleep, 0.99]]
E_null = [[+0.0, 1.0]]
E_arr_name = ['E_sleep', 'E_mauvais', 'E_normal', 'E_bon', 'E_null']
E_arr = [E_sleep, E_mauvais, E_normal, E_bon, E_null]

name_list = list()
grid_sys_list = list()
human_list = list()
dp_list = list()
ttc_list = list()
vi_list = list()
ttc_controller_list = list()
cost2go_list = list()
cost2go_ttc_list = list()

u_dim = [10,2]
x_dim = [300,150]
sys.x_grid = x_dim
dt = 0.02
sys.use_human_model = True

#COSTFUNCTION
cf = costfunction.DriverModelCostFunction.from_sys(sys)
cf.confort_coef = 5
cf.override_coef = 500
cf.security_coef = 25
cf.xbar = np.array( [(sys.x_ub[0]-1), 0] ) # target
sys.cost_function = cf


#%%
for d in range(len(E_arr)):
    E = E_arr[d]
    
    #GRIDSYS
    print('----------------------------------------------------------')
    print('dx max: ', sys.dmax)
    print('x dim: ', sys.x_grid)
    print('x0 boundaries: ' + str(sys.x_lb[0]) + ' to ' + str(str(sys.x_ub[0])))
    print('x1 boundaries: ' + str(sys.x_lb[1]) + ' to ' + str(str(sys.x_ub[1])))
    print('Esperance: ', E)
    print('----------------------------------------------------------')
    
    grid_sys = discretizer2.GridDynamicSystem(sys, sys.x_grid, u_dim, dt, esperance = E)

    #DYNAMICPROGRAMMING
    dp = dprog.DynamicProgrammingWithLookUpTable( grid_sys, cf, esperance = E)
    dp.compute_steps(5000,  treshhold=0.001, animate_policy=False, animate_cost2go=False, jmax = 100)
    vi_controller = dprog.LookUpTableController( grid_sys , dp.pi )
    vi_controller.k = 2
    
    cf_list = list((sys.cost_function.g, sys.cost_function.g_confort, sys.cost_function.g_security, sys.cost_function.g_override))
    human_list.append(np.copy(sys.plot_human_model_ttc(plot=False)))
    name_list.append(sys.road[-1]+'_'+E_arr_name[d])
    grid_sys_list.append(grid_sys)
    
    #VI
    cl_sys_vi = controller.ClosedLoopSystem( sys , vi_controller ) 
    cl_sys_vi.cost_function = sys.cost_function
    c_vi = cost2go2.cost2go_list(grid_sys, sys, cl_sys_vi, cf_list)
    c_vi.compute_steps(print_iteration=True)
    
    dp_list.append(dp)
    vi_list.append(cl_sys_vi)
    cost2go_list.append(c_vi)
    
    #TTC
    ttc_controller = BaselineController.TTCController(sys, grid_sys, slip_max=-slip_data[1])
    cl_sys_ttc = controller.ClosedLoopSystem( sys , ttc_controller )  
    cl_sys_ttc.cost_function = sys.cost_function
    
    c2g_ttc = cost2go2.cost2go_list(grid_sys, sys, cl_sys_ttc, cf_list)
    c2g_ttc.compute_steps(print_iteration=True)
    
    ttc_list.append(ttc_controller)
    ttc_controller_list.append(cl_sys_ttc)
    cost2go_ttc_list.append(c2g_ttc)

#%%
c_test = cost2go2.cost2go_esperance(grid_sys, sys, E_arr, cl_sys_ttc, cf_list)
c_test.create_x_next()

#%% Flag

plotting_flag = True
saving_flag = False

#%% PLOTTING
if plotting_flag:
    for d in range(len(E_arr)):
        fig, axs = plt.subplots(3, 5)
        plt.ion()
        fig.suptitle(str(name_list[d]))
        xname = sys.state_label[0] + ' ' + sys.state_units[0]
        yname = sys.state_label[1] + ' ' + sys.state_units[1]  
              
        u_vi = grid_sys.get_grid_from_array(grid_sys.get_input_from_policy(dp_list[d].pi, 0)).T
        u_ttc = ttc_list[d].c_array().T
        u_human = human_list[d].T
        
        c2g_tot = cost2go_list[d].cost2go_map_list[0].T
        c2g_confort = cost2go_list[d].cost2go_map_list[1].T
        c2g_security = cost2go_list[d].cost2go_map_list[2].T
        c2g_override = cost2go_list[d].cost2go_map_list[3].T
        
        c2g_ttc_tot = cost2go_ttc_list[d].cost2go_map_list[0].T
        c2g_ttc_confort = cost2go_ttc_list[d].cost2go_map_list[1].T
        c2g_ttc_security = cost2go_ttc_list[d].cost2go_map_list[2].T
        c2g_ttc_override = cost2go_ttc_list[d].cost2go_map_list[3].T
        
        c2g_tot_diff = cost2go_list[d].cost2go_map_list[0].T - cost2go_ttc_list[d].cost2go_map_list[0].T
        c2g_confort_diff = cost2go_list[d].cost2go_map_list[1].T - cost2go_ttc_list[d].cost2go_map_list[1].T
        c2g_security_diff = cost2go_list[d].cost2go_map_list[2].T - cost2go_ttc_list[d].cost2go_map_list[2].T
        c2g_override_diff = cost2go_list[d].cost2go_map_list[3].T - cost2go_ttc_list[d].cost2go_map_list[3].T
        
        
        cl_sys_vi = vi_list[d]
        cl_sys_ttc = ttc_controller_list[d]
        
        x0 = np.array([-10.,4.5])
        
        cl_sys_vi.x0 = x0
        cl_sys_ttc.x0 = x0
        
        sim_vi = s.SimulatorV2(cl_sys_vi, x0_end=sys.x_ub[0])
        args_vi = sim_vi.traj_to_args(sim_vi.traj)
        
        sim_ttc = s.SimulatorV2(cl_sys_ttc, x0_end=sys.x_ub[0])
        args_ttc = sim_vi.traj_to_args(sim_ttc.traj)
        
        axs[0][0].set_title('VI Optimale Command')
        i = axs[0][0].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], u_vi, shading='gouraud')
        axs[0][0].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
        fig.colorbar(i, ax=axs[0][0])
        axs[0][0].grid(True)
       
        axs[0][1].set_title('Cost2go')
        i = axs[0][1].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], c2g_tot, shading='gouraud')
        axs[0][1].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
        fig.colorbar(i, ax=axs[0][1])
        axs[0][1].grid(True)
        
        axs[0][2].set_title('Confort Coef: '+str(cf.confort_coef))
        i = axs[0][2].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], c2g_confort, shading='gouraud')
        axs[0][2].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
        fig.colorbar(i, ax=axs[0][2])
        axs[0][2].grid(True)  
        
        axs[0][3].set_title('Security Coef: ' + str(cf.security_coef))
        i = axs[0][3].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], c2g_security, shading='gouraud')
        axs[0][3].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
        fig.colorbar(i, ax=axs[0][3])
        axs[0][3].grid(True)        
        
        axs[0][4].set_title('Override Coef: '+str(cf.override_coef))
        i = axs[0][4].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], c2g_override, shading='gouraud')
        axs[0][4].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
        fig.colorbar(i, ax=axs[0][4])
        axs[0][4].grid(True)
        
        axs[1][0].set_title('TTC Command')
        i = axs[1][0].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], u_ttc, shading='gouraud')
        axs[1][0].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
        fig.colorbar(i, ax=axs[1, 0])
        axs[1][0].grid(True)

        i = axs[1][1].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], c2g_ttc_tot, shading='gouraud')
        axs[1][1].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
        fig.colorbar(i, ax=axs[1, 1])
        axs[1][1].grid(True)
        
        i = axs[1][2].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], c2g_ttc_confort, shading='gouraud')
        axs[1][2].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
        fig.colorbar(i, ax=axs[1, 2])
        axs[1][2].grid(True)  
        
        i = axs[1][3].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], c2g_ttc_security, shading='gouraud')
        axs[1][3].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
        fig.colorbar(i, ax=axs[1, 3])
        axs[1][3].grid(True)        
        
        i = axs[1][4].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], c2g_ttc_override, shading='gouraud')
        axs[1][4].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
        fig.colorbar(i, ax=axs[1 ,4])
        axs[1][4].grid(True)   
        
        axs[2][0].set_title('Human Command')     
        i = axs[2][0].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], u_human, shading='gouraud')
        axs[2][0].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
        fig.colorbar(i, ax=axs[2, 0])
        axs[2][0].grid(True)

        i = axs[2][1].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], c2g_tot_diff, shading='gouraud')
        axs[2][1].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
        fig.colorbar(i, ax=axs[2, 1])
        axs[2][1].grid(True)
        
        i = axs[2][2].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], c2g_confort_diff, shading='gouraud')
        axs[2][2].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
        fig.colorbar(i, ax=axs[2, 2])
        axs[2][2].grid(True)  
        
        i = axs[2][3].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], c2g_security_diff, shading='gouraud')
        axs[2][3].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
        fig.colorbar(i, ax=axs[2, 3])
        axs[2][3].grid(True)        
        
        i = axs[2][4].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], c2g_override_diff, shading='gouraud')
        axs[2][4].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
        fig.colorbar(i, ax=axs[2 ,4])
        axs[2][4].grid(True)  
        
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
        