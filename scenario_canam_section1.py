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
temps_debut = time.time()

roads = ['AsphalteWet']
drivers = ['Null']
comparaison = ['NoDiff', 'Normal','Bool','NearObstacle','HighSpeed']

x_dims = [[600,100]]
x_lbs = [[-100., 0]]
tm_arr = [4.2]
controler = ['Vi','Ttc','Msd', 'Human']

## SCÉNARIO
roads_to_test = [0]  
drivers_to_test = [0]
controlers_to_test = [0,1,2,3]
compare_mode = [0]

print('---------------------------------------')
print('Scénario')
print('Roads: ', roads_to_test)
print('Driver: ', drivers_to_test)
print('Controlers: ', controlers_to_test)
print('---------------------------------------')

## CREATING LIST FOR DATA
name_list = list()
grid_sys_list = list()
grid_sys_controller_list = list()
human_list = list()

vi_list = list()
ttc_list = list()
msd_list = list()
human_list = list()

vi_cl_list = list()
ttc_cl_list = list()
msd_cl_list = list()
human_cl_list = list()

vi_c2g_list = list()
ttc_c2g_list = list()
msd_c2g_list = list()
human_c2g_list = list()


#%%
compute_c2g_flag = True

for r in range(len(roads_to_test)):
    for d in range(len(drivers_to_test)): 
        print('---------------------------------------')
        print('Iteration: ' + str(1 + r*len(drivers_to_test) + d))
        print('Roads: ', roads[roads_to_test[r]])
        print('Driver: ', drivers[drivers_to_test[d]])

        u_dim = [150,2]
        ### SYSTEM CONFIGURATION
        sys = system.LongitudinalFrontWheelDriveCarWithDriverModel()
        sys.mass = 760
        sys.lenght = 3.35
        sys.xc = sys.lenght/2
        sys.yc = 1.74/2 
        sys.cdA = 0.3 * (1.84*1.74)
        sys.x_ub = np.array([0 , 20.0])
        sys.x_lb = np.array(x_lbs[roads_to_test[r]])
        sys.u_ub = np.array([0.0, 1])
        sys.u_lb = np.array([-0.3, 0])
        sys.u_dim = u_dim
        sys.m = len(sys.u_dim)
        sys.obs_dist = sys.x_ub[0]
        sys.x_grid = x_dims[roads_to_test[r]]
        sys.road = sys.roads[roads[roads_to_test[r]]]
        sys.timing = 0.75
        sys.driver = sys.drivers[drivers[drivers_to_test[d]]]
        
        
        slip_data = sys.return_max_mu()
        sys.dmax = sys.f([0,sys.x_ub[1]],[-slip_data[1],1])
        sys.best_slip = sys.return_max_mu()[1]
        
        ### DRIVER MODEL CONFIGURATION
        sys.tm = tm_arr[roads_to_test[r]]
        sys.tm_dot = 0.75
        sys.tf = 1.75
        
        ### COST FUNCTION CONFIGURATION
        cf = costfunction.DriverModelCostFunction.from_sys(sys) 
        cf.confort_coef = 0.01
        cf.override_coef = 10
        cf.security_coef = 100
        cf.security_slope = 3
        cf.security_distance = sys.lenght
        sys.cost_function = cf
        cf_list = list((sys.cost_function.g, sys.cost_function.g_confort, sys.cost_function.g_security, sys.cost_function.g_override))
        
        print('dx max: ', sys.dmax)
        print('x dim: ', sys.x_grid)
        print('x0 boundaries: ' + str(sys.x_lb[0]) + ' to ' + str(str(sys.x_ub[0])))
        print('x1 boundaries: ' + str(sys.x_lb[1]) + ' to ' + str(str(sys.x_ub[1])))
        print('Esperance: ', sys.driver[0])
        
        ### GRID SYSTEM CONFIGURATION
        dt = 0.02 
        
        controlers_dim = [500,500]
        controlers_dim = x_dims[roads_to_test[r]]
        
        print('Number of states: ',sys.x_grid[0] * sys.x_grid[1])
        print('Number of actions: ', u_dim[0])
        print('Number of actions-states: ', sys.x_grid[0] * sys.x_grid[1] * u_dim[0])
        print('----------------------------------------------------------')
        
        temps_debut_grid_sys = time.time()
        for c2t in controlers_to_test:
            if c2t == 0:
                    grid_sys = discretizer2.GridDynamicSystem(sys, sys.x_grid, u_dim, dt, esperance = sys.driver[0], print_data=False)
                    grid_sys_list.append(grid_sys)
                    
        grid_sys_controller = discretizer2.GridDynamicSystem(sys, controlers_dim, u_dim, dt, esperance = sys.driver[0], print_data=False, lookup = False)
        grid_sys_controller_list.append(grid_sys_controller)
        temps_fin_grid_sys = time.time()
        print('Grid Sys Computing Time: '+ str(temps_fin_grid_sys - temps_debut_grid_sys) + ' secondes' )
        
        ## SAVING METADATA
        name_list.append('Road: ' + roads[roads_to_test[r]] + '     Driver: ' + drivers[drivers_to_test[d]])
        
        
        ### CONTROLER CONFIGURATION
        for c in range(len(controlers_to_test)): 
            ## VALUE ITERATION
            if controler[controlers_to_test[c]] == 'Vi':
                dp = dprog.DynamicProgrammingWithLookUpTable(grid_sys, cf, esperance = sys.driver[0])
                dp.compute_steps(5000,  treshhold=0.0001, animate_policy=False, animate_cost2go=False, jmax = 1000)
                vi_controller = dprog.LookUpTableController( grid_sys , dp.pi )
                vi_controller.k = 2
                cl_sys_vi = controller.ClosedLoopSystem( sys , vi_controller ) 
                cl_sys_vi.cost_function = sys.cost_function
                
                if compute_c2g_flag: 
                     c2g_vi = cost2go2.cost2go_list(grid_sys, sys, cl_sys_vi, cf_list)
                     c2g_vi.compute_steps(print_iteration=False)
                     vi_c2g_list.append(c2g_vi)
                
                ## SAVING VI
                vi_list.append(vi_controller)
                vi_cl_list.append(cl_sys_vi)
                
            ## TIME TO COLLISION
            if controler[controlers_to_test[c]] == 'Ttc':
                ttc_controller = BaselineController.TTCController(sys, grid_sys_controller)
                cl_sys_ttc = controller.ClosedLoopSystem( sys , ttc_controller )  
                cl_sys_ttc.cost_function = sys.cost_function
                if compute_c2g_flag: 
                     c2g_ttc = cost2go2.cost2go_list(grid_sys_controller, sys, cl_sys_ttc, cf_list)
                     c2g_ttc.compute_steps(print_iteration=False)
                     ttc_c2g_list.append(c2g_ttc)
                
                ## SAVING TTC
                ttc_list.append(ttc_controller)
                ttc_cl_list.append(cl_sys_ttc)
            
            ## MINIMAL STOPPING DISTANCE    
            if controler[controlers_to_test[c]] == 'Msd':
                msd_controller = BaselineController.MSDController(sys, grid_sys_controller)
                cl_sys_msd = controller.ClosedLoopSystem( sys , msd_controller )  
                cl_sys_msd.cost_function = sys.cost_function
                if compute_c2g_flag: 
                     c2g_msd = cost2go2.cost2go_list(grid_sys_controller, sys, cl_sys_msd, cf_list)
                     c2g_msd.compute_steps(print_iteration=False)
                     msd_c2g_list.append(c2g_msd)
                
                ## SAVING MSD
                msd_list.append(msd_controller)
                msd_cl_list.append(cl_sys_msd)
                
            if controler[controlers_to_test[c]] == 'Human':
                human_controller = BaselineController.humanController(sys, grid_sys_controller)
                cl_sys_human = controller.ClosedLoopSystem( sys , human_controller ) 
                cl_sys_human.cost_function = sys.cost_function
                if compute_c2g_flag: 
                     c2g_human = cost2go2.cost2go_list(grid_sys_controller, sys, cl_sys_human, cf_list)
                     c2g_human.compute_steps(print_iteration=False)
                     human_c2g_list.append(c2g_human)
                
                human_list.append(human_controller)
                human_cl_list.append(cl_sys_human)


#%% ONLY PLOT C2G
if compute_c2g_flag:
    for n in range(len(name_list)):
        controlers = list()
        cost_to_go = list()
        
        fig, axs = plt.subplots(1, len(controlers_to_test))
        plt.ion()
        fig.suptitle('Coût à venir des différentes loi de commande')
        xname = sys.state_label[0] + ' ' + sys.state_units[0]
        yname = sys.state_label[1] + ' ' + sys.state_units[1] 
        
        for con in controlers_to_test:
            if controler[controlers_to_test[con]] == 'Vi':
                grid_sys_plot = grid_sys_list[n]
            else:
                grid_sys_plot = grid_sys_controller_list[n]
                
            if controler[con] == 'Vi':  
                controlers.append(grid_sys_list[n].get_grid_from_array(grid_sys_list[n].get_input_from_policy(vi_list[n].pi, 0)).T)
                cost_to_go.append(vi_c2g_list[n])
                i = axs[con].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], vi_c2g_list[n].cost2go_map_list[0].T, shading='gouraud', cmap = 'plasma')
                
            if controler[con] == 'Ttc':  
                controlers.append(ttc_list[n].c_array().T)
                cost_to_go.append(ttc_c2g_list[n])
                i = axs[con].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], ttc_c2g_list[n].cost2go_map_list[0].T, shading='gouraud', cmap = 'plasma')
                
            if controler[con] == 'Msd':  
                controlers.append(msd_list[n].c_array().T)
                cost_to_go.append(msd_c2g_list[n])
                i = axs[con].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], msd_c2g_list[n].cost2go_map_list[0].T, shading='gouraud', cmap = 'plasma')
                
            if controler[con] == 'Human':  
                controlers.append(human_list[n].c_array().T)
                cost_to_go.append(human_c2g_list[n])
                i = axs[con].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], human_c2g_list[n].cost2go_map_list[0].T, shading='gouraud', cmap = 'plasma')
                
            axs[con].axis([grid_sys_plot.x_level[0][0], grid_sys_plot.x_level[0][-1], grid_sys_plot.x_level[1][0], grid_sys_plot.x_level[1][-1]])
            fig.colorbar(i, ax=axs[con])
            axs[con].grid(True)
            
