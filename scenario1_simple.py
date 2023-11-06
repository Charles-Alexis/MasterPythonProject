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

roads = ['AsphalteDry','CementDry','AsphalteWet','CobblestoneDry','CobblestoneWet','Snow','Ice']
drivers = ['Good','Normal','Bad','Sleepy','Null']
comparaison = ['NoDiff', 'Normal','Bool','NearObstacle','HighSpeed']

x_dims = [[300,80],[300,85],[300,100],[500,120],[1000,150],[1500,600],[160,600]]

x_lbs = [[-5., 0],[-5., 0],[-5., 0],[-5., 0],[-10., 0],[-15., 0],[-15., 0]]
tm_arr = [2.1, 2.15, 2.2, 2.5, 2.8, 4,0, 5.0]
controler = ['Vi','Ttc','Msd', 'Human']

## SCÉNARIO
roads_to_test = [1]  
drivers_to_test = [0,1,2]
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

for r in range(len(roads_to_test)):
    for d in range(len(drivers_to_test)): 
        print('---------------------------------------')
        print('Iteration: ' + str(1 + r*len(drivers_to_test) + d))
        print('Roads: ', roads[roads_to_test[r]])
        print('Driver: ', drivers[drivers_to_test[d]])

        u_dim = [150,2]
        ### SYSTEM CONFIGURATION
        sys = system.LongitudinalFrontWheelDriveCarWithDriverModel()
        sys.mass = 20
        sys.lenght = 0.48
        sys.xc = 0.24
        sys.yc = 0.15 
        sys.mass = 20
        sys.cdA = 0.3 * 0.105
        sys.x_ub = np.array([0 , 4.5])
        sys.x_lb = np.array(x_lbs[roads_to_test[r]])
        sys.u_ub = np.array([0.0, 1])
        sys.u_lb = np.array([-0.3, 0])
        sys.u_dim = u_dim
        sys.m = len(sys.u_dim)
        sys.obs_dist = sys.x_ub[0]
        sys.x_grid = x_dims[roads_to_test[r]]
        sys.road = sys.roads[roads[roads_to_test[r]]]
        sys.driver = sys.drivers[drivers[drivers_to_test[d]]]
        slip_data = sys.return_max_mu()
        sys.dmax = sys.f([0,sys.x_ub[1]],[-slip_data[1],1])
        
        ### DRIVER MODEL CONFIGURATION
        sys.tm = tm_arr[roads_to_test[r]]
        sys.tm_dot = 0.75
        sys.tf = 1.75
        
        ### COST FUNCTION CONFIGURATION
        cf = costfunction.DriverModelCostFunction.from_sys(sys) 
        cf.confort_coef = 0.01
        cf.override_coef = 1
        cf.security_coef = 100
        cf.security_distance = 0.25
        sys.cost_function = cf
        cf_list = list((sys.cost_function.g, sys.cost_function.g_confort, sys.cost_function.g_security, sys.cost_function.g_override))
        
        print('dx max: ', sys.dmax)
        print('x dim: ', sys.x_grid)
        print('x0 boundaries: ' + str(sys.x_lb[0]) + ' to ' + str(str(sys.x_ub[0])))
        print('x1 boundaries: ' + str(sys.x_lb[1]) + ' to ' + str(str(sys.x_ub[1])))
        print('Esperance: ', sys.driver[0])
        
        ### GRID SYSTEM CONFIGURATION
        dt = 0.02 
        
        print('Number of states: ',sys.x_grid[0] * sys.x_grid[1])
        print('Number of actions: ', u_dim[0])
        print('Number of actions-states: ', sys.x_grid[0] * sys.x_grid[1] * u_dim[0])
        print('----------------------------------------------------------')
        
        temps_debut_grid_sys = time.time()
        grid_sys = discretizer2.GridDynamicSystem(sys, sys.x_grid, u_dim, dt, esperance = sys.driver[0], print_data=False)
        temps_fin_grid_sys = time.time()
        print('Grid Sys Computing Time: '+ str(temps_fin_grid_sys - temps_debut_grid_sys) + ' secondes' )
        
        ## SAVING METADATA
        name_list.append('Road: ' + roads[roads_to_test[r]] + '     Driver: ' + drivers[drivers_to_test[d]])
        grid_sys_list.append(grid_sys)
        
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
                
                
                c2g_vi = cost2go2.cost2go_list(grid_sys, sys, cl_sys_vi, cf_list)
                c2g_vi.compute_steps(print_iteration=False)
                
                ## SAVING VI
                vi_list.append(vi_controller)
                vi_cl_list.append(cl_sys_vi)
                vi_c2g_list.append(c2g_vi)
                
            ## TIME TO COLLISION
            if controler[controlers_to_test[c]] == 'Ttc':
                ttc_controller = BaselineController.TTCController(sys, grid_sys, slip_max=-slip_data[1])
                cl_sys_ttc = controller.ClosedLoopSystem( sys , ttc_controller )  
                cl_sys_ttc.cost_function = sys.cost_function
                c2g_ttc = cost2go2.cost2go_list(grid_sys, sys, cl_sys_ttc, cf_list)
                c2g_ttc.compute_steps(print_iteration=False)
                
                ## SAVING TTC
                ttc_list.append(ttc_controller)
                ttc_cl_list.append(cl_sys_ttc)
                ttc_c2g_list.append(c2g_ttc)
            
            ## MINIMAL STOPPING DISTANCE    
            if controler[controlers_to_test[c]] == 'Msd':
                msd_controller = BaselineController.MSDController(sys, grid_sys)
                cl_sys_msd = controller.ClosedLoopSystem( sys , msd_controller )  
                cl_sys_msd.cost_function = sys.cost_function
                c2g_msd = cost2go2.cost2go_list(grid_sys, sys, cl_sys_msd, cf_list)
                c2g_msd.compute_steps(print_iteration=False)
                
                ## SAVING MSD
                msd_list.append(msd_controller)
                msd_cl_list.append(cl_sys_msd)
                msd_c2g_list.append(c2g_msd)
                
            if controler[controlers_to_test[c]] == 'Human':
                human_controller = BaselineController.humanController(sys, grid_sys)
                cl_sys_human = controller.ClosedLoopSystem( sys , human_controller ) 
                cl_sys_human.cost_function = sys.cost_function
                c2g_human = cost2go2.cost2go_list(grid_sys, sys, cl_sys_human, cf_list)
                c2g_human.compute_steps(print_iteration=False)
                
                human_list.append(human_controller)
                human_cl_list.append(cl_sys_human)
                human_c2g_list.append(c2g_human)


#%% PLOT EVERYTHING
for n in range(len(name_list)):
    
    x0 = np.array([-10.0,4.5])
           
    controlers = list()
    cost_to_go = list()
    sim_list = list()
    
    for con in controlers_to_test:
        if controler[con] == 'Vi':  
            controlers.append(grid_sys_list[n].get_grid_from_array(grid_sys_list[n].get_input_from_policy(vi_list[n].pi, 0)).T)
            cost_to_go.append(vi_c2g_list[n])
            vi_cl_list[n].x0 = x0
            sim_vi = s.SimulatorV2(vi_cl_list[n], x0_end=sys.x_ub[0])
            sim_list.append(sim_vi.traj_to_args(sim_vi.traj))
            
        if controler[con] == 'Ttc':  
            controlers.append(ttc_list[n].c_array().T)
            cost_to_go.append(ttc_c2g_list[n])
            ttc_cl_list[n].x0 = x0
            sim_ttc = s.SimulatorV2(ttc_cl_list[n], x0_end=sys.x_ub[0])
            sim_list.append(sim_ttc.traj_to_args(sim_ttc.traj))
            
        if controler[con] == 'Msd':  
            controlers.append(msd_list[n].c_array().T)
            cost_to_go.append(msd_c2g_list[n])
            msd_cl_list[n].x0 = x0
            sim_msd = s.SimulatorV2(msd_cl_list[n], x0_end=sys.x_ub[0])
            sim_list.append(sim_msd.traj_to_args(sim_msd.traj))
            
        if controler[con] == 'Human':  
            controlers.append(human_list[n].c_array().T)
            cost_to_go.append(human_c2g_list[n])
            human_cl_list[n].x0 = x0
            sim_human = s.SimulatorV2(human_cl_list[n], x0_end=sys.x_ub[0])
            sim_list.append(sim_human.traj_to_args(sim_human.traj))
             
    for c2c in range(len(compare_mode)):
         fig, axs = plt.subplots(len(controlers_to_test), 5)
         plt.ion()
         fig.suptitle(str(name_list[n])+'     Comparaison: '+comparaison[compare_mode[c2c]] + ' Vi - controler (- = vi meilleur)')
         xname = sys.state_label[0] + ' ' + sys.state_units[0]
         yname = sys.state_label[1] + ' ' + sys.state_units[1] 
         if len(controlers_to_test) > 1:
             axs[0][0].set_title('Controler')
             axs[0][1].set_title('Cost to Go Total')
             axs[0][2].set_title('Cost to Go Confort: ' + str(cf.confort_coef))
             axs[0][3].set_title('Cost to Go Security: '+ str(cf.security_coef))
             axs[0][4].set_title('Cost to Go Override: '+ str(cf.override_coef))
         else:
             axs[0].set_title('Controler')
             axs[1].set_title('Cost to Go Total')
             axs[2].set_title('Cost to Go Confort: ' + str(cf.confort_coef))
             axs[3].set_title('Cost to Go Security: '+ str(cf.security_coef))
             axs[4].set_title('Cost to Go Override: '+ str(cf.override_coef))
 
         ### PLOTTING 
         if len(controlers_to_test) > 1:
             for c in range(len(controlers_to_test)):
                if comparaison[compare_mode[c2c]] == 'NoDiff':
                      controler_map = controlers[c]
                      controler_map[controler_map==0] = 0
                      cost_to_go_total = cost_to_go[c].cost2go_map_list[0].T
                      cost_to_go_confort = cost_to_go[c].cost2go_map_list[1].T
                      cost_to_go_security = cost_to_go[c].cost2go_map_list[2].T
                      cost_to_go_override = cost_to_go[c].cost2go_map_list[3].T
                   
                if comparaison[compare_mode[c2c]] == 'Normal':            
                     # controler_map = controlers[c]
                     if c == 0:
                          controler_map = controlers[c]
                          cost_to_go_total = cost_to_go[c].cost2go_map_list[0].T
                          cost_to_go_confort = cost_to_go[c].cost2go_map_list[1].T
                          cost_to_go_security = cost_to_go[c].cost2go_map_list[2].T
                          cost_to_go_override = cost_to_go[c].cost2go_map_list[3].T
                     else:
                          controler_map = controlers[0] - controlers[c]
                          cost_to_go_total = cost_to_go[0].cost2go_map_list[0].T - cost_to_go[c].cost2go_map_list[0].T
                          cost_to_go_confort = cost_to_go[0].cost2go_map_list[1].T - cost_to_go[c].cost2go_map_list[1].T
                          cost_to_go_security = cost_to_go[0].cost2go_map_list[2].T - cost_to_go[c].cost2go_map_list[2].T
                          cost_to_go_override = cost_to_go[0].cost2go_map_list[3].T - cost_to_go[c].cost2go_map_list[3].T

                if comparaison[compare_mode[c2c]] == 'Bool':
                     if c == 0:            
                          controler_map = controlers[c]
                          cost_to_go_total = cost_to_go[c].cost2go_map_list[0].T
                          cost_to_go_confort = cost_to_go[c].cost2go_map_list[1].T
                          cost_to_go_security = cost_to_go[c].cost2go_map_list[2].T
                          cost_to_go_override = cost_to_go[c].cost2go_map_list[3].T
                     else:
                          controler_map = controlers[0] - controlers[c]
                          controler_map[controler_map==0] = 0
                          controler_map[controler_map<0] = -1
                          controler_map[controler_map>0] = 1
                          cost_to_go_total = cost_to_go[0].cost2go_map_list[0].T - cost_to_go[c].cost2go_map_list[0].T
                          cost_to_go_confort = cost_to_go[0].cost2go_map_list[1].T - cost_to_go[c].cost2go_map_list[1].T
                          cost_to_go_security = cost_to_go[0].cost2go_map_list[2].T - cost_to_go[c].cost2go_map_list[2].T
                          cost_to_go_override = cost_to_go[0].cost2go_map_list[3].T - cost_to_go[c].cost2go_map_list[3].T
                          
                          controler_map[controler_map==0] = 0
                          controler_map[controler_map<0] = -1
                          controler_map[controler_map>0] = 1
                          
                          cost_to_go_total[cost_to_go_total==0] = 0
                          cost_to_go_total[cost_to_go_total<0] = -1
                          cost_to_go_total[cost_to_go_total>0] = 1
                          
                          cost_to_go_confort[cost_to_go_confort==0] = 0
                          cost_to_go_confort[cost_to_go_confort<0] = -1
                          cost_to_go_confort[cost_to_go_confort>0] = 1
                          
                          cost_to_go_security[cost_to_go_security==0] = 0
                          cost_to_go_security[cost_to_go_security<0] = -1
                          cost_to_go_security[cost_to_go_security>0] = 1
                          
                          cost_to_go_override[cost_to_go_override==0] = 0
                          cost_to_go_override[cost_to_go_override<0] = -1
                          cost_to_go_override[cost_to_go_override>0] = 1
                  
                
                plotting_colors = 'viridis'
                
                axs[c][0].set_ylabel(controler[controlers_to_test[c]]) 
                i = axs[c][0].pcolormesh(grid_sys_list[n].x_level[0], grid_sys_list[n].x_level[1], controler_map, shading='gouraud', cmap = plotting_colors)
                axs[c][0].axis([grid_sys_list[n].x_level[0][0], grid_sys_list[n].x_level[0][-1], grid_sys_list[n].x_level[1][0], grid_sys_list[n].x_level[1][-1]])
                fig.colorbar(i, ax=axs[c][0])
                
                axs[c][0].grid(True)
                 
                i = axs[c][1].pcolormesh(grid_sys_list[n].x_level[0], grid_sys_list[n].x_level[1], cost_to_go_total, shading='gouraud', cmap = plotting_colors)
                axs[c][1].axis([grid_sys_list[n].x_level[0][0], grid_sys_list[n].x_level[0][-1], grid_sys_list[n].x_level[1][0], grid_sys_list[n].x_level[1][-1]])
                fig.colorbar(i, ax=axs[c][1])
                axs[c][1].grid(True)
                axs[c][1].set_yticklabels([])
                
                
                i = axs[c][2].pcolormesh(grid_sys_list[n].x_level[0], grid_sys_list[n].x_level[1], cost_to_go_confort, shading='gouraud', cmap = plotting_colors)
                axs[c][2].axis([grid_sys_list[n].x_level[0][0], grid_sys_list[n].x_level[0][-1], grid_sys_list[n].x_level[1][0], grid_sys_list[n].x_level[1][-1]])
                fig.colorbar(i, ax=axs[c][2])
                axs[c][2].grid(True)
                axs[c][2].set_yticklabels([])
                 
                i = axs[c][3].pcolormesh(grid_sys_list[n].x_level[0], grid_sys_list[n].x_level[1], cost_to_go_security, shading='gouraud', cmap = plotting_colors)
                axs[c][3].axis([grid_sys_list[n].x_level[0][0], grid_sys_list[n].x_level[0][-1], grid_sys_list[n].x_level[1][0], grid_sys_list[n].x_level[1][-1]])
                fig.colorbar(i, ax=axs[c][3])
                axs[c][3].grid(True)
                axs[c][3].set_yticklabels([])
                 
                i = axs[c][4].pcolormesh(grid_sys_list[n].x_level[0], grid_sys_list[n].x_level[1], cost_to_go_override, shading='gouraud', cmap = plotting_colors)
                axs[c][4].axis([grid_sys_list[n].x_level[0][0], grid_sys_list[n].x_level[0][-1], grid_sys_list[n].x_level[1][0], grid_sys_list[n].x_level[1][-1]])
                fig.colorbar(i, ax=axs[c][4])
                axs[c][4].grid(True)
                axs[c][4].set_yticklabels([])

                if controler[controlers_to_test[c]] != 'Human': 
                     axs[c][0].plot(sim_list[c][:,2],sim_list[c][:,3])
                     axs[c][0].plot(sim_list[c][:,0],sim_list[c][:,1])
                      
                     axs[c][1].plot(sim_list[c][:,2],sim_list[c][:,3])
                     axs[c][1].plot(sim_list[c][:,0],sim_list[c][:,1])
                      
                     axs[c][2].plot(sim_list[c][:,2],sim_list[c][:,3])
                     axs[c][2].plot(sim_list[c][:,0],sim_list[c][:,1])
                      
                     axs[c][3].plot(sim_list[c][:,2],sim_list[c][:,3])
                     axs[c][3].plot(sim_list[c][:,0],sim_list[c][:,1])
                      
                     axs[c][4].plot(sim_list[c][:,2],sim_list[c][:,3])
                     axs[c][4].plot(sim_list[c][:,0],sim_list[c][:,1])
                     
                else:
                     for simu in sim_list:
                          axs[c][0].plot(simu[:,2],simu[:,3])
                          axs[c][0].plot(simu[:,0],simu[:,1])
                           
                          axs[c][1].plot(simu[:,2],simu[:,3])
                          axs[c][1].plot(simu[:,0],simu[:,1])
                           
                          axs[c][2].plot(simu[:,2],simu[:,3])
                          axs[c][2].plot(simu[:,0],simu[:,1])
                           
                          axs[c][3].plot(simu[:,2],simu[:,3])
                          axs[c][3].plot(simu[:,0],simu[:,1])
                           
                          axs[c][4].plot(simu[:,2],simu[:,3])
                          axs[c][4].plot(simu[:,0],simu[:,1])
                
                
         else: 
             for c in range(len(controlers_to_test)):
                 args = sim_list[c].traj_to_args(sim_list[c].traj)
                 
                 axs[0].set_ylabel(controler[c])
                 i = axs[0].pcolormesh(grid_sys_list[n].x_level[0], grid_sys_list[n].x_level[1], controlers[c], shading='gouraud')
                 axs[0].axis([grid_sys_list[n].x_level[0][0], grid_sys_list[n].x_level[0][-1], grid_sys_list[n].x_level[1][0], grid_sys_list[n].x_level[1][-1]])
                 fig.colorbar(i, ax=axs[0])
                 axs[0].grid(True)
                 
                 axs[1].set_ylabel(controler[c])
                 i = axs[1].pcolormesh(grid_sys_list[n].x_level[0], grid_sys_list[n].x_level[1], cost_to_go[c].cost2go_map_list[0].T, shading='gouraud')
                 axs[1].axis([grid_sys_list[n].x_level[0][0], grid_sys_list[n].x_level[0][-1], grid_sys_list[n].x_level[1][0], grid_sys_list[n].x_level[1][-1]])
                 fig.colorbar(i, ax=axs[1])
                 
                 axs[2].set_ylabel(controler[c])
                 i = axs[2].pcolormesh(grid_sys_list[n].x_level[0], grid_sys_list[n].x_level[1], cost_to_go[c].cost2go_map_list[1].T, shading='gouraud')
                 axs[2].axis([grid_sys_list[n].x_level[0][0], grid_sys_list[n].x_level[0][-1], grid_sys_list[n].x_level[1][0], grid_sys_list[n].x_level[1][-1]])
                 fig.colorbar(i, ax=axs[2])
                 
                 axs[3].set_ylabel(controler[c])
                 i = axs[3].pcolormesh(grid_sys_list[n].x_level[0], grid_sys_list[n].x_level[1], cost_to_go[c].cost2go_map_list[2].T, shading='gouraud')
                 axs[3].axis([grid_sys_list[n].x_level[0][0], grid_sys_list[n].x_level[0][-1], grid_sys_list[n].x_level[1][0], grid_sys_list[n].x_level[1][-1]])
                 fig.colorbar(i, ax=axs[3])
                 
                 axs[4].set_ylabel(controler[c])
                 i = axs[4].pcolormesh(grid_sys_list[n].x_level[0], grid_sys_list[n].x_level[1], cost_to_go[c].cost2go_map_list[3].T, shading='gouraud')
                 axs[4].axis([grid_sys_list[n].x_level[0][0], grid_sys_list[n].x_level[0][-1], grid_sys_list[n].x_level[1][0], grid_sys_list[n].x_level[1][-1]])
                 fig.colorbar(i, ax=axs[4])
             
                 axs[0].plot(args[:,2],args[:,3])
                 axs[0].plot(args[:,0],args[:,1])
                 
                 axs[1].plot(args[:,2],args[:,3])
                 axs[1].plot(args[:,0],args[:,1])
                 
                 axs[2].plot(args[:,2],args[:,3])
                 axs[2].plot(args[:,0],args[:,1])
                 
                 axs[3].plot(args[:,2],args[:,3])
                 axs[3].plot(args[:,0],args[:,1])
                 
                 axs[4].plot(args[:,2],args[:,3])
                 axs[4].plot(args[:,0],args[:,1])
        
        
        
        
        
        
        
        
        