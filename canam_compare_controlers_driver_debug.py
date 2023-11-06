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
import plotter
###############################################################################
from datetime import date
import datetime
import os
import warnings
warnings.filterwarnings('ignore')
###############################################################################
temps_debut_debut = time.time()

## POSSIBILITÉ DE TEST
roads = ['AsphalteDry','CementDry','AsphalteWet','CobblestoneDry','CobblestoneWet','Snow','Ice']
drivers = ['Good','Normal','Bad','Sleepy','Null']
comparaison = ['NoDiff', 'Normal','Bool','NearObstacle','HighSpeed']
controler = ['Vi','Ttc','Msd', 'Human']

## SCÉNARIO
roads_to_test = [0]  
drivers_to_test = [0,1,2]
coef_name = ['Confort','Override','Security']
coef_to_test = [[1, 3, 10]]
controlers_to_test = [0,1,2,3]
compare_mode = [0]
x_dim = [200, 200]
u_dim = [11,2]


roads_to_test_name = list()
for rtt in range(len(roads_to_test)): roads_to_test_name.append(roads[roads_to_test[rtt]])

drivers_to_test_name = list()
for rtt in range(len(drivers_to_test)): drivers_to_test_name.append(drivers[drivers_to_test[rtt]])

controlers_to_test_name = list()
for rtt in range(len(controlers_to_test)): controlers_to_test_name.append(controler[controlers_to_test[rtt]])

print('---------------------------------------')
print('Scénario')
print('Roads: ', roads_to_test)
print('Driver: ', drivers_to_test)
print('Controlers: ', controlers_to_test)
print('Coefs: ', coef_to_test)
print('---------------------------------------')

## CREATING LIST FOR DATA
name_list = list()
test_list = list()
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
plot_c2g_flag = True
sim_flag = False
iteration = 1
for coef in coef_to_test:
    for r in range(len(roads_to_test)):
        for d in range(len(drivers_to_test)): 
            print('---------------------------------------')
            #print('Iteration: ' + str(1 + r*len(drivers_to_test) + d))
            print('Iteration: ' + str(iteration))
            iteration = iteration+1
            print('Roads: ', roads[roads_to_test[r]])
            print('Driver: ', drivers[drivers_to_test[d]])
    
            ### SYSTEM CONFIGURATION
            sys = system.LongitudinalFrontWheelDriveCarWithDriverModel()
            sys.mass = 760
            sys.lenght = 3.35
            sys.xc = sys.lenght/2
            sys.yc = 1.74/2 
            sys.cdA = 0.3 * (1.84*1.74)
            sys.x_ub = np.array([0 , 20.0])
            sys.x_lb = [-80., 0]
            sys.u_ub = np.array([0.0, 1])
            sys.u_lb = np.array([-0.3, 0])
            sys.u_dim = u_dim
            sys.u_level = (np.arange(0,(sys.u_dim[0])) - (sys.u_dim[0]-1)) /  ((sys.u_dim[0]-1)/-sys.u_lb[0])
            sys.m = len(sys.u_dim)
            sys.obs_dist = sys.x_ub[0]
            sys.x_grid = x_dim
            sys.road = sys.roads[roads[roads_to_test[r]]]
            sys.timing = 0.5
            sys.driver = sys.drivers[drivers[drivers_to_test[d]]]
            sys.mu_coef = -sys.return_max_mu()[0]
            
            sys.roads_ind = roads_to_test[r]
            sys.drivers_ind = drivers_to_test[d] 
    
            slip_data = sys.return_max_mu()
            sys.dmax = sys.f([0,sys.x_ub[1]],[slip_data[1],1])
            sys.best_slip = sys.return_max_mu()[1]
            
            ### DRIVER MODEL CONFIGURATION
            sys.tm_dot = -0.4
            sys.tf = 1.75
            sys.tm_coef = 0.8
            
            sys.find_buuged_states()
            
            ### COST FUNCTION CONFIGURATION
            cf = costfunction.DriverModelCostFunction.from_sys(sys)
            cf.confort_coef = coef[0]
            cf.override_coef = coef[1]
            cf.security_coef = coef[2]
            cf.security_slope = 10
            cf.security_distance = 3
            sys.cost_function = cf

            ### COST FUNCTION CONFIGURATION FOR HUMAN MODEL
            cf_human = costfunction.DriverModelCostFunction_forhumanmodel.from_sys(sys)
            cf_human.confort_coef = coef[0]
            cf_human.override_coef = coef[1]
            cf_human.security_coef = coef[2]
            cf_human.security_slope = 10
            cf_human.security_distance = 3
            
            print('dx max: ', sys.dmax)
            print('x dim: ', sys.x_grid)
            print('x0 boundaries: ' + str(sys.x_lb[0]) + ' to ' + str(str(sys.x_ub[0])))
            print('x1 boundaries: ' + str(sys.x_lb[1]) + ' to ' + str(str(sys.x_ub[1])))
            print('Esperance: ', sys.driver[0])
            
            ### GRID SYSTEM CONFIGURATION
            dt = 0.1
            
            controlers_dim = sys.x_grid
            
            print('Number of states: ',sys.x_grid[0] * sys.x_grid[1])
            print('Number of actions: ', u_dim[0])
            print('Number of actions-states: ', sys.x_grid[0] * sys.x_grid[1] * u_dim[0])
            print('----------------------------------------------------------')
            
            temps_debut_grid_sys = time.time()
            for c2t in controlers_to_test:
                if c2t == 0:
                        grid_sys = discretizer2.GridDynamicSystem(sys, sys.x_grid, u_dim, dt, esperance = sys.driver[0], print_data=True)
                        grid_sys_list.append(grid_sys)
                        
            grid_sys_controller = discretizer2.GridDynamicSystem(sys, controlers_dim, u_dim, dt, esperance = sys.driver[0], print_data=False, lookup = False)
            grid_sys_controller_list.append(grid_sys_controller)
            grid_sys_controller.find_buuged_states()
            temps_fin_grid_sys = time.time()
            print('Grid Sys Computing Time: '+ str(temps_fin_grid_sys - temps_debut_grid_sys) + ' secondes' )
            
            ## SAVING METADATA
            #name_list.append('Road: ' + roads[roads_to_test[r]] + '     Driver: ' + drivers[drivers_to_test[d]]+ ' COEF= ' + str(coef) )
            name_list.append('Confort: ' + str(coef[0]) + ' Override: ' + str(coef[1]) + ' Sécurité: ' + str(coef[2]) )
            test_list.append([roads_to_test[r], drivers_to_test[d]])
            
            ### CONTROLER CONFIGURATION
            for c in range(len(controlers_to_test)): 
                ## VALUE ITERATION
                if controler[controlers_to_test[c]] == 'Vi':
                    sys.cost_function = cf
                    dp = dprog.DynamicProgrammingWithLookUpTable(grid_sys, cf, esperance = sys.driver[0], cf_flag = False)
                    dp.compute_steps(5000,  treshhold=0.05, animate_policy=False, animate_cost2go=False, jmax = 1000)
                    vi_controller = dprog.LookUpTableController( grid_sys , dp.pi )
                    vi_controller.k = 2
                    cl_sys_vi = controller.ClosedLoopSystem( sys , vi_controller ) 
                    cl_sys_vi.cost_function = sys.cost_function                
                    ## SAVING VI
                    vi_list.append(vi_controller)
                    vi_cl_list.append(cl_sys_vi)
                    
                    if compute_c2g_flag: 
                         c2g_vi = cost2go2.cost2go_list_2(grid_sys_controller, sys, cf, cl_sys_vi.controller.c)
                         vi_c2g_list.append(c2g_vi)
                    
                ## TIME TO COLLISION
                if controler[controlers_to_test[c]] == 'Ttc':
                    sys.cost_function = cf
                    ttc_controller = BaselineController.TTCController(sys, grid_sys_controller, security_distance=(3))
                    ttc_controller.constant_dec_flag = False
                    ttc_controller.constant_dec= -3.0
                    cl_sys_ttc = controller.ClosedLoopSystem( sys , ttc_controller )  
                    cl_sys_ttc.cost_function = sys.cost_function
                    ttc_list.append(ttc_controller)
                    ttc_cl_list.append(cl_sys_ttc)
                    if compute_c2g_flag: 
                         c2g_ttc = cost2go2.cost2go_list_2(grid_sys_controller, sys, cf, cl_sys_ttc.controller.c)
                         ttc_c2g_list.append(c2g_ttc)
                         
                ## MINIMAL STOPPING DISTANCE    
                if controler[controlers_to_test[c]] == 'Msd':
                    sys.cost_function = cf
                    msd_controller = BaselineController.MSDController(sys, grid_sys_controller, security_distance=(3))
                    cl_sys_msd = controller.ClosedLoopSystem(sys , msd_controller)  
                    cl_sys_msd.cost_function = sys.cost_function
                    msd_list.append(msd_controller)
                    msd_cl_list.append(cl_sys_msd)
                    if compute_c2g_flag: 
                         c2g_msd = cost2go2.cost2go_list_2(grid_sys_controller, sys, cf, cl_sys_msd.controller.c)
                         msd_c2g_list.append(c2g_msd)
                    
                    
                if controler[controlers_to_test[c]] == 'Human':
                    sys.cost_function = cf_human
                    human_controller = BaselineController.humanController(sys, grid_sys_controller)
                    cl_sys_human = controller.ClosedLoopSystem( sys , human_controller ) 
                    cl_sys_human.cost_function = sys.cost_function
                    human_list.append(human_controller)
                    human_cl_list.append(cl_sys_human)
                    if compute_c2g_flag: 
                         c2g_human = cost2go2.cost2go_list_2(grid_sys_controller, sys, cf_human, cl_sys_human.controller.c)
                         human_c2g_list.append(c2g_human)
                         

#%% PLOTTER TESTING
## CREATING LIST FOR DATA
metadata_list = [roads_to_test, drivers_to_test, controlers_to_test, coef_to_test, test_list, name_list, grid_sys_list, grid_sys_controller_list]
controler_list = [vi_list, ttc_list, msd_list, human_list]
closedloop_list = [vi_cl_list, ttc_cl_list, msd_cl_list, human_cl_list]
cost2go_list = [vi_c2g_list, ttc_c2g_list, msd_c2g_list, human_c2g_list]
pc = plotter.plotter(metadata_list, controler_list, closedloop_list, cost2go_list, plot_cost_to_go = False)

# pc.plot_cost_multiple_controler_and_road_param_diff([0,1,2,3,4,5,6])
# pc.plot_cost_multiple_controler_and_road_param()
# pc.plot_cost_multiple_controler_and_single_road_param(1)
# pc.plot_cost_multiple_controler_and_single_road_param_diff(1)
pc.plot_similarity(2,0,0)
pc.plot_similarity(2,1,0)
pc.plot_similarity(2,2,0)
#%%
# dp.animate_everything()

temps_fin_fin = time.time()


print('DID ALL THE SHIT IN: ' + str(datetime.timedelta(seconds=temps_fin_fin-temps_debut_debut)))