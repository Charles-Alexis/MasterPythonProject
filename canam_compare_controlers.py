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
#%%
###############################################################################
temps_debut_debut = time.time()

## POSSIBILITÉ DE TEST
roads = ['AsphalteDry','CementDry','AsphalteWet','CobblestoneDry','CobblestoneWet','Snow','Ice']
drivers = ['Good','Normal','Bad','Sleepy','Null']
comparaison = ['NoDiff', 'Normal','Bool','NearObstacle','HighSpeed']
controler = ['Vi','Ttc','Msd', 'Human']

## SCÉNARIO
roads_to_test = [2]  
drivers_to_test = [4]
coef_name = ['Confort','Override','Security' ]
coef_to_test = [[1, 10, 100]]
controlers_to_test = [0,1,2,3]
compare_mode = [0]
x_dim = [150, 150]
u_dim = [21,2]


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
plot_c2g_flag = False
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
            sys.dmax = sys.f([-80,sys.x_ub[1]],[slip_data[1],1])
            sys.best_slip = sys.return_max_mu()[1]

            ### DRIVER MODEL CONFIGURATION
            sys.tm_dot = -0.40
            sys.tf = 1.75
            sys.tm_coef = 0.6
            
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
            print('mu/slip', sys.return_max_mu())
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
                    dp.compute_steps(1000,  treshhold=0.00001, animate_policy=False, animate_cost2go=False, jmax = 1000)
                    vi_controller = dprog.LookUpTableController( grid_sys , dp.pi )
                    vi_controller.k = 2
                    cl_sys_vi = controller.ClosedLoopSystem( sys , vi_controller ) 
                    cl_sys_vi.cost_function = sys.cost_function                
                    ## SAVING VI
                    vi_list.append(vi_controller)
                    vi_cl_list.append(cl_sys_vi)
                    
                    if compute_c2g_flag: 
                         print('CALCULATING COST 2 GO VI')
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
                         print('CALCULATING COST 2 GO TTC')
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
                         print('CALCULATING COST 2 GO MSD')
                         c2g_msd = cost2go2.cost2go_list_2(grid_sys_controller, sys, cf, cl_sys_msd.controller.c)
                         msd_c2g_list.append(c2g_msd)
                    
                    
                if controler[controlers_to_test[c]] == 'Human':
                    sys.cost_function = cf
                    human_controller = BaselineController.humanController(sys, grid_sys_controller)
                    cl_sys_human = controller.ClosedLoopSystem( sys , human_controller ) 
                    cl_sys_human.cost_function = sys.cost_function
                    human_list.append(human_controller)
                    human_cl_list.append(cl_sys_human)
                    if compute_c2g_flag: 
                         print('CALCULATING COST 2 GO HUMAN')
                         c2g_human = cost2go2.cost2go_list_2(grid_sys_controller, sys, cf_human, cl_sys_human.controller.c)
                         human_c2g_list.append(c2g_human)

#%% SIMULATION
sim_vi = s.SimulatorSeminaire(cl_sys_vi, cf)
sim_vi.plot_data()   

sim_ttc = s.SimulatorSeminaire(cl_sys_ttc, cf)
sim_ttc.plot_data()   

sim_msd = s.SimulatorSeminaire(cl_sys_msd, cf)
sim_msd.plot_data()   

sim_human = s.SimulatorSeminaire(cl_sys_human, cf)
sim_human.plot_data()                    

#%% PLOTTER TESTING
## CREATING LIST FOR DATA
metadata_list = [roads_to_test, drivers_to_test, controlers_to_test, coef_to_test, test_list, name_list, grid_sys_list, grid_sys_controller_list]
controler_list = [vi_list, ttc_list, msd_list, human_list]
closedloop_list = [vi_cl_list, ttc_cl_list, msd_cl_list, human_cl_list]
cost2go_list = [vi_c2g_list, ttc_c2g_list, msd_c2g_list, human_c2g_list]
pc = plotter.plotter(metadata_list, controler_list, closedloop_list, cost2go_list, plot_cost_to_go = False)
# pc.plotting()

# if len(roads_to_test) > 1:
#     pc.save_everything_roads()
    
# if len(drivers_to_test) > 1:
#     pc.save_everything_driver(roads_to_test[0])   
#%%
# dp.animate_everything()

temps_fin_fin = time.time()


print('DID ALL THE SHIT IN: ' + str(datetime.timedelta(seconds=temps_fin_fin-temps_debut_debut)))

#%% COMPUTING ADPTATION ROAD

sys_test = system.LongitudinalFrontWheelDriveCarWithDriverModel()
sys_test.mass = 760
sys_test.lenght = 3.35
sys_test.xc = sys_test.lenght/2
sys_test.yc = 1.74/2 
sys_test.cdA = 0.3 * (1.84*1.74)
sys_test.x_ub = np.array([0 , 20.0])
sys_test.x_lb = [-80., 0]
sys_test.u_ub = np.array([0.0, 1])
sys_test.u_lb = np.array([-0.3, 0])
sys_test.u_dim = u_dim
sys_test.u_level = (np.arange(0,(sys_test.u_dim[0])) - (sys_test.u_dim[0]-1)) /  ((sys_test.u_dim[0]-1)/-sys_test.u_lb[0])
sys_test.m = len(sys_test.u_dim)
sys_test.obs_dist = sys_test.x_ub[0]
sys_test.x_grid = x_dim
sys_test.road = sys_test.roads['Snow']
sys_test.timing = 0.5
sys_test.driver = sys_test.drivers['Null']
sys_test.mu_coef = -sys_test.return_max_mu()[0]

sys_test.roads_ind = 5
sys_test.drivers_ind = 4

slip_data = sys_test.return_max_mu()
sys_test.dmax = sys_test.f([-80,sys_test.x_ub[1]],[slip_data[1],1])
sys_test.best_slip = sys_test.return_max_mu()[1]

### DRIVER MODEL CONFIGURATION
sys_test.tm_dot = -0.40
sys_test.tf = 1.75
sys_test.tm_coef = 0.6

sys_test.find_buuged_states()

### COST FUNCTION CONFIGURATION
cf_test = costfunction.DriverModelCostFunction.from_sys(sys_test)
cf_test.confort_coef = coef[0]
cf_test.override_coef = coef[1]
cf_test.security_coef = coef[2]
cf_test.security_slope = 10
cf_test.security_distance = 3
sys_test.cost_function = cf_test

### COST FUNCTION CONFIGURATION FOR HUMAN MODEL
cf_human_test = costfunction.DriverModelCostFunction_forhumanmodel.from_sys(sys_test)
cf_human_test.confort_coef = coef[0]
cf_human_test.override_coef = coef[1]
cf_human_test.security_coef = coef[2]
cf_human_test.security_slope = 10
cf_human_test.security_distance = 3

grid_sys_controller_test = grid_sys_controller_list[4]
grid_sys_list_test = grid_sys_list[4]

cl_sys_vi_test = vi_cl_list[2]
cl_sys_msd_test = msd_cl_list[2]
cl_sys_ttc_test = ttc_cl_list[2]

c2g_vi_test = cost2go2.cost2go_list_2(grid_sys_controller_test, sys_test, cf_test, cl_sys_vi_test.controller.c)
c2g_msd_test = cost2go2.cost2go_list_2(grid_sys_controller_test, sys_test, cf_test, cl_sys_msd_test.controller.c)
c2g_ttc_test = cost2go2.cost2go_list_2(grid_sys_controller_test, sys_test, cf_test, cl_sys_ttc_test.controller.c)

# fig, axs = plt.subplots(2,2, figsize=(8,8))
# fig.suptitle('VI')
# grid = grid_sys_controller_test
# pc.plot_function(fig,axs[0][0],grid,c2g_vi_test.J_total.T, name = 'Coûts à venir totaux')
# pc.plot_function(fig,axs[0][1],grid,c2g_vi_test.J_security.T, name = 'Coûts à venir pour la sécurité')
# pc.plot_function(fig,axs[1][0],grid,c2g_vi_test.J_confort.T, name = 'Coûts à venir pour le confort')
# pc.plot_function(fig,axs[1][1],grid,c2g_vi_test.J_override.T, name = 'Coûts à venir pour la liberté')
# plt.tight_layout() 

# fig, axs = plt.subplots(2,2, figsize=(8,8))
# fig.suptitle('MSD')
# grid = grid_sys_controller_test
# pc.plot_function(fig,axs[0][0],grid,c2g_msd_test.J_total.T, name = 'Coûts à venir totaux')
# pc.plot_function(fig,axs[0][1],grid,c2g_msd_test.J_security.T, name = 'Coûts à venir pour la sécurité')
# pc.plot_function(fig,axs[1][0],grid,c2g_msd_test.J_confort.T, name = 'Coûts à venir pour le confort')
# pc.plot_function(fig,axs[1][1],grid,c2g_msd_test.J_override.T, name = 'Coûts à venir pour la liberté')
# plt.tight_layout() 

# fig, axs = plt.subplots(2,2, figsize=(8,8))
# fig.suptitle('TTC')
# grid = grid_sys_controller_test
# pc.plot_function(fig,axs[0][0],grid,c2g_ttc_test.J_total.T, name = 'Coûts à venir totaux')
# pc.plot_function(fig,axs[0][1],grid,c2g_ttc_test.J_security.T, name = 'Coûts à venir pour la sécurité')
# pc.plot_function(fig,axs[1][0],grid,c2g_ttc_test.J_confort.T, name = 'Coûts à venir pour le confort')
# pc.plot_function(fig,axs[1][1],grid,c2g_ttc_test.J_override.T, name = 'Coûts à venir pour la liberté')
# plt.tight_layout() 

fig, axs = plt.subplots(2,2, figsize=(8,8))
fig.suptitle('Route utilisée pour la simulation: Neige \nLoi de commandes pour la route: Asphalte Humide - Loi de commandes pour la route: Neige')
grid = grid_sys_controller_test
pc.plot_function(fig,axs[0][0],grid, (c2g_vi_test.J_total - pc.cost_array[5][4][0]).T, name = 'Coûts à venir totaux')
pc.plot_function(fig,axs[0][1],grid, (c2g_vi_test.J_security - pc.cost_array_security[5][4][0]).T, name = 'Coûts à venir pour la sécurité')
pc.plot_function(fig,axs[1][0],grid, (c2g_vi_test.J_confort - pc.cost_array_confort[5][4][0]).T, name = 'Coûts à venir pour le confort')
pc.plot_function(fig,axs[1][1],grid, (c2g_vi_test.J_override - pc.cost_array_override[5][4][0]).T, name = 'Coûts à venir pour la liberté')
plt.tight_layout() 

fig, axs = plt.subplots(2,2, figsize=(8,8))
fig.suptitle('Route utilisée pour la simulation: Neige \nLoi de commandes pour la route: Asphalte Humide - Loi de commandes pour la route: Neige')
grid = grid_sys_controller_test
pc.plot_function(fig,axs[0][0],grid, (pc.cost_array[5][4][0]).T, name = 'Coûts à venir totaux')
pc.plot_function(fig,axs[0][1],grid, (pc.cost_array_security[5][4][0]).T, name = 'Coûts à venir pour la sécurité')
pc.plot_function(fig,axs[1][0],grid, (pc.cost_array_confort[5][4][0]).T, name = 'Coûts à venir pour le confort')
pc.plot_function(fig,axs[1][1],grid, (pc.cost_array_override[5][4][0]).T, name = 'Coûts à venir pour la liberté')
plt.tight_layout() 


# fig, axs = plt.subplots(2,2, figsize=(8,8))
# fig.suptitle('MSD--')
# grid = grid_sys_controller_test
# pc.plot_function(fig,axs[0][0],grid, (c2g_msd_test.J_total - pc.cost_array[5][4][2]).T, name = 'Coûts à venir totaux')
# pc.plot_function(fig,axs[0][1],grid, (c2g_msd_test.J_security - pc.cost_array_security[5][4][2]).T, name = 'Coûts à venir pour la sécurité')
# pc.plot_function(fig,axs[1][0],grid, (c2g_msd_test.J_confort - pc.cost_array_confort[5][4][2]).T, name = 'Coûts à venir pour le confort')
# pc.plot_function(fig,axs[1][1],grid, (c2g_msd_test.J_override - pc.cost_array_override[5][4][2]).T, name = 'Coûts à venir pour la liberté')
# plt.tight_layout() 


# fig, axs = plt.subplots(2,2, figsize=(8,8))
# fig.suptitle('TTC--')
# grid = grid_sys_controller_test
# pc.plot_function(fig,axs[0][0],grid, (c2g_ttc_test.J_total - pc.cost_array[5][4][1]).T, name = 'Coûts à venir totaux')
# pc.plot_function(fig,axs[0][1],grid, (c2g_ttc_test.J_security - pc.cost_array_security[5][4][1]).T, name = 'Coûts à venir pour la sécurité')
# pc.plot_function(fig,axs[1][0],grid, (c2g_ttc_test.J_confort - pc.cost_array_confort[5][4][1]).T, name = 'Coûts à venir pour le confort')
# pc.plot_function(fig,axs[1][1],grid, (c2g_ttc_test.J_override - pc.cost_array_override[5][4][1]).T, name = 'Coûts à venir pour la liberté')
# plt.tight_layout() 

#%% COMPUTING ADPTATION DRIVER
sys_test = system.LongitudinalFrontWheelDriveCarWithDriverModel()
sys_test.mass = 760
sys_test.lenght = 3.35
sys_test.xc = sys_test.lenght/2
sys_test.yc = 1.74/2 
sys_test.cdA = 0.3 * (1.84*1.74)
sys_test.x_ub = np.array([0 , 20.0])
sys_test.x_lb = [-80., 0]
sys_test.u_ub = np.array([0.0, 1])
sys_test.u_lb = np.array([-0.3, 0])
sys_test.u_dim = u_dim
sys_test.u_level = (np.arange(0,(sys_test.u_dim[0])) - (sys_test.u_dim[0]-1)) /  ((sys_test.u_dim[0]-1)/-sys_test.u_lb[0])
sys_test.m = len(sys_test.u_dim)
sys_test.obs_dist = sys_test.x_ub[0]
sys_test.x_grid = x_dim
sys_test.road = sys_test.roads['AsphalteWet']
sys_test.timing = 0.5
sys_test.driver = sys_test.drivers['Good']
sys_test.mu_coef = -sys_test.return_max_mu()[0]

sys_test.roads_ind = 2
sys_test.drivers_ind = 0

slip_data = sys_test.return_max_mu()
sys_test.dmax = sys_test.f([-80,sys_test.x_ub[1]],[slip_data[1],1])
sys_test.best_slip = sys_test.return_max_mu()[1]

### DRIVER MODEL CONFIGURATION
sys_test.tm_dot = -0.40
sys_test.tf = 1.75
sys_test.tm_coef = 0.6

sys_test.find_buuged_states()

### COST FUNCTION CONFIGURATION
cf_test = costfunction.DriverModelCostFunction.from_sys(sys_test)
cf_test.confort_coef = coef[0]
cf_test.override_coef = coef[1]
cf_test.security_coef = coef[2]
cf_test.security_slope = 10
cf_test.security_distance = 3
sys_test.cost_function = cf_test

### COST FUNCTION CONFIGURATION FOR HUMAN MODEL
cf_human_test = costfunction.DriverModelCostFunction_forhumanmodel.from_sys(sys_test)
cf_human_test.confort_coef = coef[0]
cf_human_test.override_coef = coef[1]
cf_human_test.security_coef = coef[2]
cf_human_test.security_slope = 10
cf_human_test.security_distance = 3

grid_sys_controller_test = grid_sys_controller_list[0]
grid_sys_list_test = grid_sys_list[0]

cl_sys_vi_test = vi_cl_list[4]
cl_sys_msd_test = msd_cl_list[4]
cl_sys_ttc_test = ttc_cl_list[4]

c2g_vi_test = cost2go2.cost2go_list_2(grid_sys_controller_test, sys_test, cf_test, cl_sys_vi_test.controller.c)
# c2g_msd_test = cost2go2.cost2go_list_2(grid_sys_controller_test, sys_test, cf_test, cl_sys_msd_test.controller.c)
# c2g_ttc_test = cost2go2.cost2go_list_2(grid_sys_controller_test, sys_test, cf_test, cl_sys_ttc_test.controller.c)

fig, axs = plt.subplots(2,2, figsize=(8,8))
fig.suptitle('Route utilisée pour la simulation: Asphalte Mouillé \n Conducteur de simulation: Expérimenté \nLoi de commandes pour le conducteur: sans espérence - Loi de commandes pour le conducteur: expérimenté')
grid = grid_sys_controller_test
pc.plot_function(fig,axs[0][0],grid, (c2g_vi_test.J_total - pc.cost_array[sys_test.roads_ind][sys_test.drivers_ind][0]).T, name = 'Coûts à venir totaux')
pc.plot_function(fig,axs[0][1],grid, (c2g_vi_test.J_security - pc.cost_array_security[sys_test.roads_ind][sys_test.drivers_ind][0]).T, name = 'Coûts à venir pour la sécurité')
pc.plot_function(fig,axs[1][0],grid, (c2g_vi_test.J_confort - pc.cost_array_confort[sys_test.roads_ind][sys_test.drivers_ind][0]).T, name = 'Coûts à venir pour le confort')
pc.plot_function(fig,axs[1][1],grid, (c2g_vi_test.J_override - pc.cost_array_override[sys_test.roads_ind][sys_test.drivers_ind][0]).T, name = 'Coûts à venir pour la liberté')
plt.tight_layout() 


















