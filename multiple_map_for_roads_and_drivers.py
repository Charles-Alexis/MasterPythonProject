#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 18:27:33 2023

@author: clearpath-robot
"""

###############################################################################
import numpy as np
import time
###############################################################################
import system
from datetime import date
import os
import costfunction
import discretizer2
import dynamic_programming as dprog
import BaselineController 
import simulationv2 as s
import controller
import copy 
import matplotlib.pyplot as plt

###############################################################################


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
sys.x_ub = np.array([5, 4.5])
sys.x_lb = np.array([-10., 0])
sys.u_ub = np.array([0.0, 1])
sys.u_lb = np.array([-0.3, 0])

# HUMAN DRIVER PARAMETERS
tm_roads = [2.2, 2.8, 3.4, 4.0, 4.2]
tm_dot_driver = [0.1, 0.4, 0.5, 0.7, 0.9]
sys.tf = 1.75
sys.tm_dot = 0.4    
sys.x_grid = [801,101]
test_name = 'Tmargin: '+str(sys.tm)+' Treaction: '+str(sys.tf)+' Tdmargin: '+str(sys.tm_dot)

#COSTFUNCTION
cf = costfunction.DriverModelCostFunction.from_sys(sys)
cf.confort_coef = 4
cf.override_coef = 4
cf.security_coef = 50
cf.xbar = np.array( [(sys.x_ub[0]-1), 0] ) # target
sys.cost_function = cf
#LIST
vi_list = list()
arg_list = list()
human_list = list()


x_dim = [801, 101]
u_dim = [20, 2]
dt = 0.01

directory = 'xmaxx_policymap_final/'+str(date.today().day)+'_'+str(date.today().month)+'_'+str(date.today().year)+'_'+str(time.localtime().tm_hour)+'h'+str(time.localtime().tm_min)
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
    f.write('Time Margin fro Roads: ' + str(tm_roads)+'\n')  
    f.write('Decceleration driver: ' + str(tm_dot_driver)+'\n')       
    f.write('pos_dim: '+str(x_dim[0])+'\n')      
    f.write('vit_dim: '+str(x_dim[1])+'\n')    
    f.write('slip: '+str(u_dim[0])+'\n')        
    f.write('dt: '+str(dt)+'\n')
os.chdir(current_directory)


### LOOP VI ###
time_debut = time.time()
for r in range(len(roads)):
     for d in range(len(tm_dot_driver)):
          road_nbr = r
          driver_nbr = d
          sys.tm = tm_roads[road_nbr]
          sys.tm_dot = tm_dot_driver[driver_nbr]
          road = roads[road_nbr]
          sys.road = sys.roads[road]
          grid_sys = discretizer2.GridDynamicSystem(sys, x_dim, u_dim, dt)
          #DYNAMICPROGRAMMING
          dp = dprog.DynamicProgrammingWithLookUpTable( grid_sys, cf)
          dp.compute_steps(1000,  treshhold=0.0005)
          vi_list.append(copy.copy(dp))
          vi_controller = dprog.LookUpTableController( grid_sys , dp.pi )
          vi_controller.k = 2
          
          cl_sys = controller.ClosedLoopSystem( sys , vi_controller ) 
          cl_sys.cost_function = sys.cost_function
          x0 = np.array([-10.,4.5])
          cl_sys.x0 = x0
          sim = s.SimulatorV2(cl_sys, x0_end=sys.x_ub[0])
          args = sim.traj_to_args(sim.traj)
          arg_list.append(args)
          
          human_list.append(sys.plot_human_model_ttc(plot=False))
          
          os.chdir(final_directory)
          dp.save_latest('xmaxx_' + road + '_' + str(tm_dot_driver[driver_nbr]) )
          os.chdir(current_directory)
          
          print("Finish " +roads[road_nbr])
          print("time since the begining of iterations: " +str(time.time()-time_debut))
     

fig, axs = plt.subplots(2, 5)
plt.ion()
fig.suptitle('Confort Coef: '+str(cf.confort_coef)+' Security Coef: ' + str(cf.security_coef) + ' Override Coef: '+str(cf.override_coef))
xname = sys.state_label[0] + ' ' + sys.state_units[0]
yname = sys.state_label[1] + ' ' + sys.state_units[1]  


for r in range(len(roads)):
     uk_0 = grid_sys.get_input_from_policy(vi_list[r].pi, 0)
     u0 = grid_sys.get_grid_from_array(uk_0)
     
     axs[0][r].set_title(roads[r])
     i1 = axs[0][r].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], u0.T, shading='gouraud')
     axs[0][r].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
     fig.colorbar(i1, ax=axs[0, r])
     axs[0][r].grid(True)
     
     axs[1][r].set_title('Human Model')
     i1 = axs[1][r].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], human_list[r].T, shading='gouraud')
     axs[1][r].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
     fig.colorbar(i1, ax=axs[1, r])
     axs[1][r].grid(True)
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     