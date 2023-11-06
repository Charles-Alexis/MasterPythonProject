# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:39:54 2022

@author: Charles-Alexis

THIS CODE IS USE TO CREATE MULTIPLE POLICY MAP BASED ON A SINGLE COST FUNCTION

"""
###############################################################################
import numpy as np
import time
from datetime import date
import os
###############################################################################
import system
import costfunction
import discretizer2
import dynamic_programming as dprog

import matplotlib.pyplot as plt

###############################################################################
temps_debut = time.time()

#%%
sys = system.LongitudinalFrontWheelDriveCarWithDriverModel()
sys.lenght = 0.48
sys.xc = 0.24
sys.yc = 0.15
sys.mass = 20
sys.cdA = 0.3 * 0.105

sys.x_ub = np.array([0, 4.5])
sys.x_lb = np.array([-10., 0])
sys.u_ub = np.array([0.0, 1])
sys.u_lb = np.array([-0.3, 0])



slip_data = sys.return_max_mu()
dx = sys.f([0,sys.x_ub[1]],[-slip_data[1],1])
time_dist = ((-1*sys.x_ub[1])-(-1*sys.x_lb[1]))/dx[1]
displacement = (0.5*((-1*sys.x_ub[1])+(-1*sys.x_lb[1]))*time_dist)
sys.driver_xmaxx_fort = {
'5': [displacement + np.abs(displacement*0.4), displacement + np.abs(displacement*0.4) + 1.,'5'],
'4': [displacement + np.abs(displacement*0.3), displacement + np.abs(displacement*0.3) + 1.,'4'],
'3': [displacement + np.abs(displacement*0.2), displacement + np.abs(displacement*0.2) + 1.,'3'],
'2': [displacement + np.abs(displacement*0.1), displacement + np.abs(displacement*0.1) + 1.,'2'],
'1': [displacement + np.abs(displacement*0.0), displacement + np.abs(displacement*0.0) + 1.,'1'],
}

sys.driver = sys.driver_xmaxx_fort['1']
sys.use_human_model = False
#COSTFUNCTION
cf = costfunction.DriverModelCostFunction.from_sys(sys)
cf.confort_coef = 10
cf.override_coef = 0
cf.security_coef = 500
cf.xbar = np.array( [(sys.x_ub[0]-1), 0] ) # target
sys.cost_function = cf


x_dim = [201,401]
u_dim = [20,2]
dt = 0.02
#%%

directory = 'xmaxx_policymap/'+str(date.today().day)+'_'+str(date.today().month)+'_'+str(date.today().year)+'_'+str(time.localtime().tm_hour)+'h'+str(time.localtime().tm_min)
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
    f.write( str(sys.driver_xmaxx_fort)+'\n')     
    f.write('pos_dim: '+str(x_dim[0])+'\n')      
    f.write('vit_dim: '+str(x_dim[1])+'\n')    
    f.write('slip: '+str(u_dim[0])+'\n')        
    f.write('dt: '+str(dt)+'\n')
os.chdir(current_directory)
            
nbr_of_iter = len(sys.roads) * len(sys.driver_xmaxx_fort)
iteration = 0
temps_debut = time.time()
driver = '1'
for road in sys.roads:
    sys.road = sys.roads[road]    

    grid_sys = discretizer2.GridDynamicSystem(sys, x_dim, u_dim, dt)
           
    dp = dprog.DynamicProgrammingWithLookUpTable( grid_sys, cf)
    dp.compute_steps(1000,  treshhold=0.0001)

    os.chdir(final_directory)
    dp.save_latest('xmaxx_' + road + '_' + driver )
    os.chdir(current_directory)
  
    iteration = iteration + 1
    print(str(iteration) + '/' + str(nbr_of_iter) + ' done in ' + str((time.time()-temps_debut)))



