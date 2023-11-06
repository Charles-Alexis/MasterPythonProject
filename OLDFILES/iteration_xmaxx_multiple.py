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
import valueiteration
import discretizer
import system
import costfunction

###############################################################################

## SYSTEM IS BASED ON XMAXX DATA
sys = system.LongitudinalFrontWheelDriveCarWithDriverModel()
sys.lenght = 0.6
sys.xc = 0.3
sys.yc = 0.175
sys.mass = 25
sys.cdA = 0.3 * 0.105
sys.x_ub = np.array([+20.0, 6.0])
sys.x_lb = np.array([0, 0])
sys.u_ub = np.array([0.0, 1])
sys.u_lb = np.array([-0.3, 0])
grid_sys = discretizer.GridDynamicSystem(sys, (111, 111), (5, 2), 0.1)
cf = costfunction.DriverModelCostFunction.from_sys(sys)
cf.confort_coef = 1
cf.override_coef = 1
cf.security_coef = 25
cf.xbar = np.array( [0, 0] ) # target
sys.cost_function = cf
sys.driver = sys.driver_xmaxx_fort['Good']

#%%

directory = 'xmaxx_policymap/'+str(date.today().day)+'_'+str(date.today().month)+'_'+str(date.today().year)+'_'+str(time.localtime().tm_hour)+'h'+str(time.localtime().tm_min) +'_'+ str(cf.confort_coef) +'_'+ str(cf.override_coef) +'_'+ str(cf.security_coef)
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)
   
temps_debut = time.time()

os.chdir(final_directory)
with open('cf_config.txt', 'w') as f:
    f.write('Confort Coefficient: ' + str(cf.confort_coef)+'\n')  
    f.write('Override Coefficient: ' + str(cf.override_coef)+'\n')  
    f.write('Security Coefficient: ' + str(cf.security_coef)+'\n') 
    f.write(str(sys.roads)+'\n')   
    f.write( str(sys.driver_xmaxx_fort)+'\n')     
os.chdir(current_directory)
            
nbr_of_iter = len(sys.roads) * len(sys.driver_xmaxx_fort)
iteration = 0
temps_debut = time.time()
for road in sys.roads:
    sys.road = sys.roads[road]
    for driver in sys.driver_xmaxx_fort:
            sys.driver = sys.driver_xmaxx_fort[driver]
            vi = valueiteration.ValueIteration_ND(grid_sys, cf)
            vi.threshold = 0.1
            vi.uselookuptable = False
            vi.initialize()
            vi.compute_steps(500,False)
            
            os.chdir(final_directory)
            vi.save_data('xmaxx_' + road + '_' + driver )
            os.chdir(current_directory)
            
            iteration = iteration + 1
            print(str(iteration) + '/' + str(nbr_of_iter) + ' done in ' + str((time.time()-temps_debut)))



