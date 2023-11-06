# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:39:54 2022

@author: Charles-Alexis

THIS CODE IS USE TO VIEW MULTIPLE POLICY MAP BASED ON A SINGLE COST FUNCTION

"""
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import os
import ast

###############################################################################
import discretizer2
import BaselineController
import system
import costfunction
import simulationv2 as s
###############################################################################
import controller
import dynamic_programming as dprog
import cost2go2 

class xmaxx_viewing:
    def __init__(self, map_directory_name = '/27_3_2023_9h54'):
        ## GETTING VALUES IN FOLDER
        self.current_dir = os.getcwd()
        self.map_directory_name = map_directory_name
        self.map_directory = self.current_dir + '/xmaxx_policymap_Esp' + self.map_directory_name
        with open(self.map_directory + '/cf_config.txt','r') as f:
             self.config_txt = f.readlines()
        
        
        ### ROAD AND DRIVER SETUP
        self.roads = ['AsphalteDry', 'AsphalteWet', 'CobblestoneWet', 'Snow']
        self.E_name = ['AsphalteDry', 'AsphalteWet', 'CobblestoneWet', 'Snow']
        
        ### SYSTEM SETUP
        self.sys = system.LongitudinalFrontWheelDriveCarWithDriverModel()
        self.sys.mass = 20
        self.sys.lenght = 0.48
        self.sys.xc = 0.24
        self.sys.yc = 0.15 
        self.sys.mass = 20
        self.sys.cdA = 0.3 * 0.105
        self.sys.x_ub = np.array([0 , 4.5])
        self.sys.x_lb = np.array([-15., 0])
        self.sys.u_ub = np.array([0.0, 1])
        self.sys.u_lb = np.array([-0.20, 0])
        self.sys.obs_dist = self.sys.x_ub[0]
        
        ### DRIVER SETUP
        self.tm_roads = [2.2, 2.5, 2.8, 3.8, 4.2]
        self.tm_dot_driver = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.timing_conservateur = -0.2
        self.timing_normal = +0.0
        self.timing_aggressif = +0.2
        self.timing_sleep = +10.0
        self.E_mauvais = [[self.timing_conservateur, 0.10], [self.timing_normal, 0.30], [self.timing_aggressif, 0.59], [self.timing_sleep, 0.01]]
        self.E_normal = [[self.timing_conservateur, 0.33], [self.timing_normal, 0.33], [self.timing_aggressif, 0.33], [self.timing_sleep, 0.01]]
        self.E_bon = [[self.timing_conservateur, 0.30], [self.timing_normal, 0.59], [self.timing_aggressif, 0.10], [self.timing_sleep, 0.01]]
        self.E_sleep = [[self.timing_conservateur, 0.01], [self.timing_normal, 0.01], [self.timing_aggressif, 0.01], [self.timing_sleep, 0.97]]
        self.E_null = [[+0.0, 1.0]]
        
        r=0
        road = self.roads[r]
        self.sys.road = self.sys.roads[road]
        self.sys.tm = self.tm_roads[r]
        self.sys.tf = 1.75
        self.sys.tm_dot = self.tm_dot_driver[r]
        self.sys.tm_dot = 0.5
        self.sys.x_grid = [550,150]
        self.slip_data = self.sys.return_max_mu()
        self.sys.dmax = self.sys.f([0,self.sys.x_ub[1]],[-self.slip_data[1],1])
        self.sys.use_human_model = True
        
        #COSTFUNCTION
        self.cf = costfunction.DriverModelCostFunction.from_sys(self.sys)
        self.cf.confort_coef = 5
        self.cf.override_coef = 2000
        self.cf.security_coef = 25
        self.cf.xbar = np.array( [(self.sys.x_ub[0]-1), 0] ) # target
        self.sys.cost_function = self.cf
        
        
        self.roads_array_flag = True
        self.drivers_array_flag = True
        
        self.E_arr = [self.E_null, self.E_mauvais, self.E_normal, self.E_bon, self.E_sleep] 
        self.E_arr_name = ['E_null', 'E_mauvais', 'E_normal', 'E_bon', 'E_sleep']
        
        
        self.dt = 0.02
        #GRIDSYS
        self.grid_sys = discretizer2.GridDynamicSystem(self.sys, self.sys.x_grid, [10, 2], self.dt, esperance = self.E_arr[0], lookup=False)
        
        self.dp_list = list()
        self.name_list = list()
        self.load_data()
        self.plot_road(0)
        self.plot_road(1)
        self.plot_road(2)
        self.plot_road(3)
        
    def load_data(self):
        print('Loading Data')
        for d in self.E_arr_name:
            for r in self.roads:
                name = 'xmaxx_'+r + '_' + str(d)
                # print(name)
                
                self.name_list.append(name)
                self.sys.road = self.sys.roads[r]
                
                dp = dprog.DynamicProgrammingWithLookUpTable(self.grid_sys, self.cf, compute_cost = False)
                dp.load_J_next(self.map_directory+'/'+name)
                self.dp_list.append(dp)
                    
    def plot_road(self, road_ind):
        print('PLOTTING U FOR ROAD: ', self.roads[road_ind])
        
        fig, axs = plt.subplots(1, len(self.E_arr))
        plt.ion()
        fig.suptitle('Confort Coef: '+str(self.cf.confort_coef)+' Security Coef: ' + str(self.cf.security_coef) + ' Override Coef: '+str(self.cf.override_coef))
        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]  
        
        for i in range(len(self.E_arr)):
            index = road_ind + i*int(len(self.name_list)/5)
            u0 = self.grid_sys.get_grid_from_array(self.grid_sys.get_input_from_policy(self.dp_list[index].pi, 0))
            
            axs[i].set_title(self.name_list[index])
            i_p = axs[i].pcolormesh(self.grid_sys.x_level[0], self.grid_sys.x_level[1], u0.T, shading='gouraud')
            axs[i].axis([self.grid_sys.x_level[0][0], self.grid_sys.x_level[0][-1], self.grid_sys.x_level[1][0], self.grid_sys.x_level[1][-1]])
            axs[i].grid(True)

        
            
             
if __name__ == '__main__':                                               
     v = xmaxx_viewing()  