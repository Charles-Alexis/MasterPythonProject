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
    def __init__(self, map_directory_name = '/5_3_2023_23h35'):
        ## GETTING VALUES IN FOLDER
        self.current_dir = os.getcwd()
        self.map_directory_name = map_directory_name
        self.map_directory = self.current_dir + '/xmaxx_policymap_final' + self.map_directory_name
        with open(self.map_directory + '/cf_config.txt','r') as f:
             self.config_txt = f.readlines()

        ## SETUP SYS
        self.sys = system.LongitudinalFrontWheelDriveCarWithDriverModel()
        self.sys.roads =  ast.literal_eval(self.config_txt[3][0:-1])
        self.roads_array = list()
        for r in self.sys.roads: self.roads_array.append(r)
        
        ## SETUP DRIVER DISTANCE
        self.tm_array = np.zeros([5])
        self.tmd_array = np.zeros([5])
        self.setup_sys()

        ## SETUP COSTFUNCTION        
        self.cf = costfunction.DriverModelCostFunction.from_sys(self.sys)          
        self.setup_cf()
        self.sys.cost_function = self.cf
        
        ## SETUP LIST AND GRIDSYS 
        self.nom_control = list()
        self.vi_u = list()
        self.grid_syss = list()
        self.setup_gridsys()

#        self.plot_policy(self.tmd_array[2], 'AsphalteDry', [-10,4.5])
        self.plot_policies()

#        ## SETUP TTC 
#        self.ttc_security_factor = 1/1
#        self.ttc_distance = np.array([0.708,  1.187, 2.88, 5.952, 21.739]) * self.ttc_security_factor
#        self.ttc_slip = np.array([-0.143, -0.112, -0.116, -0.052, -0.022])        
#        
#        ## HUMAN MODEL AND COST2GO
#        self.human_controler = BaselineController.humanController(self.sys, self.grid_sys, self.sys.human_model_exp)
#        self.cl_sys_human = controller.ClosedLoopSystem(self.sys , self.human_controler)
#        self.cl_sys_human.cost_function = self.sys.cost_function
#        self.c_human = cost2go2.cost2go_list(self.grid_sys, self.sys, self.cl_sys_human, self.cf_list)
        
    def plot_roads(self, road_name):
       pass
  
    def plot_driver(self, driv_nbr):
       pass
                       
    def plot_policy(self, driv, road, x0):
       name = road+'_'+str(driv)
       for index_temp in range(len(self.nom_control)):
          if name == self.nom_control[index_temp]:
              index = index_temp
       ## FOR FIGURE
       fig, axs = plt.subplots(2, 3)
       plt.ion()
       fig.suptitle('Optimale baseline for Driver: ' + str(driv) + ' On road: ' + road)
       xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
       yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]
       
       uk_0 = self.grid_syss[index].get_input_from_policy(self.vi_u[index].pi, 0)
       u0 = self.grid_syss[index].get_grid_from_array(uk_0)
       
       self.plot_to_figure(fig, axs[0][0], u0, xname, yname, index,'VI')
       
    def plot_policies(self):
        nbr_driv = len(self.sys.tm_dot)
        nbr_road = len(self.sys.roads)
        
        fig, axs = plt.subplots(nbr_road, nbr_driv)
        plt.ion()
        fig.suptitle('Optimale commands')
        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]
       
        i = 0
        j = 0
        for driver in self.sys.tm_dot:
            for road in self.sys.roads:
                name = road+'_'+str(driver)
                for index_temp in range(len(self.nom_control)):
                    if name == self.nom_control[index_temp]:
                        index = index_temp
                        
                uk_0 = self.grid_syss[index].get_input_from_policy(self.vi_u[index].pi, 0)
                u0 = self.grid_syss[index].get_grid_from_array(uk_0)
                
                self.plot_to_figure(fig, axs[j][i], u0.T, xname, yname, index, name)
                
                i = i+1
            i = 0
            j = j+1
                
          
    def traj_to_args(self, traj):
        args = np.zeros([len(traj.u),4])
        for t in range(len(args)):
          if traj.u[t,1] > 0:
              args[t,3] = traj.x[t,1]
              args[t,2] = traj.x[t,0]
              args[t,1] = np.nan
              args[t,0] = np.nan
          else:
              args[t,1] = traj.x[t,1]
              args[t,0] = traj.x[t,0]
              args[t,2] = np.nan
              args[t,3] = np.nan
        
        return args
        
    def setup_sys(self):
        print('Setup SYSTEM')
        self.sys.lenght = 0.48
        self.sys.xc = 0.24
        self.sys.yc = 0.15
        self.sys.mass = 20
        self.sys.cdA = 0.3 * 0.105
        self.sys.x_ub = np.array([5, 4.5])
        self.sys.x_lb = np.array([-10., 0])
        self.sys.u_ub = np.array([0.0, 1])
        self.sys.u_lb = np.array([-0.3, 0])        
        self.sys.obs_dist = self.sys.x_ub[0]
        
        self.sys.tf = 1.75
        self.sys.tm_dot = 0.4    
        self.sys.x_grid = [801,101]
        
        ### GETTING TM AND TMDOT ARRAYS
        temp = ''
        i =0
        for l in self.config_txt[4][1:-2]:
             if l is not ',': temp = temp + l
             else:
                  self.tm_array[i] = float(temp)
                  i = i+1
                  temp = ''
        self.tm_array[i] = float(temp)

        temp = ''
        i =0
        for l in self.config_txt[5][1:-2]:
             if l is not ',': temp = temp + l
             else:
                  self.tmd_array[i] = float(temp)
                  i = i+1
                  temp = ''
        self.tmd_array[i] = float(temp)
        
        self.sys.tm = self.tm_array
        self.sys.tm_dot = self.tmd_array
        
        
    def setup_cf(self):
        print('Setup COST FUNCTiON')
        i=0
        for c in self.config_txt[0]:
             if c is ':':
                  self.confort_coef = self.config_txt[0][i+1:-1]
             i=i+1
        i=0
        for c in self.config_txt[1]:
             if c is ':':
                  self.override_coef = self.config_txt[1][i+1:-1]
             i=i+1
        i=0             
        for c in self.config_txt[2]:
             if c is ':':
                  self.security_coef = self.config_txt[2][i+1:-1]
             i=i+1   
             
        self.cf.confort_coef = self.confort_coef
        self.cf.override_coef = self.override_coef
        self.cf.security_coef = self.security_coef
        self.cf.xbar = np.array([0, 0])
        self.cf_list = list((self.cf.g, self.cf.g_confort, self.cf.g_security, self.cf.g_override))
        
    def setup_gridsys(self):
        print('Setup GRID SYSTEM')
        i=0
        for c in self.config_txt[-4]:
            if c is ':':
                self.pos_dim = self.config_txt[-4][i+1:-1]
            i=i+1 
        i=0
        for c in self.config_txt[-3]:
            if c is ':':
                self.vit_dim = self.config_txt[-3][i+1:-1]
            i=i+1 
        i=0
        for c in self.config_txt[-2]:
            if c is ':':
                self.slip_dim = self.config_txt[-2][i+1:-1]
            i=i+1           
        i=0
        for c in self.config_txt[-1]:
            if c is ':':
                self.dt_dim = self.config_txt[-1][i+1:-1]
            i=i+1    
          
        self.grid_sys = discretizer2.GridDynamicSystem(self.sys, [int(self.pos_dim), int(self.vit_dim)], [int(self.slip_dim), 2], float(self.dt_dim),lookup=False)
        for d in self.sys.tm_dot:
            for r in self.roads_array:
                name = r + '_' + str(d)
                print(name)
                self.nom_control.append(name)
                self.sys.road = self.sys.roads[r]
                
                self.grid_syss.append(self.grid_sys)
                dp = dprog.DynamicProgrammingWithLookUpTable(self.grid_sys, self.cf, compute_cost = False)
                dp.load_J_next(self.map_directory+'/'+'xmaxx_'+name)
                self.vi_u.append(dp)
                
    def plot_to_figure(self, fig, ax, data, xname, yname, index, name):
       ax.set_title(name)
       ax.set(xlabel=xname, ylabel=yname)
       i1 = ax.pcolormesh(self.grid_syss[index].x_level[0], self.grid_syss[index].x_level[1], data, shading='gouraud')
       ax.axis([self.grid_syss[index].x_level[0][0], self.grid_syss[index].x_level[0][-1], self.grid_syss[index].x_level[1][0], self.grid_syss[index].x_level[1][-1]])
       fig.colorbar(i1, ax=ax)
       ax.grid(True)
       
       
                
if __name__ == '__main__':                                               
     v = xmaxx_viewing()   