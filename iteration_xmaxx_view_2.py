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
    def __init__(self, map_directory_name = '/13_2_2023_11h2'):
        ## GETTING VALUES IN FOLDER
        self.current_dir = os.getcwd()
        self.map_directory_name = map_directory_name
        self.map_directory = self.current_dir + '/xmaxx_policymap' + self.map_directory_name
        with open(self.map_directory + '/cf_config.txt','r') as f:
             self.config_txt = f.readlines()

        ## SETUP SYS
        self.sys = system.LongitudinalFrontWheelDriveCarWithDriverModel()
        self.sys.roads =  ast.literal_eval(self.config_txt[3][0:-1])
        self.sys.driver_xmaxx =  ast.literal_eval(self.config_txt[4][0:-1])
        self.roads_array = list()
        for r in self.sys.roads: self.roads_array.append(r)
        self.driver_comp = '1'
        
        ## SETUP DRIVER DISTANCE
        slip_data = self.sys.return_max_mu()
        dx = self.sys.f([0,self.sys.x_ub[1]],[-slip_data[1],1])
        time_dist = ((-1*self.sys.x_ub[1])-(-1*self.sys.x_lb[1]))/dx[1]
        displacement = (0.5*((-1*self.sys.x_ub[1])+(-1*self.sys.x_lb[1]))*time_dist)
        self.sys.driver_xmaxx_fort = {
        '5': [displacement + np.abs(displacement*0.4)-1, displacement + np.abs(displacement*0.4) + 1.,'5'],
        '4': [displacement + np.abs(displacement*0.3)-1, displacement + np.abs(displacement*0.3) + 1.,'4'],
        '3': [displacement + np.abs(displacement*0.2)-1, displacement + np.abs(displacement*0.2) + 1.,'3'],
        '2': [displacement + np.abs(displacement*0.1)-1, displacement + np.abs(displacement*0.1) + 1.,'2'],
        '1': [displacement + np.abs(displacement*0.0)-1, displacement + np.abs(displacement*0.0) + 1.,'1'],
        }
        self.sys.driver = self.sys.driver_xmaxx_fort[self.driver_comp]
        self.sys.use_human_model = True
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

        ## SETUP TTC 
        self.ttc_security_factor = 1/1
        self.ttc_distance = np.array([0.708,  1.187, 2.88, 5.952, 21.739]) * self.ttc_security_factor
        self.ttc_slip = np.array([-0.143, -0.112, -0.116, -0.052, -0.022])        
        
        ## SETUP OPTIMAL HUMAN MODEL
        self.optimal_human_model_directory = self.current_dir + '/optimal_human_model/'
        self.nom_human_model = list()
        self.optimal_human_model = list()
        self.setup_optimal_human_model()
        # self.plot_optimal_human_model()
        
        ## HUMAN MODEL AND COST2GO
        self.human_controler = BaselineController.humanController(self.sys, self.grid_sys, self.sys.human_model_exp)
        self.cl_sys_human = controller.ClosedLoopSystem(self.sys , self.human_controler)
        self.cl_sys_human.cost_function = self.sys.cost_function
        self.c_human = cost2go2.cost2go_list(self.grid_sys, self.sys, self.cl_sys_human, self.cf_list)
        # self.c_human.compute_steps(print_iteration=False)
        
        ## TEST AND PLOTTING
#        self.plot_policy('5', 'Snow', 0, -10, 4.0)   
#        self.plot_policy('4', 'Snow', 0, -10., 4.0)     
#        self.plot_policy('3', 'Snow', 0, -10., 4.0)   
#        self.plot_policy('2', 'Snow', 0, -10., 4.0)      
#        self.plot_policy('1', 'Snow', 0, -10, 4.0)    
#
#        self.plot_policy('5', 'CobblestoneWet', 0, -10, 4.0)   
#        self.plot_policy('4', 'CobblestoneWet', 0, -10., 4.0)     
#        self.plot_policy('3', 'CobblestoneWet', 0, -10., 4.0)   
#        self.plot_policy('2', 'CobblestoneWet', 0, -10., 4.0)      
#        self.plot_policy('1', 'CobblestoneWet', 0, -10, 4.0) 
        
        #self.plot_policy('5', 'Snow', 0, -10., 4.0)   
        # self.plot_policy('3', 'Snow', 0, 0, 4.0)      
        # self.plot_policy('1', 'Snow', 0, 0, 4.0)        
        # self.plot_policy('5', 'AsphalteWet', 0, -10, 4.0)  
        # self.plot_policy('1', 'AsphalteWet', 0, -10, 4.0)  
        # self.plot_policy('3', 'AsphalteDry', 0, -10, 4.0)    
                     
        # self.plot_all_commands()
#        self.plot_roads('CobblestoneWet')
        self.plot_driver('1')
        # self.plot_roads_driver('CobblestoneWet')
        
        # self.plot_roads('Snow')
        # self.plot_roads_driver('Snow')
        
        
    def plot_roads(self, road_name):
        nbr_road = len(self.sys.roads)
         
        fig, axs = plt.subplots(2, nbr_road)
        plt.ion()
        fig.suptitle('Optimale commands for road' + str(road_name))
        ## SETUP SYSTEM
        self.sys.road = self.sys.roads[road_name]                
        slip_data = self.sys.return_max_mu()
        dx = self.sys.f([0,self.sys.x_ub[1]],[-slip_data[1],1])
        time_dist = ((-1*self.sys.x_ub[1])-(-1*self.sys.x_lb[1]))/dx[1]
        displacement = (0.5*((-1*self.sys.x_ub[1])+(-1*self.sys.x_lb[1]))*time_dist)
        self.sys.driver_xmaxx_fort = {
        '5': [displacement + np.abs(displacement*0.4)-1, displacement + np.abs(displacement*0.4) + 1.,'5'],
        '4': [displacement + np.abs(displacement*0.3)-1, displacement + np.abs(displacement*0.3) + 1.,'4'],
        '3': [displacement + np.abs(displacement*0.2)-1, displacement + np.abs(displacement*0.2) + 1.,'3'],
        '2': [displacement + np.abs(displacement*0.1)-1, displacement + np.abs(displacement*0.1) + 1.,'2'],
        '1': [displacement + np.abs(displacement*0.0)-1, displacement + np.abs(displacement*0.0) + 1.,'1'],
        }

        j = 0
        for driver in self.sys.driver_xmaxx_fort:
             name = road_name +'_'+driver
             for index_temp in range(len(self.nom_control)):
                  if name == self.nom_control[index_temp]:
                       index = index_temp
             
             self.sys.driver = self.sys.driver_xmaxx_fort[driver]
             self.sys.x_grid = self.grid_sys.x_grid_dim
             human_model = self.sys.plot_human_model_exp(plot=False)

             ## MAKING CLOSED LOOP SYSTEM
             uk_0 = self.grid_syss[index].get_input_from_policy(self.vi_u[index].pi, 0)
             u0 = self.grid_syss[index].get_grid_from_array(uk_0)
             self.grid_sys = discretizer2.GridDynamicSystem(self.sys, [int(self.pos_dim), int(self.vit_dim)], [int(self.slip_dim), 2], float(self.dt_dim),lookup=False)
             ctl = dprog.LookUpTableController(self.grid_sys, self.vi_u[index].pi)
             self.sys.use_human_model = False
             self.cl_sys_vi = controller.ClosedLoopSystem( self.sys , ctl)
             self.cl_sys_vi.cost_function = self.sys.cost_function
             self.cl_sys_vi.x0 = np.array([-2.5, 4.0])
             ## SIMULATION
             self.sim = s.SimulatorV2(self.cl_sys_vi, x0_end=0) 
             args = self.traj_to_args(self.sim.traj)
           
             ## PLOTTING
             axs[0][j].set_title(name)
             i1 = axs[0][j].pcolormesh(self.grid_syss[index].x_level[0], self.grid_syss[index].x_level[1], u0.T, shading='gouraud')
             axs[0][j].axis([self.grid_syss[index].x_level[0][0], self.grid_syss[index].x_level[0][-1], self.grid_syss[index].x_level[1][0], self.grid_syss[index].x_level[1][-1]])
             fig.colorbar(i1, ax=axs[0][j])
             axs[0][j].grid(True)
             axs[0][j].plot(args[:,2],args[:,3])
             axs[0][j].plot(args[:,0],args[:,1])
             
             axs[1][j].set_title('HUMAN MODEL '+name)
             i1 = axs[1][j].pcolormesh(self.grid_syss[index].x_level[0], self.grid_syss[index].x_level[1], human_model.T, shading='gouraud')
             axs[1][j].axis([self.grid_syss[index].x_level[0][0], self.grid_syss[index].x_level[0][-1], self.grid_syss[index].x_level[1][0], self.grid_syss[index].x_level[1][-1]])
             fig.colorbar(i1, ax=axs[1][j])
             axs[1][j].grid(True)
             axs[1][j].plot(args[:,2],args[:,3])
             axs[1][j].plot(args[:,0],args[:,1])
             
             
             j = j+1
  
    def plot_roads_driver(self, road_name):
        nbr_road = len(self.sys.roads)
         
        fig, axs = plt.subplots(1, nbr_road)
        plt.ion()
        fig.suptitle('Human Model for road: ' + str(road_name))

        j = 0
        for driver in self.sys.driver_xmaxx_fort:
             name = road_name +'_'+driver
             for index_temp in range(len(self.nom_control)):
                  if name == self.nom_control[index_temp]:
                       index = index_temp
                
             self.sys.road = self.sys.roads[road_name]                
             slip_data = self.sys.return_max_mu()
             dx = self.sys.f([0,self.sys.x_ub[1]],[-slip_data[1],1])
             time_dist = ((-1*self.sys.x_ub[1])-(-1*self.sys.x_lb[1]))/dx[1]
             displacement = (0.5*((-1*self.sys.x_ub[1])+(-1*self.sys.x_lb[1]))*time_dist)
             self.sys.driver_xmaxx_fort = {
             '5': [displacement + np.abs(displacement*0.4), displacement + np.abs(displacement*0.4) + 1.,'5'],
             '4': [displacement + np.abs(displacement*0.3), displacement + np.abs(displacement*0.3) + 1.,'4'],
             '3': [displacement + np.abs(displacement*0.2), displacement + np.abs(displacement*0.2) + 1.,'3'],
             '2': [displacement + np.abs(displacement*0.1), displacement + np.abs(displacement*0.1) + 1.,'2'],
             '1': [displacement + np.abs(displacement*0.0), displacement + np.abs(displacement*0.0) + 1.,'1'],
             }
             self.sys.driver = self.sys.driver_xmaxx_fort[driver]
             self.sys.x_grid = self.grid_sys.x_grid_dim
             self.sys.cost_function = self.cf
             
             human_model = self.sys.plot_human_model(plot=False)
             self.human_controler = BaselineController.humanController(self.sys, self.grid_sys, self.sys.human_model)
             self.cl_sys_human = controller.ClosedLoopSystem(self.sys , self.human_controler)
             self.cl_sys_human.x0 = np.array([-10,4.0])
             
             self.cl_sys_human.cost_function = self.sys.cost_function
             self.sim_human = s.SimulatorV2(self.cl_sys_human, x0_end=0) 
             args = self.sim_human.traj_to_args(self.sim_human.traj)
           
             axs[j].set_title(name)
             i1 = axs[j].pcolormesh(self.grid_syss[index].x_level[0], self.grid_syss[index].x_level[1], human_model.T, shading='gouraud')
             axs[j].axis([self.grid_syss[index].x_level[0][0], self.grid_syss[index].x_level[0][-1], self.grid_syss[index].x_level[1][0], self.grid_syss[index].x_level[1][-1]])
             fig.colorbar(i1, ax=axs[j])
             axs[j].grid(True)
             axs[j].plot(args[:,2],args[:,3])
             axs[j].plot(args[:,0],args[:,1])
             j = j+1

    def plot_driver(self, driv_nbr):
        nbr_road = len(self.sys.roads)
        name = 'Optimale'
        fig, axs = plt.subplots(1, nbr_road)
        plt.ion()
        fig.suptitle('Optimale commands for driver: ' + str(name))
        ## SETUP SYSTEM               
        
        slip_data = self.sys.return_max_mu()
        dx = self.sys.f([0,self.sys.x_ub[1]],[-slip_data[1],1])
        time_dist = ((-1*self.sys.x_ub[1])-(-1*self.sys.x_lb[1]))/dx[1]
        displacement = (0.5*((-1*self.sys.x_ub[1])+(-1*self.sys.x_lb[1]))*time_dist)
        self.sys.driver_xmaxx_fort = {
        '5': [displacement + np.abs(displacement*0.4)-1, displacement + np.abs(displacement*0.4) + 1.,'5'],
        '4': [displacement + np.abs(displacement*0.3)-1, displacement + np.abs(displacement*0.3) + 1.,'4'],
        '3': [displacement + np.abs(displacement*0.2)-1, displacement + np.abs(displacement*0.2) + 1.,'3'],
        '2': [displacement + np.abs(displacement*0.1)-1, displacement + np.abs(displacement*0.1) + 1.,'2'],
        '1': [displacement + np.abs(displacement*0.0)-1, displacement + np.abs(displacement*0.0) + 1.,'1'],
        }
        self.sys.driver = driv_nbr
        self.sys.driver = self.sys.driver_xmaxx_fort[driv_nbr]
        self.sys.use_human_model = False

        j = 0
        for road in self.sys.roads:
             name = str(road) +'_'+driv_nbr
             for index_temp in range(len(self.nom_control)):
                  if name == self.nom_control[index_temp]:
                       index = index_temp
             self.sys.road = self.sys.roads[road] 
             
             self.sys.x_grid = self.grid_sys.x_grid_dim

             ## MAKING CLOSED LOOP SYSTEM
             uk_0 = self.grid_syss[index].get_input_from_policy(self.vi_u[index].pi, 0)
             u0 = self.grid_syss[index].get_grid_from_array(uk_0)
             
             self.grid_sys = discretizer2.GridDynamicSystem(self.sys, [int(self.pos_dim), int(self.vit_dim)], [int(self.slip_dim), 2], float(self.dt_dim),lookup=False)
             ctl = dprog.LookUpTableController(self.grid_sys, self.vi_u[index].pi)
             self.sys.use_human_model = False
             self.cl_sys_vi = controller.ClosedLoopSystem( self.sys , ctl)
             self.cl_sys_vi.cost_function = self.sys.cost_function
             self.cl_sys_vi.x0 = np.array([-10., 4.0])
             ## SIMULATION
             self.sim = s.SimulatorV2(self.cl_sys_vi, x0_end=0) 
             args = self.traj_to_args(self.sim.traj)
           
             ## PLOTTING
             axs[j].set_title(name)
             i1 = axs[j].pcolormesh(self.grid_syss[index].x_level[0], self.grid_syss[index].x_level[1], u0.T, shading='gouraud')
             axs[j].axis([self.grid_syss[index].x_level[0][0], self.grid_syss[index].x_level[0][-1], self.grid_syss[index].x_level[1][0], self.grid_syss[index].x_level[1][-1]])
             fig.colorbar(i1, ax=axs[j])
             axs[j].grid(True)
             axs[j].plot(args[:,2],args[:,3])
             axs[j].plot(args[:,0],args[:,1])                          
             j = j+1
                  
        
    def plot_all_commands(self):
        nbr_driv = len(self.sys.driver_xmaxx_fort)
        nbr_road = len(self.sys.roads)
        
        fig, axs = plt.subplots(nbr_road, nbr_driv)
        plt.ion()
        fig.suptitle('Optimale commands')
        i = 0
        j = 0
        for driver in self.sys.driver_xmaxx_fort:
            for road in self.sys.roads:
                name = road+'_'+driver
                for index_temp in range(len(self.nom_control)):
                    if name == self.nom_control[index_temp]:
                        index = index_temp
                        

                
                self.sys.road = self.sys.roads[road]                
                slip_data = self.sys.return_max_mu()
                dx = self.sys.f([0,self.sys.x_ub[1]],[-slip_data[1],1])
                time_dist = ((-1*self.sys.x_ub[1])-(-1*self.sys.x_lb[1]))/dx[1]
                displacement = (0.5*((-1*self.sys.x_ub[1])+(-1*self.sys.x_lb[1]))*time_dist)
                self.sys.driver_xmaxx_fort = {
                '5': [displacement + np.abs(displacement*0.4), displacement + np.abs(displacement*0.4) + 1.,'5'],
                '4': [displacement + np.abs(displacement*0.3), displacement + np.abs(displacement*0.3) + 1.,'4'],
                '3': [displacement + np.abs(displacement*0.2), displacement + np.abs(displacement*0.2) + 1.,'3'],
                '2': [displacement + np.abs(displacement*0.1), displacement + np.abs(displacement*0.1) + 1.,'2'],
                '1': [displacement + np.abs(displacement*0.0), displacement + np.abs(displacement*0.0) + 1.,'1'],
                }
                self.sys.driver = self.sys.driver_xmaxx_fort[driver]


                uk_0 = self.grid_syss[index].get_input_from_policy(self.vi_u[index].pi, 0)
                u0 = self.grid_syss[index].get_grid_from_array(uk_0)
                self.grid_sys = discretizer2.GridDynamicSystem(self.sys, [int(self.pos_dim), int(self.vit_dim)], [int(self.slip_dim), 2], float(self.dt_dim),lookup=False)
                ctl = dprog.LookUpTableController(self.grid_sys, self.vi_u[index].pi)
                self.cl_sys_vi = controller.ClosedLoopSystem( self.sys , ctl)
                self.cl_sys_vi.cost_function = self.sys.cost_function
                self.cl_sys_vi.x0 = np.array([-10., 4.0])
                
                self.sim = s.SimulatorV2(self.cl_sys_vi, x0_end=0) 
                args = self.traj_to_args(self.sim.traj)
                
                axs[i][j].set_title(name)
                i1 = axs[i][j].pcolormesh(self.grid_syss[index].x_level[0], self.grid_syss[index].x_level[1], u0.T, shading='gouraud')
                axs[i][j].axis([self.grid_syss[index].x_level[0][0], self.grid_syss[index].x_level[0][-1], self.grid_syss[index].x_level[1][0], self.grid_syss[index].x_level[1][-1]])
                fig.colorbar(i1, ax=axs[i, j])
                axs[i][j].grid(True)
                axs[i][j].plot(args[:,2],args[:,3])
                axs[i][j].plot(args[:,0],args[:,1])
                
                i = i+1
            i = 0
            j = j+1
                
                
    def plot_policy(self, driv, road, k, pos_ini, vit_ini):
        ## FINDING CONROLERS IN ARRAY
        name = road+'_'+driv
        index = -1
        for i in range(len(self.nom_control)):
            if name == self.nom_control[i]:
                index = i
        ## FOR FIGURE
        fig, axs = plt.subplots(2, 3)
        plt.ion()
        fig.suptitle('Optimale baseline for Driver: ' + driv + ' On road: ' + road)
        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]  
        self.sys.road = self.sys.roads[road]
        
        slip_data = self.sys.return_max_mu()
        dx = self.sys.f([0,self.sys.x_ub[1]],[-slip_data[1],1])
        time_dist = ((-1*self.sys.x_ub[1])-(-1*self.sys.x_lb[1]))/dx[1]
        displacement = (0.5*((-1*self.sys.x_ub[1])+(-1*self.sys.x_lb[1]))*time_dist)
        self.sys.driver_xmaxx_fort = {
        '5': [displacement + np.abs(displacement*0.4), displacement + np.abs(displacement*0.4) + 1.,'5'],
        '4': [displacement + np.abs(displacement*0.3), displacement + np.abs(displacement*0.3) + 1.,'4'],
        '3': [displacement + np.abs(displacement*0.2), displacement + np.abs(displacement*0.2) + 1.,'3'],
        '2': [displacement + np.abs(displacement*0.1), displacement + np.abs(displacement*0.1) + 1.,'2'],
        '1': [displacement + np.abs(displacement*0.0), displacement + np.abs(displacement*0.0) + 1.,'1'],
        }
        self.sys.driver = self.sys.driver_xmaxx_fort[driv]
                
        #GETTING CONTROL LAWS        
        uk_0 = self.grid_syss[index].get_input_from_policy(self.vi_u[index].pi, k)
        u0 = self.grid_syss[index].get_grid_from_array(uk_0)
        
        ttc_controller = BaselineController.TTCController(self.sys, self.grid_sys, self.sys.human_model, ttc_ref=self.ttc_distance[index%len(self.sys.roads)], position_obs=-0.5, slip_cmd=self.ttc_slip[index%len(self.sys.roads)])
        u0_ttc = ttc_controller.c_array()
        
        x0   = np.array([pos_ini, vit_ini])
        
        ctl = dprog.LookUpTableController(self.grid_sys, self.vi_u[index].pi)
        self.cl_sys_vi = controller.ClosedLoopSystem( self.sys , ctl)
        self.cl_sys_vi.cost_function = self.sys.cost_function
        self.cl_sys_vi.x0 = x0

        self.debug_val = u0_ttc 
        self.cl_sys_ttc = controller.ClosedLoopSystem(self.sys , ttc_controller)
        self.cl_sys_ttc.cost_function = self.sys.cost_function 
        self.cl_sys_ttc.x0 = x0
        
        self.cl_sys_human.x0 = x0
        
        self.sim = s.SimulatorV2(self.cl_sys_vi, x0_end=0) 
        self.sim_ttc = s.SimulatorV2(self.cl_sys_ttc, x0_end=0)
        self.sim_human = s.SimulatorV2(self.cl_sys_human, x0_end=0) 
        
        self.sim.cf = self.cf
        
        self.sys.x_grid = self.grid_sys.x_grid_dim
        human_model = self.sys.plot_human_model(plot=False)
        
        ## COST2GO
        self.c_vi = cost2go2.cost2go_list(self.grid_sys, self.sys, self.cl_sys_vi, self.cf_list)
        self.c_vi.compute_steps(print_iteration=False)
        self.c_ttc = cost2go2.cost2go_list(self.grid_sys, self.sys, self.cl_sys_ttc, self.cf_list)
        self.c_ttc.compute_steps(print_iteration=False)
        
        args = self.traj_to_args(self.sim.traj)
        args2 = self.traj_to_args(self.sim_ttc.traj)
        args3 = self.sim_human.traj_to_args(self.sim_human.traj)
        
        axs[0][0].set_title('VI OPTIMAL U')
        axs[0][0].set(xlabel=xname, ylabel=yname)
        i1 = axs[0][0].pcolormesh(self.grid_syss[index].x_level[0], self.grid_syss[index].x_level[1], u0.T, shading='gouraud')
        axs[0][0].axis([self.grid_syss[index].x_level[0][0], self.grid_syss[index].x_level[0][-1], self.grid_syss[index].x_level[1][0], self.grid_syss[index].x_level[1][-1]])
        fig.colorbar(i1, ax=axs[0, 0])
        axs[0][0].grid(True)
        axs[0][0].plot(args[:,2],args[:,3])
        axs[0][0].plot(args[:,0],args[:,1])
        
        axs[0][1].set_title('TTC U')
        axs[0][1].set(xlabel=xname, ylabel=yname)
        i2 = axs[0][1].pcolormesh(self.grid_syss[index].x_level[0], self.grid_syss[index].x_level[1], u0_ttc.T, shading='gouraud')
        axs[0][1].axis([self.grid_syss[index].x_level[0][0], self.grid_syss[index].x_level[0][-1], self.grid_syss[index].x_level[1][0], self.grid_syss[index].x_level[1][-1]])
        fig.colorbar(i2, ax=axs[0, 1])
        axs[0][1].grid(True)
        axs[0][1].plot(args2[:,2],args2[:,3])
        axs[0][1].plot(args2[:,0],args2[:,1])
        
        axs[0][2].set_title('HUMAN MODEL COMMANDS')
        axs[0][2].set(xlabel=xname, ylabel=yname)
        i3 = axs[0][2].pcolormesh(self.grid_syss[index].x_level[0], self.grid_syss[index].x_level[1], human_model.T, shading='gouraud')
        axs[0][2].axis([self.grid_syss[index].x_level[0][0], self.grid_syss[index].x_level[0][-1], self.grid_syss[index].x_level[1][0], self.grid_syss[index].x_level[1][-1]])
        fig.colorbar(i3, ax=axs[0, 2])
        axs[0][2].grid(True)
        axs[0][2].plot(args[:,2],args[:,3])
        axs[0][2].plot(args[:,0],args[:,1])
        axs[0][2].plot(args2[:,2],args2[:,3])
        axs[0][2].plot(args2[:,0],args2[:,1])
          
        axs[1][0].set_title('VI cost2go')
        axs[1][0].set(xlabel=xname, ylabel=yname)
        i4 = axs[1][0].pcolormesh(self.grid_sys.x_level[0], self.grid_sys.x_level[1], np.clip(self.c_vi.cost2go_map_list[0].T,0,np.median(self.c_vi.cost2go_map_list[0])), shading='gouraud')
        axs[1][0].axis([self.grid_sys.x_level[0][0], self.grid_sys.x_level[0][-1], self.grid_sys.x_level[1][0], self.grid_sys.x_level[1][-1]])
        fig.colorbar(i4, ax=axs[1, 0])
        axs[1][0].grid(True)
        axs[1][0].plot(args[:,2],args[:,3])
        axs[1][0].plot(args[:,0],args[:,1])
          
        axs[1][1].set_title('TTC cost2go')
        axs[1][1].set(xlabel=xname, ylabel=yname)
        i5 = axs[1][1].pcolormesh(self.grid_sys.x_level[0], self.grid_sys.x_level[1], np.clip(self.c_ttc.cost2go_map_list[0].T,0,np.median(self.c_ttc.cost2go_map_list[0])), shading='gouraud')
        axs[1][1].axis([self.grid_sys.x_level[0][0], self.grid_sys.x_level[0][-1], self.grid_sys.x_level[1][0], self.grid_sys.x_level[1][-1]])
        fig.colorbar(i5, ax=axs[1, 1])
        axs[1][1].grid(True)
        axs[1][1].plot(args2[:,2],args2[:,3])
        axs[1][1].plot(args2[:,0],args2[:,1])
          
        axs[1][2].set_title('Human cost2go')
        axs[1][2].set(xlabel=xname, ylabel=yname)
        i6 = axs[1][2].pcolormesh(self.grid_sys.x_level[0], self.grid_sys.x_level[1], np.clip(self.c_human.cost2go_map_list[0].T,0,np.median(self.c_human.cost2go_map_list[0])), shading='gouraud')
        axs[1][2].axis([self.grid_sys.x_level[0][0], self.grid_sys.x_level[0][-1], self.grid_sys.x_level[1][0], self.grid_sys.x_level[1][-1]])
        fig.colorbar(i6, ax=axs[1, 2])
        axs[1][2].grid(True)
        axs[1][2].plot(args[:,2],args[:,3])
        axs[1][2].plot(args[:,0],args[:,1])
        axs[1][2].plot(args2[:,2],args2[:,3])
        axs[1][2].plot(args2[:,0],args2[:,1])
        axs[1][2].plot(args3[:,2],args3[:,3])
        axs[1][2].plot(args3[:,0],args3[:,1])
        
        fig2, axs2 = plt.subplots(9, 3)
        plt.ion()
        fig2.suptitle('Simulation parameters for ' + driv + ' On road: ' + road)
        self.sim.plot_trajectories_new_figure('VI', axs2[:,0], print_label=True)
        self.sim_ttc.plot_trajectories_new_figure('TTC', axs2[:,1], print_label=True)
        self.sim_human.plot_trajectories_new_figure('HUMAN', axs2[:,2], print_label=True)
        
        
    def plot_policies(self, driv, k):
        offset = 0
        if driv == 'Ok':
            offset = 6
        if driv == 'Good':
            offset = 6
        
        fig, axs = plt.subplots(2, 6)
        fig2, axs2 = plt.subplots(2, 6)
        plt.ion()
        fig.suptitle('Optimale baseline commands for Driver: ' + driv)
        fig2.suptitle('TTC commands for Driver: ' + driv)
        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]  
        
        for i in range(6):
            uk_0 = self.grid_syss[i+offset].get_input_from_policy(self.vi_u[i+offset].pi, k)
            u0 = self.grid_syss[i+offset].get_grid_from_array(uk_0)
            
            self.sys.road = self.sys.roads[self.roads_array[i]]
            self.sys.driver = self.sys.driver_xmaxx[driv]
            
            ctl = dprog.LookUpTableController(self.grid_sys , self.vi_u[i+offset].pi)
            self.cl_sys = controller.ClosedLoopSystem( self.sys , ctl )
            self.cl_sys.cost_function = self.sys.cost_function
            x0   = np.array([0,3])
            self.cl_sys.x0 = x0
            
            ttc_controller = BaselineController.TTCController(self.sys, self.grid_sys, self.sys.human_model, ttc_ref=self.ttc_distance[i], position_obs=20, slip_cmd=self.ttc_slip[i])
            u0_ttc = ttc_controller.c_array()
            self.debug_val = u0_ttc 
            self.cl_sys_ttc = controller.ClosedLoopSystem(self.sys , ttc_controller)
            self.cl_sys_ttc.cost_function = self.sys.cost_function 
            self.cl_sys_ttc.x0 = x0
            
            self.sim = s.SimulatorV2(self.cl_sys, x0_end=20) 
            self.sim_ttc = s.SimulatorV2(self.cl_sys_ttc, x0_end=20)  
            self.sys.x_grid = self.grid_sys.x_grid_dim
            human_model = self.sys.plot_human_model(plot=False)
            
            args = self.traj_to_args(self.sim.traj)
            args2 = self.traj_to_args(self.sim_ttc.traj)
            
            j = i*2
            w = 0
            if j>4:
                w = 1
                j = j-6
              
            #### VI DP
            axs[w][j].set_title(self.roads_array[i])
            axs[w][j].set(xlabel=xname, ylabel=yname)
            i1 = axs[w][j].pcolormesh(self.grid_syss[i+offset].x_level[0], self.grid_syss[i+offset].x_level[1], u0.T, shading='gouraud')
            axs[w][j].axis([self.grid_syss[i+offset].x_level[0][0], self.grid_syss[i+offset].x_level[0][-1], self.grid_syss[i+offset].x_level[1][0], self.grid_syss[i+offset].x_level[1][-1]])
            fig.colorbar(i1, ax=axs[w, j])
            axs[w][j].grid(True)
            axs[w][j].plot(args[:,2],args[:,3])
            axs[w][j].plot(args[:,0],args[:,1])
            
            axs[w][j+1].set_title(self.roads_array[i])
            axs[w][j+1].set(xlabel=xname, ylabel=yname)
            i1 = axs[w][j+1].pcolormesh(self.grid_syss[i+offset].x_level[0], self.grid_syss[i+offset].x_level[1], human_model.T, shading='gouraud')
            axs[w][j+1].axis([self.grid_syss[i+offset].x_level[0][0], self.grid_syss[i+offset].x_level[0][-1], self.grid_syss[i+offset].x_level[1][0], self.grid_syss[i+offset].x_level[1][-1]])
            fig.colorbar(i1, ax=axs[w, j+1])
            axs[w][j+1].grid(True)
            axs[w][j+1].plot(args[:,2],args[:,3])
            axs[w][j+1].plot(args[:,0],args[:,1])  
            
            #### TTC
            axs2[w][j].set_title(self.roads_array[i])
            axs2[w][j].set(xlabel=xname, ylabel=yname)
            i2 = axs2[w][j].pcolormesh(self.grid_syss[i+offset].x_level[0], self.grid_syss[i+offset].x_level[1], u0_ttc.T, shading='gouraud')
            axs2[w][j].axis([self.grid_syss[i+offset].x_level[0][0], self.grid_syss[i+offset].x_level[0][-1], self.grid_syss[i+offset].x_level[1][0], self.grid_syss[i+offset].x_level[1][-1]])
            fig2.colorbar(i2, ax=axs2[w, j])
            axs2[w][j].grid(True)
            axs2[w][j].plot(args2[:,2],args2[:,3])
            axs2[w][j].plot(args2[:,0],args2[:,1])
            
            axs2[w][j+1].set_title(self.roads_array[i])
            axs2[w][j+1].set(xlabel=xname, ylabel=yname)
            i2 = axs2[w][j+1].pcolormesh(self.grid_syss[i+offset].x_level[0], self.grid_syss[i+offset].x_level[1], human_model.T, shading='gouraud')
            axs2[w][j+1].axis([self.grid_syss[i+offset].x_level[0][0], self.grid_syss[i+offset].x_level[0][-1], self.grid_syss[i+offset].x_level[1][0], self.grid_syss[i+offset].x_level[1][-1]])
            fig2.colorbar(i2, ax=axs2[w, j+1])
            axs2[w][j+1].grid(True)
            axs2[w][j+1].plot(args2[:,2],args2[:,3])
            axs2[w][j+1].plot(args2[:,0],args2[:,1])  
          
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
        self.sys.x_ub = np.array([0, 4.5])
        self.sys.x_lb = np.array([-10., 0])
        self.sys.u_ub = np.array([0.0, 1])
        self.sys.u_lb = np.array([-0.3, 0])
        self.sys.driver = self.sys.driver_xmaxx_fort[self.driver_comp]
        self.sys.obs_dist = self.sys.x_ub[0]  

        
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
        for c in self.config_txt[5]:
            if c is ':':
                self.pos_dim = self.config_txt[5][i+1:-1]
            i=i+1 
        i=0
        for c in self.config_txt[6]:
            if c is ':':
                self.vit_dim = self.config_txt[6][i+1:-1]
            i=i+1 
        i=0
        for c in self.config_txt[7]:
            if c is ':':
                self.slip_dim = self.config_txt[7][i+1:-1]
            i=i+1           
        i=0
        for c in self.config_txt[8]:
            if c is ':':
                self.dt_dim = self.config_txt[8][i+1:-1]
            i=i+1    
          
        self.grid_sys = discretizer2.GridDynamicSystem(self.sys, [int(self.pos_dim), int(self.vit_dim)], [int(self.slip_dim), 2], float(self.dt_dim),lookup=False)
        for d in self.sys.driver_xmaxx:
            for r in self.roads_array:
                name = r + '_' + d
                print(name)
                self.nom_control.append(name)
                self.sys.road = self.sys.roads[r]
                
                self.grid_syss.append(self.grid_sys)
                dp = dprog.DynamicProgrammingWithLookUpTable(self.grid_sys, self.cf, compute_cost = False)
                dp.load_J_next(self.map_directory+'/'+'xmaxx_'+name)
                self.vi_u.append(dp)
                
        
    def setup_optimal_human_model(self):
            for r in self.roads_array:
                name = r
                self.nom_human_model.append(name)
                self.sys.road = self.sys.roads[r]
                dp = dprog.DynamicProgrammingWithLookUpTable(self.grid_sys, self.cf, compute_cost = False)
                dp.load_J_next(self.optimal_human_model_directory+name)
                self.optimal_human_model.append(dp)
                         
    def plot_optimal_human_model(self):
        nbr_road = len(self.sys.roads)
         
        fig, axs = plt.subplots(1, nbr_road)
        plt.ion()
        fig.suptitle('Optimale HUMNA MODEL for road')
        j = 0      
        for road in self.sys.roads:
             name = road
             for index_temp in range(len(self.nom_human_model)):
                  if name == self.nom_human_model[index_temp]:
                       index = index_temp
                       
             self.sys.road = self.sys.roads[road]
             
             slip_data = self.sys.return_max_mu()
             dx = self.sys.f([0,self.sys.x_ub[1]],[-slip_data[1],1])
             time_dist = ((-1*self.sys.x_ub[1])-(-1*self.sys.x_lb[1]))/dx[1]
             displacement = (0.5*((-1*self.sys.x_ub[1])+(-1*self.sys.x_lb[1]))*time_dist)
             self.sys.driver_xmaxx_fort = {
             '5': [displacement + np.abs(displacement*0.4), displacement + np.abs(displacement*0.4) + 1.,'5'],
             '4': [displacement + np.abs(displacement*0.3), displacement + np.abs(displacement*0.3) + 1.,'4'],
             '3': [displacement + np.abs(displacement*0.2), displacement + np.abs(displacement*0.2) + 1.,'3'],
             '2': [displacement + np.abs(displacement*0.1), displacement + np.abs(displacement*0.1) + 1.,'2'],
             '1': [displacement + np.abs(displacement*0.0), displacement + np.abs(displacement*0.0) + 1.,'1'],
             }
             self.sys.driver = self.sys.driver_xmaxx_fort['3']
             
             uk_0 = self.grid_syss[index].get_input_from_policy(self.optimal_human_model[index].pi, 0)
             u0 = self.grid_syss[index].get_grid_from_array(uk_0)
             self.grid_sys = discretizer2.GridDynamicSystem(self.sys, [int(self.pos_dim), int(self.vit_dim)], [int(self.slip_dim), 2], float(self.dt_dim),lookup=False)
             ctl = dprog.LookUpTableController(self.grid_sys, self.optimal_human_model[index].pi)
             self.cl_sys_optimal_human = controller.ClosedLoopSystem( self.sys , ctl)
             self.cl_sys_optimal_human.cost_function = self.sys.cost_function
             self.cl_sys_optimal_human.x0 = np.array([-10, 4.0])
           
             self.sim = s.SimulatorV2(self.cl_sys_optimal_human, x0_end=0) 
             args = self.traj_to_args(self.sim.traj)
           
             axs[j].set_title(name)
             i1 = axs[j].pcolormesh(self.grid_syss[index].x_level[0], self.grid_syss[index].x_level[1], u0.T, shading='gouraud')
             axs[j].axis([self.grid_syss[index].x_level[0][0], self.grid_syss[index].x_level[0][-1], self.grid_syss[index].x_level[1][0], self.grid_syss[index].x_level[1][-1]])
             fig.colorbar(i1, ax=axs[j])
             axs[j].grid(True)
             axs[j].plot(args[:,2],args[:,3])
             axs[j].plot(args[:,0],args[:,1])
             j = j+1        
                
if __name__ == '__main__':                                               
     v = xmaxx_viewing()   