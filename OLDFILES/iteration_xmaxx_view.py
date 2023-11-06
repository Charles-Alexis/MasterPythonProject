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
import copy
import ast

###############################################################################
import valueiteration
import discretizer
import BaselineController
import system
import costfunction
import simulationv2
import simulationv2 as s
###############################################################################

from pyro.control import controller

class xmaxx_viewing:
    def __init__(self, map_directory_name = '/26_10_2022_12h19_0.1_0.1_100_inf', mu_ref = 0.8):
        self.current_dir = os.getcwd()
        self.map_directory_name = map_directory_name
        self.map_directory = self.current_dir + '/xmaxx_policymap' + self.map_directory_name
        with open(self.map_directory + '/cf_config.txt','r') as f:
             self.config_txt = f.readlines()
     
        self.driver_comp = 'Good'
        self.mu_ref = mu_ref
        self.sigma_ref = 0.1     
        
        self.new_map = 0
        self.new_map_1 = 0

        self.policy_plot_vi_AsphalteDry = 0
        self.policy_plot_vi_CementDry = 0
        self.policy_plot_vi_AsphalteWet = 0
        self.policy_plot_vi_CobblestoneWet = 0
        self.policy_plot_vi_Snow = 0
        self.policy_plot_vi_Ice = 0
        self.policy_plot_vi_AsphalteDry_1 = 0
        self.policy_plot_vi_CementDry_1 = 0
        self.policy_plot_vi_AsphalteWet_1 = 0
        self.policy_plot_vi_CobblestoneWet_1 = 0
        self.policy_plot_vi_Snow_1 = 0
        self.policy_plot_vi_Ice_1 = 0
        
        self.sys = system.LongitudinalFrontWheelDriveCarWithDriverModel()
        ## IF ANY CHANGES OCCURS TO MAP
        self.sys.roads =  ast.literal_eval(self.config_txt[3][0:-1])
        self.sys.driver_xmaxx =  ast.literal_eval(self.config_txt[4][0:-1])
        self.setup_sys()
        self.grid_sys = discretizer.GridDynamicSystem(self.sys, (111, 111), (5, 2), 0.1)

        self.cl_sys_AsphalteDry_vi = 0
        self.cl_sys_CementDry_vi = 0
        self.cl_sys_AsphalteWet_vi = 0
        self.cl_sys_CobblestoneWet_vi = 0
        self.cl_sys_Snow_vi = 0
        self.cl_sys_Ice_vi = 0

        self.cf = costfunction.DriverModelCostFunction.from_sys(self.sys)
        self.cf.print_security()
        self.confort_coef = float(self.config_txt[0][-4] + self.config_txt[0][-3] + self.config_txt[0][-2])
        self.override_coef = float(self.config_txt[1][-4] + self.config_txt[1][-3] + self.config_txt[1][-2])
        self.security_coef = float(self.config_txt[2][-4] + self.config_txt[2][-3] + self.config_txt[2][-2])

        self.setup_cf()
        self.sys.cost_function = self.cf

        self.vi = valueiteration.ValueIteration_ND(self.grid_sys, self.cf)
        self.vi.threshold = 0.5
        self.vi.uselookuptable = False
        self.vi.initialize()
        self.vi_combined = copy.deepcopy(self.vi)
        
        self.vi_AsphalteDry = copy.deepcopy(self.vi)         #mu = 1.11
        self.vi_CementDry = copy.deepcopy(self.vi)         #mu = 1.01
        self.vi_AsphalteWet = copy.deepcopy(self.vi)         #mu = 0.75
        self.vi_CobblestoneWet = copy.deepcopy(self.vi)     #mu = 0.40
        self.vi_Snow = copy.deepcopy(self.vi)             #mu = 0.15
        self.vi_Ice = copy.deepcopy(self.vi)             #mu = 0.05
        self.mus = [1.11, 1.01, 0.75, 0.40, 0.15, 0.05]
        self.vi_names = ['AsphalteDry', 'CementDry', 'AsphalteWet', 'CobblestoneWet', 'Snow', 'Ice']
        self.vi_ctl = [self.vi_AsphalteDry.ctl, self.vi_CementDry.ctl, self.vi_AsphalteWet.ctl, self.vi_CobblestoneWet.ctl, self.vi_Snow.ctl, self.vi_Ice.ctl]
        #TODO: Changer la creation des differents controller vi si une route vien s'ajoute on s'enleve   
        self.viewer(vi=True, baseline = True, cost2go=True, driv=self.sys.driver_xmaxx[self.driver_comp])

    def setup_sys(self):
        self.sys.lenght = 0.6
        self.sys.xc = 0.3
        self.sys.yc = 0.175
        self.sys.mass = 25
        self.sys.cdA = 0.3 * 0.105
        self.sys.x_ub = np.array([+20.0, 6.0])
        self.sys.x_lb = np.array([0, 0])
        self.sys.u_ub = np.array([0.0, 1])
        self.sys.u_lb = np.array([-0.3, 0])
        self.sys.driver = self.sys.driver_xmaxx[self.driver_comp]
        self.sys.obs_dist = self.sys.x_ub[0]

    def setup_cf(self):
        self.cf.confort_coef = self.confort_coef
        self.cf.override_coef = self.override_coef
        self.cf.security_coef = self.security_coef
        self.cf.xbar = np.array([0, 0])

    def setup_vi(self, driv):
        self.sys.driver = driv
        self.vi_AsphalteDry.load_data(self.map_directory + '/xmaxx_AsphalteDry_' + driv[-1])
        self.vi_AsphalteDry.assign_interpol_controller()
        self.vi_AsphalteDry.ctl.vi_law = self.vi_AsphalteDry.vi_law
        self.vi_CementDry.load_data(self.map_directory + '/xmaxx_CementDry_' + driv[-1])
        self.vi_CementDry.assign_interpol_controller()
        self.vi_CementDry.ctl.vi_law = self.vi_CementDry.vi_law
        self.vi_AsphalteWet.load_data(self.map_directory + '/xmaxx_AsphalteWet_' + driv[-1])
        self.vi_AsphalteWet.assign_interpol_controller()
        self.vi_AsphalteWet.ctl.vi_law = self.vi_AsphalteWet.vi_law
        self.vi_CobblestoneWet.load_data(self.map_directory + '/xmaxx_CobblestoneWet_' + driv[-1])
        self.vi_CobblestoneWet.assign_interpol_controller()
        self.vi_CobblestoneWet.ctl.vi_law = self.vi_CobblestoneWet.vi_law
        self.vi_Snow.load_data(self.map_directory + '/xmaxx_Snow_' + driv[-1])
        self.vi_Snow.assign_interpol_controller()
        self.vi_Snow.ctl.vi_law = self.vi_Snow.vi_law
        self.vi_Ice.load_data(self.map_directory + '/xmaxx_Ice_' + driv[-1])
        self.vi_Ice.assign_interpol_controller()
        self.vi_Ice.ctl.vi_law = self.vi_Ice.vi_law

    def policy_save(self):
         self.policy_plot_vi_AsphalteDry = self.vi_AsphalteDry.u_policy_grid[0].copy().T
         self.policy_plot_vi_CementDry = self.vi_CementDry.u_policy_grid[0].copy().T
         self.policy_plot_vi_AsphalteWet = self.vi_AsphalteWet.u_policy_grid[0].copy().T
         self.policy_plot_vi_CobblestoneWet = self.vi_CobblestoneWet.u_policy_grid[0].copy().T
         self.policy_plot_vi_Snow = self.vi_Snow.u_policy_grid[0].copy().T
         self.policy_plot_vi_Ice = self.vi_Ice.u_policy_grid[0].copy().T       
         self.policy_plot_vi_AsphalteDry_1 = self.vi_AsphalteDry.u_policy_grid[1].copy().T
         self.policy_plot_vi_CementDry_1 = self.vi_CementDry.u_policy_grid[1].copy().T
         self.policy_plot_vi_AsphalteWet_1 = self.vi_AsphalteWet.u_policy_grid[1].copy().T
         self.policy_plot_vi_CobblestoneWet_1 = self.vi_CobblestoneWet.u_policy_grid[1].copy().T
         self.policy_plot_vi_Snow_1 = self.vi_Snow.u_policy_grid[1].copy().T
         self.policy_plot_vi_Ice_1 = self.vi_Ice.u_policy_grid[1].copy().T
         self.vi_AsphalteDry.Jplot = self.vi_AsphalteDry.J.copy()
         self.vi_CementDry.Jplot = self.vi_CementDry.J.copy()         
         self.vi_AsphalteWet.Jplot = self.vi_AsphalteWet.J.copy()
         self.vi_CobblestoneWet.Jplot = self.vi_CobblestoneWet.J.copy()
         self.vi_Snow.Jplot = self.vi_Snow.J.copy()
         self.vi_Ice.Jplot = self.vi_Ice.J.copy()   
   
    def viewer(self, vi, baseline, cost2go, driv):
         self.setup_vi(driv)
         self.policy_save() 
         self.sys.plot_human_model()
         if vi is True:
             self.viewing()
         if baseline is True:
             self.viewing_baseline()
         if cost2go is True:
             self.viewing_cost2go()
             
         self.setup_cl_sys_road('AsphalteDry', 1.5, -0.143)
         self.setup_cl_sys_road('AsphalteDry', 1.5, -0.136)
         self.setup_cl_sys_road('AsphalteWet', 2.4, -0.112)
         self.setup_cl_sys_road('CobblestoneWet', 5.8, -0.116)
         self.setup_cl_sys_road('Snow', 12, -0.052)
         self.setup_cl_sys_road('Ice', 43, -0.022)
         
    def setup_cl_sys_road(self, road_name, ttc_min, slip_ref):
        vi = 6
        pi = 0
        
        
        self.sys.road = self.sys.roads[road_name]
        cf = costfunction.DriverModelCostFunction.from_sys(self.sys)
        cf.security_coef =self.security_coef
        cf.confort_coef =self.confort_coef
        cf.override_coef =self.override_coef
        
        self.cl_sys_vi = controller.ClosedLoopSystem(self.sys, self.vi_AsphalteDry.ctl)
        self.cl_sys_vi.x0 = np.array([pi, vi])
        self.cl_sys_vi.cost_function = cf
        
        self.ttc_controller = BaselineController.TTCController(self.sys.n, self.sys.m, self.sys.p, self.sys.human_model, self.sys.f, ttc_ref = ttc_min, position_obs=20, slip_cmd=slip_ref)
        self.cl_sys_ttc = controller.ClosedLoopSystem(self.sys , self.ttc_controller)
        self.cl_sys_ttc.x0 = np.array([pi, vi])
        self.cl_sys_ttc.cost_function = cf
        
        self.sim_test = s.SimulatorV2_vi_vs_ttc(self.cl_sys_vi, self.cl_sys_ttc, x0_end = 20, x1_end = 0.03)
        self.sim_test.plot_trajectories(name = self.sys.road[-1] + '_' + self.sys.driver[-1])         
  
         
    def create_cl_sys(self):
        self.action_policy = np.zeros(np.shape(self.new_map))
        for index_x in range(np.shape(self.new_map[0])[0]):
             for index_y in range(np.shape(self.new_map[1])[0]):
                  temp_val = self.find_nearest(self.new_map[index_y][index_x], self.grid_sys.actions_input[:,0])
                  if self.grid_sys.actions_input[temp_val ,0] <  -0.01: #TODO: trouver une facon de rendre ca numerique
                       temp_val = temp_val + 1
                  self.action_policy[index_y][index_x] = temp_val
        self.action_policy = self.action_policy.astype(int) 
        self.vi_combined.action_policy = self.action_policy     
        self.vi_combined.assign_interpol_controller()
        self.vi_combined.ctl.vi_law = self.vi_combined.vi_law        

    def sim_v2(self, x0, road):
        self.sys.road = road
        self.cl_sys = controller.ClosedLoopSystem( self.sys , self.vi_combined.ctl )
        self.cl_sys.cost_function = self.cf
        self.cl_sys.x0 = x0
        
        sim_s = simulationv2.SimulatorV2(self.cl_sys)
        sim_s.x0_end = 30
        sim_s.x1_end = 0.1
        traj_s = sim_s.compute()
        sim_s.plot_trajectory()
                     
  
    def viewing_baseline(self):        
         fig, axs = plt.subplots(2, 3)
         plt.ion()
         fig.suptitle('Optimale baseline for Driver: ' + self.sys.driver[-1])
         xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
         yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]  
         
         axs[0][0].set_title('AsphalteDry')
         axs[0][0].set(xlabel=xname, ylabel=yname)
         i1 = axs[0][0].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.policy_plot_vi_AsphalteDry_1, shading='gouraud')
         axs[0][0].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i1, ax=axs[0, 0])
         axs[0][0].grid(True)
         
         axs[0][1].set_title('CementDry')
         axs[0][1].set(xlabel=xname, ylabel=yname)
         i2 = axs[0][1].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.policy_plot_vi_CementDry_1, shading='gouraud')
         axs[0][1].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i2, ax=axs[0, 1])
         axs[0][1].grid(True)
         
         axs[0][2].set_title('AphalteWet')
         axs[0][2].set(xlabel=xname, ylabel=yname)
         i3 = axs[0][2].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.policy_plot_vi_AsphalteWet_1, shading='gouraud')
         axs[0][2].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i3, ax=axs[0, 2])
         axs[0][2].grid(True)    
         
         axs[1][0].set_title('CobblestoneWet')
         axs[1][0].set(xlabel=xname, ylabel=yname)
         i4 = axs[1][0].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.policy_plot_vi_CobblestoneWet_1, shading='gouraud')
         axs[1][0].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i4, ax=axs[1, 0])
         axs[1][0].grid(True)
         
         axs[1][1].set_title('Snow')
         axs[1][1].set(xlabel=xname, ylabel=yname)
         i5 = axs[1][1].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.policy_plot_vi_Snow_1, shading='gouraud')
         axs[1][1].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i5, ax=axs[1, 1])
         axs[1][1].grid(True)

         axs[1][2].set_title('Ice')
         axs[1][2].set(xlabel=xname, ylabel=yname)
         i6 = axs[1][2].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.policy_plot_vi_Ice_1, shading='gouraud')
         axs[1][2].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i6, ax=axs[1, 2])
         axs[1][2].grid(True)
     
    def viewing(self):         
         fig, axs = plt.subplots(2, 3)
         plt.ion()
         fig.suptitle('Optimal commands for Driver: ' + self.sys.driver[-1])
         xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
         yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]  
         
         axs[0][0].set_title('AsphalteDry')
         axs[0][0].set(xlabel=xname, ylabel=yname)
         i1 = axs[0][0].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.policy_plot_vi_AsphalteDry, shading='gouraud')
         axs[0][0].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i1, ax=axs[0, 0])
         axs[0][0].grid(True)
         
         axs[0][1].set_title('CementDry')
         axs[0][1].set(xlabel=xname, ylabel=yname)
         i2 = axs[0][1].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.policy_plot_vi_CementDry, shading='gouraud')
         axs[0][1].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i2, ax=axs[0, 1])
         axs[0][1].grid(True)
         
         axs[0][2].set_title('AphalteWet')
         axs[0][2].set(xlabel=xname, ylabel=yname)
         i3 = axs[0][2].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.policy_plot_vi_AsphalteWet, shading='gouraud')
         axs[0][2].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i3, ax=axs[0, 2])
         axs[0][2].grid(True)    
         
         axs[1][0].set_title('CobblestoneWet')
         axs[1][0].set(xlabel=xname, ylabel=yname)
         i4 = axs[1][0].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.policy_plot_vi_CobblestoneWet, shading='gouraud')
         axs[1][0].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i4, ax=axs[1, 0])
         axs[1][0].grid(True)
         
         axs[1][1].set_title('Snow')
         axs[1][1].set(xlabel=xname, ylabel=yname)
         i5 = axs[1][1].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.policy_plot_vi_Snow, shading='gouraud')
         axs[1][1].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i5, ax=axs[1, 1])
         axs[1][1].grid(True)

         axs[1][2].set_title('Ice')
         axs[1][2].set(xlabel=xname, ylabel=yname)
         i6 = axs[1][2].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.policy_plot_vi_Ice, shading='gouraud')
         axs[1][2].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i6, ax=axs[1, 2])
         axs[1][2].grid(True)

    def viewing_cost2go(self): 
         fig, axs = plt.subplots(2, 3)
         plt.ion()
         fig.suptitle('Optimal Cost2go for Driver: ' + self.sys.driver[-1])
         xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
         yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]  
         
         axs[0][0].set_title('AsphalteDry')
         axs[0][0].set(xlabel=xname, ylabel=yname)
         i1 = axs[0][0].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.vi_AsphalteDry.Jplot.T, shading='gouraud')
         axs[0][0].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i1, ax=axs[0, 0])
         axs[0][0].grid(True)
         
         axs[0][1].set_title('CementDry')
         axs[0][1].set(xlabel=xname, ylabel=yname)
         i2 = axs[0][1].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.vi_CementDry.Jplot.T, shading='gouraud')
         axs[0][1].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i2, ax=axs[0, 1])
         axs[0][1].grid(True)

         axs[0][2].set_title('AphalteWet')
         axs[0][2].set(xlabel=xname, ylabel=yname)
         i3 = axs[0][2].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.vi_AsphalteWet.Jplot.T, shading='gouraud')
         axs[0][2].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i3, ax=axs[0, 2])
         axs[0][2].grid(True)
         
         axs[1][0].set_title('CobblestoneWet')
         axs[1][0].set(xlabel=xname, ylabel=yname)
         i4 = axs[1][0].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.vi_CobblestoneWet.Jplot.T, shading='gouraud')
         axs[1][0].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i4, ax=axs[1, 0])
         axs[1][0].grid(True)

         axs[1][1].set_title('Snow')
         axs[1][1].set(xlabel=xname, ylabel=yname)
         i5 = axs[1][1].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.vi_Snow.Jplot.T, shading='gouraud')
         axs[1][1].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i5, ax=axs[1, 1])
         axs[1][1].grid(True)

         axs[1][2].set_title('Ice')
         axs[1][2].set(xlabel=xname, ylabel=yname)
         i6 = axs[1][2].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.vi_Ice.Jplot.T, shading='gouraud')
         axs[1][2].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
         fig.colorbar(i6, ax=axs[1, 2])
         axs[1][2].grid(True)

    def gaussian_func(self, x_d, mu_d, sigma_d):
        return np.exp(-1*(((x_d-mu_d)**2)/(2*(sigma_d**2))))/(sigma_d*np.sqrt(2*np.pi))

    def plot_gaussian_func(self):
        x_d = np.arange(1001)/500.0
        plt.figure()
        plt.plot(x_d,self.gaussian_func(x_d,self.mu_ref,self.sigma_ref))
        i = 0
        for mu in self.mus:
            plt.plot(mu, self.gaussian_func(mu, self.mu_ref, self.sigma_ref), 'x')
            plt.text(mu, self.gaussian_func(mu, self.mu_ref, self.sigma_ref), self.vi_names[i] + ': ' + str(mu))
            i = i + 1
        plt.axvline(self.mu_ref)
        plt.show()

    def gaussian_cmd(self, distance, vitesse):
        i = 0
        distance = 20 - distance
        res = np.zeros(len(self.mus))
        res_p = np.zeros(len(self.mus))
        cmd_r = np.zeros([len(self.mus),4])

        for mu in self.mus:
            res[i] = self.gaussian_func(mu, self.mu_ref, self.sigma_ref)
            i = i + 1
        res_p = res / np.sum(res)        

        i = 0
        for vi in self.vi_ctl:
            cmd = vi.vi_law(np.array([distance, vitesse]))
            cmd_r[i, 0] = cmd[0]
            cmd_r[i, 1] = cmd[1]
            cmd_r[i, 2] = cmd[0] * res_p[i]
            cmd_r[i, 3] = cmd[1] * res_p[i]
            i = i + 1

        cmd_to_send = [np.sum(cmd_r[:,2]),np.sum(cmd_r[:,3])]
        return cmd_to_send

    def plot_combined_policy(self):
        res = np.zeros(len(self.mus))
        res_p = np.zeros(len(self.mus))
        i = 0
        for mu in self.mus:
            res[i] = self.gaussian_func(mu, self.mu_ref, self.sigma_ref)
            i = i + 1
        res_p = res / np.sum(res)
        i = 0 
        for mu in self.mus:
            print(str(self.vi_names[i])+': ' + str(res_p[i]*100))
            i = i + 1 
        
        self.new_map = self.policy_plot_vi_AsphalteDry*res_p[0] + self.policy_plot_vi_CementDry*res_p[1] + self.policy_plot_vi_AsphalteWet*res_p[2] + self.policy_plot_vi_CobblestoneWet*res_p[3] + self.policy_plot_vi_Snow*res_p[4] + self.policy_plot_vi_Ice*res_p[5] 
        self.new_map_1 = np.zeros(np.shape(self.new_map))
        for i in range(np.shape(self.new_map)[0]):
            for j in range(np.shape(self.new_map)[1]):
                if self.new_map[i][j] < 0 :
                    self.new_map_1[i][j] = 1.
    
        self.new_cost2go = self.vi_AsphalteDry.Jplot.T*res_p[0] + self.vi_CementDry.Jplot.T*res_p[1] + self.vi_AsphalteWet.Jplot.T*res_p[2] + self.vi_CobblestoneWet.Jplot.T*res_p[3] + self.vi_Snow.Jplot.T*res_p[4] + self.vi_Ice.Jplot.T*res_p[5]
        
        fig, axs = plt.subplots(1, 2)
        plt.ion()
        fig.suptitle('Driver: ' + self.sys.driver[-1] + ' for a ' + str(self.mu_ref)+ 'u road ' )
        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]
        
        axs[0].set_title('Slip')
        axs[0].set(xlabel=xname, ylabel=yname)
        i5 = axs[0].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.new_map, shading='gouraud')
        axs[0].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
        fig.colorbar(i5, ax=axs[0])
        axs[0].grid(True)
        
        axs[1].set_title('Override')
        axs[1].set(xlabel=xname, ylabel=yname)
        i4 = axs[1].pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.new_map_1, shading='gouraud')
        axs[1].axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
        fig.colorbar(i4, ax=axs[1])
        axs[1].grid(True)        

        fig, axs = plt.subplots(1, 1)
        plt.ion()
        fig.suptitle('Cost2Go - Driver: ' + self.sys.driver[-1] + ' for a ' + str(self.mu_ref)+ 'u road ' )
        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]
        
        axs.set_title('Cost2Go')
        axs.set(xlabel=xname, ylabel=yname)
        i5 = axs.pcolormesh(self.grid_sys.xd[0], self.grid_sys.xd[1], self.new_cost2go, shading='gouraud')
        axs.axis([self.sys.x_lb[0], self.sys.x_ub[0], self.sys.x_lb[1], self.sys.x_ub[1]])
        fig.colorbar(i5, ax=axs)
        axs.grid(True)
        
    def find_nearest(self, value, array):
         array = np.asarray(array)
         idx = (np.abs(array - value)).argmin()
         return idx
        
    def config_vi_combined(self):
         self.vi_combined.Jplot = np.copy(self.new_cost2go) 
         self.vi_combined.J = np.copy(self.new_cost2go)
         self.vi_combined.compute_step()
         self.vi_combined.assign_interpol_controller()
         self.vi_combined.ctl.vi_law = self.vi_combined.vi_law 
         self.vi_combined.plot_policy()
                 

if __name__ == '__main__':                                               
     v = xmaxx_viewing(mu_ref = 0.50)    



         