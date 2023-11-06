67#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import time
import ast
import os
import copy
###############################################################################
import valueiteration
import discretizer
import system
import costfunction
import BaselineController

import simulationv2 as s
import CustomCostFunctionSimulation as cf_sim
from pyro.control import controller
import cost2go
###############################################################################

class xmaxxvi_controller:
	def __init__(self, driver_comp = 'Ok', map_directory_name = '/21_9_2022_11h48'):
		self.current_dir = os.getcwd()
		self.map_directory_name = map_directory_name
		self.map_directory = self.current_dir + '/xmaxx_policymap' + self.map_directory_name
		with open(self.map_directory + '/cf_config.txt','r') as f:
		     self.config_txt = f.readlines()		

		self.driver_comp = driver_comp

		self.sys = system.LongitudinalFrontWheelDriveCarWithDriverModel()
		## IF ANY CHANGES OCCURS TO MAP
		self.sys.roads =  ast.literal_eval(self.config_txt[3][0:-1])
		self.sys.driver_xmaxx =  ast.literal_eval(self.config_txt[4][0:-1])
		self.setup_sys()

		self.grid_sys = discretizer.GridDynamicSystem(self.sys, (81, 81), (16, 2), 0.1)

		self.cf = costfunction.DriverModelCostFunction_ROS.from_sys(self.sys)
		self.confort_coef = self.config_txt[0][-2]
		self.override_coef = self.config_txt[1][-2]
		self.security_coef = self.config_txt[2][-2]
		self.setup_cf()
		self.sys.cost_function = self.cf

		self.vi = valueiteration.ValueIteration_ND(self.grid_sys, self.cf)
		self.vi.threshold = 0.5
		self.vi.uselookuptable = False
		self.vi.initialize()
		self.vi_AsphalteDry = copy.deepcopy(self.vi) 		#mu = 1.11
		self.vi_CementDry = copy.deepcopy(self.vi) 		#mu = 1.01
		self.vi_AsphalteWet = copy.deepcopy(self.vi) 		#mu = 0.75
		self.vi_CobblestoneWet = copy.deepcopy(self.vi) 	#mu = 0.40
		self.vi_Snow = copy.deepcopy(self.vi) 			#mu = 0.15
		self.vi_Ice = copy.deepcopy(self.vi) 			#mu = 0.05
		self.mus = [1.11, 1.01, 0.75, 0.40, 0.15, 0.05]
		self.vi_names = ['AsphalteDry', 'CementDry', 'AsphalteWet', 'CobblestoneWet', 'Snow', 'Ice']
		self.vi_ctl = [self.vi_AsphalteDry.ctl, self.vi_CementDry.ctl, self.vi_AsphalteWet.ctl, self.vi_CobblestoneWet.ctl, self.vi_Snow.ctl, self.vi_Ice.ctl]
		#TODO: Changer la creation des differents controller vi si une route vien s'ajoute on s'enleve		
		self.setup_vi()
		
		self.mu_ref = 0.8
		self.sigma_ref = 0.2


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

	def setup_vi(self):
		self.vi_AsphalteDry.load_data(self.map_directory + '/xmaxx_AsphalteDry_' + self.sys.driver[-1])
		self.vi_AsphalteDry.assign_interpol_controller()
		self.vi_AsphalteDry.ctl.vi_law = self.vi_AsphalteDry.vi_law

		self.vi_CementDry.load_data(self.map_directory + '/xmaxx_CementDry_' + self.sys.driver[-1])
		self.vi_CementDry.assign_interpol_controller()
		self.vi_CementDry.ctl.vi_law = self.vi_CementDry.vi_law

		self.vi_AsphalteWet.load_data(self.map_directory + '/xmaxx_AsphalteWet_' + self.sys.driver[-1])
		self.vi_AsphalteWet.assign_interpol_controller()
		self.vi_AsphalteWet.ctl.vi_law = self.vi_AsphalteWet.vi_law

		self.vi_CobblestoneWet.load_data(self.map_directory + '/xmaxx_CobblestoneWet_' + self.sys.driver[-1])
		self.vi_CobblestoneWet.assign_interpol_controller()
		self.vi_CobblestoneWet.ctl.vi_law = self.vi_CobblestoneWet.vi_law

		self.vi_Snow.load_data(self.map_directory + '/xmaxx_Snow_' + self.sys.driver[-1])
		self.vi_Snow.assign_interpol_controller()
		self.vi_Snow.ctl.vi_law = self.vi_Snow.vi_law

		self.vi_Ice.load_data(self.map_directory + '/xmaxx_Ice_' + self.sys.driver[-1])
		self.vi_Ice.assign_interpol_controller()
		self.vi_Ice.ctl.vi_law = self.vi_Ice.vi_law

		print('VI READY TO USE')

	def gaussian_func(self, x_d, mu_d, sigma_d):
		return np.exp(-1*(((x_d-mu_d)**2)/(2*(sigma_d**2))))/(sigma_d*np.sqrt(2*np.pi))

	def plot_gaussian_func(self):
		x_d = np.arange(1001)/500.0
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










