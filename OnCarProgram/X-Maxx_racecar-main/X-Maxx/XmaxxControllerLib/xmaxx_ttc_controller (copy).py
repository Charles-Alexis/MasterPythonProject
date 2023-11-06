#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import time
import ast
import os
import copy
###############################################################################
import discretizer_xmaxx
import system_xmaxx
import costfunction_xmaxx
import baselinecontroller_xmaxx
import dynamicprogramming_xmaxx
import simulation_xmaxx
import controller_xmaxx
import cost2go_xmaxx
###############################################################################

class xmaxxttc_controller:
	def __init__(self, driver = 'Bad', road = 'CobblestoneWet', map_directory_name = '/14_12_2022_16h28'):
		self.ready_tu_use_flag = False
		#### GET VI VALUES
		self.current_dir = '/home/nvidia/XmaxxControllerLib'
		self.map_directory_name = map_directory_name
		self.map_directory = self.current_dir + '/xmaxx_policymap' + self.map_directory_name
		with open(self.map_directory + '/cf_config.txt','r') as f:
		     self.config_txt = f.readlines()		
		
		#### SETUPING SYS
		self.driver = driver
		self.road = road
		self.sys = system_xmaxx.LongitudinalFrontWheelDriveCarWithDriverModel()
		self.sys.roads =  ast.literal_eval(self.config_txt[3][0:-1])
		self.sys.driver_xmaxx =  ast.literal_eval(self.config_txt[4][0:-1])
		self.setup_sys()

		#### SETUPING GRIDSYS
		self.setup_gridsys()
		
		#### SETUPING COSTFUNTION
		self.cf = costfunction_xmaxx.DriverModelCostFunction.from_sys(self.sys)
		self.setup_cf()
		self.sys.cost_function = self.cf

		#### SETUPING DYNAMICAL PROG ALGO
		self.ttc_security_factor = 1/1
		self.ttc_name = np.array(['AsphalteDry','CementDry','AsphalteWet','CobblestoneWet','Snow','Ice'])
		self.ttc_distance = np.array([0.708, 0.802, 1.187, 2.88, 5.952, 21.739]) * self.ttc_security_factor
		self.ttc_distance = np.array([0.708, 0.802, 1.187, 1.187, 5.952, 21.739]) * self.ttc_security_factor
		self.ttc_slip = np.array([-0.143, -0.136, -0.112, -0.116, -0.052, -0.022])
		index = 0
		for i in range(len(self.ttc_name)):
			if road == self.ttc_name[i]:
				index = i

		self.ttc_controller = baselinecontroller_xmaxx.TTCController(self.sys, self.grid_sys, self.sys.human_model, ttc_ref=self.ttc_distance[index], position_obs=20, slip_cmd=self.ttc_slip[index])
		self.cl_sys_ttc = controller_xmaxx.ClosedLoopSystem(self.sys , self.ttc_controller)
        	self.cl_sys_ttc.cost_function = self.sys.cost_function 
		print('TTC ALGO READY TU USE')
		self.ready_to_use_flag = True


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
		self.sys.driver = self.sys.driver_xmaxx[self.driver]
		self.sys.obs_dist = self.sys.x_ub[0]

	def setup_cf(self):
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
		i=0
		self.cf.confort_coef = float(self.confort_coef)
		self.cf.override_coef = float(self.override_coef)
		self.cf.security_coef = float(self.security_coef)
		self.cf.xbar = np.array([0, 0])

	def setup_gridsys(self):
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
		self.grid_sys = discretizer_xmaxx.GridDynamicSystem(self.sys, (int(self.pos_dim), int(self.vit_dim)), (int(self.slip_dim), 2), self.dt_dim, lookup=False)

	def commands(self, pos, vit, acc):
		#TODO: MODIFIER LA POSITION RELATIVE ET VITESSE RELATIVE		
		pos = pos+1.0
		if vit < 0.:
			vit = 0	
		return self.cl_sys_ttc.controller.c(np.array([pos,vit]), acc)

	









