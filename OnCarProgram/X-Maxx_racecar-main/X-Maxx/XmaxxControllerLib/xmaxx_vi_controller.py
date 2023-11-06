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

class xmaxxvi_controller:
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
                self.dp = dynamicprogramming_xmaxx.DynamicProgrammingWithLookUpTable(self.grid_sys, self.cf, compute_cost=False)
		self.name = self.road + '_' + self.driver
                self.dp.load_J_next(self.map_directory+'/xmaxx_'+self.name) 
		self.ctl = dynamicprogramming_xmaxx.LookUpTableController(self.grid_sys, self.dp.pi)
		self.ctl.k = 2
		self.cl_sys_vi = controller_xmaxx.ClosedLoopSystem(self.sys, self.ctl)
		self.cl_sys_vi.cost_function = self.sys.cost_function
		print('VI ALGO READY TU USE')
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

	def commands(self, pos, vit):
		#TODO: MODIFIER LA POSITION RELATIVE ET VITESSE RELATIVE		
		pos = 20.0-pos+1.0
		return self.cl_sys_vi.controller.c(np.array([pos,vit]),0)

	









