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
	def __init__(self, map_directory_name = '/20_1_2023_17h5', road = 'CobblestoneWet', driver= '1'):
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
		self.nom_control = list()
		self.roads_array = list()
		for r in self.sys.roads: 
			self.roads_array.append(r)
		print(self.roads_array)
		print(self.sys.driver_xmaxx)
		#### SETUPING COSTFUNTION
		self.cf = costfunction_xmaxx.DriverModelCostFunction.from_sys(self.sys)
		self.setup_cf()
		self.sys.cost_function = self.cf

		#### SETUPING GRIDSYS
		self.grid_syss = list()
		self.vi_u = list()
		self.vi_ctl = list()
		self.setup_gridsys()

		self.index = 0
		self.change_driver_road(self.driver, self.road)

		#### SETUPING DYNAMICAL PROG ALGO
		print('VI ALGO READY TU USE')
		self.ready_to_use_flag = True

	def change_driver_road(self,driver,road):
		self.driver = driver
		self.road = road
		
		name = road +'_'+driver
		for index_temp in range(len(self.nom_control)):
			if name == self.nom_control[index_temp]:
				self.index = index_temp
		print(self.index)

	def setup_sys(self):
		self.sys.lenght = 0.48
		self.sys.xc = 0.24
		self.sys.yc = 0.15
		self.sys.mass = 20
		self.sys.cdA = 0.3 * 0.105
		self.sys.x_ub = np.array([+0.0, 4.5])
		self.sys.x_lb = np.array([-10, 0])
		self.sys.u_ub = np.array([0.0, 1.0])
		self.sys.u_lb = np.array([-0.3, 0])
		self.sys.driver = self.sys.driver_xmaxx[self.driver]
		self.sys.obs_dist = self.sys.x_ub[0]
		self.compute_drivers_distances()

	def compute_drivers_distances(self):
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
		for d in self.sys.driver_xmaxx:
		    for r in self.roads_array:
		        name = r + '_' + d
		        self.nom_control.append(name)
		        self.sys.road = self.sys.roads[r]
		        
		        self.grid_syss.append(self.grid_sys)
		        dp = dynamicprogramming_xmaxx.DynamicProgrammingWithLookUpTable(self.grid_sys, self.cf, compute_cost = False)
		        dp.load_J_next(self.map_directory+'/'+'xmaxx_'+name)
		        self.vi_u.append(dp)

			self.ctl = dynamicprogramming_xmaxx.LookUpTableController(self.grid_sys, dp.pi)
			self.ctl.k = 2
			self.cl_sys_vi = controller_xmaxx.ClosedLoopSystem(self.sys, self.ctl)
			self.cl_sys_vi.cost_function = self.sys.cost_function
			self.vi_ctl.append(self.cl_sys_vi)
		print(self.nom_control)


	def commands(self, pos, vit):		
		pos = 0.0 - pos
		return self.vi_ctl[self.index].controller.c(np.array([pos,vit]),0)
		

	









