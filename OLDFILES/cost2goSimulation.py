#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:31:36 2022

@author: clearpath-robot
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:13:38 2022

@author: Charles-Alexis
"""

import numpy as np
import matplotlib.pyplot as plt
import time as t
import simulationv2 as s

class cost2go_simulation:
    """ Dynamic programming for continuous dynamic system """

    ############################
    def __init__(self, grid_sys, closed_loop_system):

        # Dynamic system
        self.grid_sys = grid_sys  # Discretized Dynamic system class
        self.sys = grid_sys.sys  # Base Dynamic system class
        self.cl_sys = closed_loop_system

        # initializes nb of dimensions and continuous inputs u
        self.n_dim = self.sys.n

        # Options
        self.target = np.array([0,0])
        
        #important matrix
        self.J = np.zeros(self.grid_sys.xgriddim, dtype=float)
        self.action_policy = np.zeros([self.grid_sys.xgriddim[0],self.grid_sys.xgriddim[1],self.grid_sys.ugriddim[1]])
        self.state = np.zeros([self.grid_sys.xgriddim[0],self.grid_sys.xgriddim[1],self.cl_sys.m])
        self.next_state = np.zeros([self.grid_sys.xgriddim[0],self.grid_sys.xgriddim[1],self.grid_sys.ugriddim[1]])
        self.cost2go = np.zeros([self.grid_sys.xgriddim[0],self.grid_sys.xgriddim[1]])
        self.g1 = np.zeros([self.grid_sys.xgriddim[0],self.grid_sys.xgriddim[1]])
        self.g2 = np.zeros([self.grid_sys.xgriddim[0],self.grid_sys.xgriddim[1]])
        self.g3 = np.zeros([self.grid_sys.xgriddim[0],self.grid_sys.xgriddim[1]])
        self.end_state = np.zeros([self.grid_sys.xgriddim[0],self.grid_sys.xgriddim[1]])
        
        self.xstate0 = np.arange(self.cl_sys.x_lb[0],self.cl_sys.x_ub[0]+0.1,(self.cl_sys.x_ub[0]-self.cl_sys.x_lb[0])/(self.grid_sys.xgriddim[0]-1))
        self.xstate1 = np.arange(self.cl_sys.x_lb[1],self.cl_sys.x_ub[1]+0.1,(self.cl_sys.x_ub[1]-self.cl_sys.x_lb[1])/(self.grid_sys.xgriddim[1]-1))
        
        self.initialize()

    def initialize(self):       
        self.states = self.grid_sys.nodes_state
        self.j_ttc = np.zeros([self.grid_sys.xgriddim[0]*self.grid_sys.xgriddim[1],5])
 
    def compute_steps(self):
        ind = 0
        for state in self.states:
            self.cl_sys.x0 = state
            self.j_ttc[ind][0] = self.cl_sys.x0[0]
            self.j_ttc[ind][1] = self.cl_sys.x0[1]
            sim = s.SimulatorV2(self.cl_sys)
            temp = sim.compute()
            self.j_ttc[ind][2] = temp.J[-1] # J
            ind = ind+1
            print(ind/len(self.states)*100)

#        pos = np.arange(0,100.1,101/81)
#        vit = np.arange(0,20.1,0.25)
#        
#        i=0 # pos
#        j=0 # vit
#        j_map = np.zeros([81,81])
#        for i in range(81):
#            for j in range(81):
#                j_map[i][j] = j_ttc[i*81+j][2]#/(j_ttc[i*81+j][3]/j_ttc[i*81+j][4])
#                j = j+1
#            i = i+1
#            j = 0
#        
#        plt.figure()
#        #plt.contourf(pos,vit,j_map, 1000)
#        plt.imshow(j_map.transpose(), origin='lower')
#        plt.colorbar()
        
