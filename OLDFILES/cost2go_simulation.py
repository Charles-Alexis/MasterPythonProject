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
        
    ##############################
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    def initialize(self):
        """ initialize cost-to-go and policy """
        # Initial evaluation
        self.states = np.zeros([len(self.xstate0),len(self.xstate1),4])
        for x0 in range(self.grid_sys.xgriddim[0]):
          for x1 in range(self.grid_sys.xgriddim[1]):
            self.states[x0][x1][0] = self.xstate0[x0]
            self.states[x0][x1][1] = self.xstate1[x1]
            self.states[x0][x1][2] = x0
            self.states[x0][x1][3] = x1
        i = 0
        for x0 in range(self.grid_sys.xgriddim[0]):
          for x1 in range(self.grid_sys.xgriddim[1]):
            x_ind = np.array([x0,x1])
            x = self.states[x0][x1][0:2]
            self.action_policy[x0][x1] = self.cl_sys.controller.c(x,0)
            self.J[x0][x1] = self.cl_sys.cost_function.h(x)+self.cl_sys.cost_function.g(x, self.action_policy[x0][x1], 0, 0)
            
            self.g1[x0][x1] = self.cl_sys.cost_function.h(x)+self.cl_sys.cost_function.g_confort(x, self.action_policy[x0][x1], 0, 0)
            self.g2[x0][x1] = self.cl_sys.cost_function.h(x)+self.cl_sys.cost_function.g_override(x, self.action_policy[x0][x1], 0, 0)
            self.g3[x0][x1] = self.cl_sys.cost_function.h(x)+self.cl_sys.cost_function.g_security(x, self.action_policy[x0][x1], 0, 0)
            
            result = x + self.cl_sys.f(x,self.action_policy[x0][x1],0)*self.grid_sys.dt
            self.next_state[x0][x1] = [ int(self.find_nearest(self.xstate0, result[0])), int(self.find_nearest(self.xstate1, result[1]))]
            
            dx_ind = self.next_state[x0][x1]
            if x[1] <= 0 or x[0] >= 100:
                self.end_state[x0][x1] = 1
            if (x_ind[0] == dx_ind[0]) and (x_ind[1] == dx_ind[1]):
                i = i +1
                self.end_state[x0][x1] = 1
        print('Nombre de state brisé: ' + str(i) +' Pourcentage= ' + str(100*i/(self.grid_sys.xgriddim[0]*self.grid_sys.xgriddim[1])) + '%')
        self.next_state = self.next_state.astype(int)

        
    def compute_step(self):
      end_temp = np.copy(self.end_state)
      J_temp = np.copy(self.J)
      
      J_temp1 = np.copy(self.g1)
      J_temp2 = np.copy(self.g2)
      J_temp3 = np.copy(self.g3)
      
      for x0 in range(self.grid_sys.xgriddim[0]):
          for x1 in range(self.grid_sys.xgriddim[1]):
            if self.end_state[x0][x1] == 0:
              x_ind = np.array([x0,x1])
              x = self.states[x0][x1][0:2]
              dx_ind = self.next_state[x0][x1]
              
              if (x_ind[0] == dx_ind[0]) and (x_ind[1] == dx_ind[1]):
                print('ERREUR DANS LA MATRIx capitain at state:', x)
            
              if self.end_state[dx_ind[0]][dx_ind[1]] == 1:
                J_temp[x0][x1] = J_temp[x0][x1] + J_temp[dx_ind[0]][dx_ind[1]]
                J_temp1[x0][x1] = J_temp1[x0][x1] + J_temp1[dx_ind[0]][dx_ind[1]]
                J_temp2[x0][x1] = J_temp2[x0][x1] + J_temp2[dx_ind[0]][dx_ind[1]]
                J_temp3[x0][x1] = J_temp3[x0][x1] + J_temp3[dx_ind[0]][dx_ind[1]]
                end_temp[x0][x1] = 1
                
      self.end_state = np.copy(end_temp)
      self.J = np.copy(J_temp)
      self.g1 = np.copy(J_temp1)
      self.g2 = np.copy(J_temp2)
      self.g3 = np.copy(J_temp3)
                    
    def compute_steps(self):
        i = 0
        end_start = sum(sum(self.end_state))-1
        end_stop = sum(sum(self.end_state))
        print(end_stop)
        while (end_stop - end_start) != 0:
            end_start = end_stop
            self.compute_step()
            i = i+1
            end_stop = sum(sum(self.end_state))
            print('nombre detat terminaux: ' + str(end_stop) + ' Pourcentage:' + str(end_stop*100/(self.grid_sys.xgriddim[0]*self.grid_sys.xgriddim[1])))
        self.J = self.J * self.grid_sys.dt
        plt.figure()
        plt.imshow(self.J.transpose(),origin='lower', aspect=(self.grid_sys.xgriddim[1]/self.grid_sys.xgriddim[0]))
        plt.colorbar()
        
        
        fig, axs = plt.subplots(2, 2)
        axs[0,0].set_title('Coûts-à-venir')
        axs[0,0].imshow(self.J.transpose(),origin='lower', aspect=(self.grid_sys.xgriddim[0]/self.grid_sys.xgriddim[1]))

        axs[0,1].set_title('Coûts-à-venir Confort')
        axs[0,1].imshow(self.g1.transpose(),origin='lower', aspect=(self.grid_sys.xgriddim[0]/self.grid_sys.xgriddim[1]))
        
        axs[1,0].set_title('Coûts-à-venir Override')
        axs[1,0].imshow(self.g2.transpose(),origin='lower', aspect=(self.grid_sys.xgriddim[0]/self.grid_sys.xgriddim[1]))            
        
        axs[1,1].set_title('Coûts-à-venir Security')
        axs[1,1].imshow(self.g3.transpose(),origin='lower', aspect=(self.grid_sys.xgriddim[0]/self.grid_sys.xgriddim[1]))            
        