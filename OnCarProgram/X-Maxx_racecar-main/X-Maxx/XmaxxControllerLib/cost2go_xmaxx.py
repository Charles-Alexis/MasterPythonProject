

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:13:38 2022

@author: Charles-Alexis
"""

import numpy as np
import matplotlib.pyplot as plt
import time as t

class cost2go:
    def __init__(self, grid_sys, sys, closed_loop_system):
        # Dynamic system
        self.grid_sys = grid_sys
        self.sys = sys
        self.cl_sys = closed_loop_system
        
        self.debug_state_nbr = 0
        self.debug_plot = np.zeros(self.grid_sys.x_grid_dim)
        
        self.nbr_iter = 0
        self.delta_terminal_state = -1
        
        ### Define terminal states
        self.finals_conditions = np.array([20., 0.1])
        
        ### MAP des couts pour chaque etats
        self.j_map = np.zeros(self.grid_sys.x_grid_dim)
        
        ### MAP des commandes du Controleur
        self.command = np.zeros([self.grid_sys.x_grid_dim[0], self.grid_sys.x_grid_dim[1], 2])
        
        ### MAP des Cost2Go
        self.cost2go_map = np.zeros(self.grid_sys.x_grid_dim)
        
        ### MAP for cost2go computing
        self.states = np.zeros([self.grid_sys.x_grid_dim[0], self.grid_sys.x_grid_dim[1], 2])
        self.next_state = np.zeros([self.grid_sys.x_grid_dim[0], self.grid_sys.x_grid_dim[1], 2])
        self.terminal_state = np.zeros(self.grid_sys.x_grid_dim)
        
        self.init_j_map()
        
    def find_nearest(self, value, array):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    def init_j_map(self):
        for pos in range(self.grid_sys.x_grid_dim[0]):
            for vit in range(self.grid_sys.x_grid_dim[1]):
                ### Position and vitesse from grid
                position = self.grid_sys.x_level[0][pos]
                vitesse = self.grid_sys.x_level[1][vit]
                self.states[pos][vit] = np.array([position,vitesse])
                self.command[pos][vit] = self.cl_sys.controller.c(np.array([position,vitesse]),0)
                self.j_map[pos][vit] = self.cl_sys.cost_function.g(np.array([position, vitesse]), self.command[pos][vit], 0) * self.grid_sys.dt
                self.cost2go_map[pos][vit] = self.j_map[pos][vit]
                
                #Indicates if state is terminal
                if position >= self.finals_conditions[0]:
                    self.terminal_state[pos][vit] = 1
                if vitesse <= self.finals_conditions[1]:
                    self.terminal_state[pos][vit] = 1
                
                #Sets where the next state is for every states
                if self.terminal_state[pos][vit] == 1:
                    self.next_state[pos][vit] = np.array([int(pos),int(vit)])
                else:
                    x_k0 = self.sys.f(np.array([position,vitesse]), self.command[pos][vit]) 
                    x_k1 = np.array([position,vitesse]) + x_k0 * self.grid_sys.dt
                    self.next_state[pos][vit][0] = int(self.find_nearest(x_k1[0],self.grid_sys.x_level[0]))
                    self.next_state[pos][vit][1] = int(self.find_nearest(x_k1[1],self.grid_sys.x_level[1]))
                if self.next_state[pos][vit][0] == pos and self.next_state[pos][vit][1] == vit:
                    self.debug_state_nbr = self.debug_state_nbr + 1
                    self.terminal_state[pos][vit] = 1
                    
                if self.terminal_state[pos][vit] == 1:
                    self.cost2go_map[pos][vit] = self.cost2go_map[pos][vit] + self.cost2go_map[int(self.next_state[pos][vit][0])][int(self.next_state[pos][vit][1])] 
                        
    def compute_step(self):
        debut = np.sum(self.terminal_state)
        for pos in range(self.grid_sys.x_grid_dim[0]):
            for vit in range(self.grid_sys.x_grid_dim[1]):
                 if self.terminal_state[pos][vit] == 0 and self.terminal_state[int(self.next_state[pos][vit][0])][int(self.next_state[pos][vit][1])] == 1:
                      self.terminal_state[pos][vit] = 1
                      self.cost2go_map[pos][vit] = self.cost2go_map[pos][vit] + self.cost2go_map[int(self.next_state[pos][vit][0])][int(self.next_state[pos][vit][1])] 
        fin = np.sum(self.terminal_state)
        self.delta_terminal_state = fin-debut
        print('After iteration nbr: ' + str(self.nbr_iter) + ', there is: ' +str(self.delta_terminal_state)+' New states')

    def compute_steps(self):
        while self.delta_terminal_state != 0:
            self.compute_step()
            self.nbr_iter = self.nbr_iter + 1
            print('there is: ' + str(np.sum(self.terminal_state)) + '/' + str(np.size(self.terminal_state))+', ' +str(np.sum(self.terminal_state)/np.size(self.terminal_state) * 100)+ '%' )
           
    def plot_cost2go_map(self):              
        fig, axs = plt.subplots(1, 1)
        plt.ion()
        fig.suptitle('Cost2Go')
        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]  
        
        axs.set(xlabel=xname, ylabel=yname)
        i1 = axs.pcolormesh(self.grid_sys.x_level[0], self.grid_sys.x_level[1], self.cost2go_map.T, shading='gouraud')
        axs.axis([self.grid_sys.x_level[0][0], self.grid_sys.x_level[0][-1], self.grid_sys.x_level[1][0], self.grid_sys.x_level[1][-1]])
        fig.colorbar(i1, ax=axs)
        axs.grid(True)

        
    def plot_bugged_states(self):
        for pos in range(self.grid_sys.x_grid_dim[0]):
            for vit in range(self.grid_sys.x_grid_dim[1]):        
                if self.next_state[pos][vit][0] == pos and self.next_state[pos][vit][1] == vit:
                    self.debug_plot[pos][vit] = 1
        
        fig, axs = plt.subplots(1, 2)
        plt.ion()
        fig.suptitle('States where the next state is the same')
        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]  
        
        axs[0].set_title('Same state')
        axs[0].set(xlabel=xname, ylabel=yname)
        i1 = axs[0].pcolormesh(self.grid_sys.x_level[0], self.grid_sys.x_level[1], self.debug_plot.T, shading='gouraud')
        axs[0].axis([self.grid_sys.x_level[0][0], self.grid_sys.x_level[0][-1], self.grid_sys.x_level[1][0], self.grid_sys.x_level[1][-1]])
        fig.colorbar(i1, ax=axs[0])
        axs[0].grid(True)
        
        axs[1].set_title('Commands')
        axs[1].set(xlabel=xname, ylabel=yname)
        i2 = axs[1].pcolormesh(self.grid_sys.x_level[0], self.grid_sys.x_level[1], self.command[:,:,0].T, shading='gouraud')
        axs[1].axis([self.grid_sys.x_level[0][0], self.grid_sys.x_level[0][-1], self.grid_sys.x_level[1][0], self.grid_sys.x_level[1][-1]])
        fig.colorbar(i2, ax=axs[1])
        axs[1].grid(True)
        
class cost2go_list:
    def __init__(self, grid_sys, sys, closed_loop_system, cf_list):
        # Dynamic system
        self.grid_sys = grid_sys
        self.sys = sys
        self.cl_sys = closed_loop_system
        
        self.print_iter = True
        
        ##LISTING setup
        self.cf_list = cf_list
        self.cf_list_name = list(('g','confort','security','override'))
        self.j_map_list = list()
        self.cost2go_map_list = list()
        self.nbr_of_cf =  len(self.cf_list)
        for i in range(self.nbr_of_cf):
            self.j_map_list.append(np.zeros(self.grid_sys.x_grid_dim))
            self.cost2go_map_list.append(np.zeros(self.grid_sys.x_grid_dim))
        
        self.debug_state_nbr = 0
        self.debug_plot = np.zeros(self.grid_sys.x_grid_dim)
        
        self.nbr_iter = 0
        self.delta_terminal_state = -1
        
        ### Define terminal states
        self.finals_conditions = np.array([20., 0.1])
        
        ### MAP des commandes du Controleur
        self.command = np.zeros([self.grid_sys.x_grid_dim[0], self.grid_sys.x_grid_dim[1], 2])
        
        ### MAP for cost2go computing
        self.states = np.zeros([self.grid_sys.x_grid_dim[0], self.grid_sys.x_grid_dim[1], 2])
        self.next_state = np.zeros([self.grid_sys.x_grid_dim[0], self.grid_sys.x_grid_dim[1], 2])
        self.terminal_state = np.zeros(self.grid_sys.x_grid_dim)
        
        self.init_j_map()
        

    
    def init_j_map(self):
        for pos in range(self.grid_sys.x_grid_dim[0]):
            for vit in range(self.grid_sys.x_grid_dim[1]):
                ### Position and vitesse from grid
                position = self.grid_sys.x_level[0][pos]
                vitesse = self.grid_sys.x_level[1][vit]
                self.states[pos][vit] = np.array([position,vitesse])
                self.command[pos][vit] = self.cl_sys.controller.c(np.array([position,vitesse]),0)

                #Indicates if state is terminal
                if position >= self.finals_conditions[0]:
                    self.terminal_state[pos][vit] = 1
                if vitesse <= self.finals_conditions[1]:
                    self.terminal_state[pos][vit] = 1
     
                #Sets where the next state is for every states
                if self.terminal_state[pos][vit] == 1:
                    self.next_state[pos][vit] = np.array([int(pos),int(vit)])
                else:
                    x_k0 = self.sys.f(np.array([position,vitesse]), self.command[pos][vit]) 
                    x_k1 = np.array([position,vitesse]) + x_k0 * self.grid_sys.dt
                    self.next_state[pos][vit][0] = int(self.find_nearest(x_k1[0],self.grid_sys.x_level[0]))
                    self.next_state[pos][vit][1] = int(self.find_nearest(x_k1[1],self.grid_sys.x_level[1]))
                if self.next_state[pos][vit][0] == pos and self.next_state[pos][vit][1] == vit:
                    self.debug_state_nbr = self.debug_state_nbr + 1
                    self.terminal_state[pos][vit] = 1
                    
                ### LISTING through CF
                for i in range(self.nbr_of_cf):    
                    self.j_map_list[i][pos][vit] = self.cf_list[i](np.array([position, vitesse]), self.command[pos][vit], 0) * self.grid_sys.dt
                    self.cost2go_map_list[i][pos][vit] = self.j_map_list[i][pos][vit]    
                    
                    if self.terminal_state[pos][vit] == 1:
                        self.cost2go_map_list[i][pos][vit] = self.cost2go_map_list[i][pos][vit] + self.cost2go_map_list[i][int(self.next_state[pos][vit][0])][int(self.next_state[pos][vit][1])] 
                        
    def compute_step(self):
        debut = np.sum(self.terminal_state)
        for pos in range(self.grid_sys.x_grid_dim[0]):
            for vit in range(self.grid_sys.x_grid_dim[1]):
                 if self.terminal_state[pos][vit] == 0 and self.terminal_state[int(self.next_state[pos][vit][0])][int(self.next_state[pos][vit][1])] == 1:
                      self.terminal_state[pos][vit] = 1
                      for i in range(self.nbr_of_cf): 
                          self.cost2go_map_list[i][pos][vit] = self.cost2go_map_list[i][pos][vit] + self.cost2go_map_list[i][int(self.next_state[pos][vit][0])][int(self.next_state[pos][vit][1])] 
        
        fin = np.sum(self.terminal_state)
        self.delta_terminal_state = fin-debut
        if self.print_iter is True:
             print('After iteration nbr: ' + str(self.nbr_iter) + ', there is: ' +str(self.delta_terminal_state)+' New states')

    def compute_steps(self):
        print('there is: ' + str(np.sum(self.terminal_state)) + '/' + str(np.size(self.terminal_state))+', ' +str(np.sum(self.terminal_state)/np.size(self.terminal_state) * 100)+ '%' )

        while self.delta_terminal_state != 0:
            self.compute_step()
            self.nbr_iter = self.nbr_iter + 1
            if self.print_iter is True:
                 print('there is: ' + str(np.sum(self.terminal_state)) + '/' + str(np.size(self.terminal_state))+', ' +str(np.sum(self.terminal_state)/np.size(self.terminal_state) * 100)+ '%' )
           
    def plot_cost2go_map(self):              
        fig, axs = plt.subplots(1, self.nbr_of_cf)
        plt.ion()
        fig.suptitle('Cost2Go')
        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]  
        for i in range(self.nbr_of_cf):
            #axs[i].set(xlabel=xname, ylabel=yname)
            i1 = axs[i].pcolormesh(self.grid_sys.x_level[0], self.grid_sys.x_level[1], self.cost2go_map_list[i].T, shading='gouraud')
            axs[i].axis([self.grid_sys.x_level[0][0], self.grid_sys.x_level[0][-1], self.grid_sys.x_level[1][0], self.grid_sys.x_level[1][-1]])
            fig.colorbar(i1, ax=axs[i])
            axs[i].grid(True)

    def plot_bugged_states(self):
        for pos in range(self.grid_sys.x_grid_dim[0]):
            for vit in range(self.grid_sys.x_grid_dim[1]):        
                if self.next_state[pos][vit][0] == pos and self.next_state[pos][vit][1] == vit:
                    self.debug_plot[pos][vit] = 1
        
        fig, axs = plt.subplots(1, 2)
        plt.ion()
        fig.suptitle('States where the next state is the same')
        xname = self.sys.state_label[0] + ' ' + self.sys.state_units[0]
        yname = self.sys.state_label[1] + ' ' + self.sys.state_units[1]  
        
        axs[0].set_title('Same state')
        axs[0].set(xlabel=xname, ylabel=yname)
        i1 = axs[0].pcolormesh(self.grid_sys.x_level[0], self.grid_sys.x_level[1], self.debug_plot.T, shading='gouraud')
        axs[0].axis([self.grid_sys.x_level[0][0], self.grid_sys.x_level[0][-1], self.grid_sys.x_level[1][0], self.grid_sys.x_level[1][-1]])
        fig.colorbar(i1, ax=axs[0])
        axs[0].grid(True)
        
        axs[1].set_title('Commands')
        axs[1].set(xlabel=xname, ylabel=yname)
        i2 = axs[1].pcolormesh(self.grid_sys.x_level[0], self.grid_sys.x_level[1], self.command[:,:,0].T, shading='gouraud')
        axs[1].axis([self.grid_sys.x_level[0][0], self.grid_sys.x_level[0][-1], self.grid_sys.x_level[1][0], self.grid_sys.x_level[1][-1]])
        fig.colorbar(i2, ax=axs[1])
        axs[1].grid(True)
        
    def find_nearest(self, value, array):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
