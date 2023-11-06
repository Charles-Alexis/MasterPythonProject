

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:13:38 2022

@author: Charles-Alexis
"""

import numpy as np
import matplotlib.pyplot as plt
import time as t
import scipy.interpolate as inter

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
        
        ### MAP des coûts pour chaque états
        self.j_map = np.zeros(self.grid_sys.x_grid_dim)
        
        ### MAP des commandes du Contrôleur
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
    def __init__(self, grid_sys, sys, closed_loop_system, cf_list, cf_list_name = list(('g','confort','security','override'))):
        # Dynamic system
        self.grid_sys = grid_sys
        self.sys = sys
        self.cl_sys = closed_loop_system
        
        self.print_iter = True
        
        ##LISTING setup
        self.cf_list = cf_list
        self.cf_list_name = cf_list_name
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
        self.finals_conditions = np.array([0, 0])
        
        ### MAP des commandes du Contrôleur
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
                    self.j_map_list[i][pos][vit] = self.cf_list[i](np.array([position, vitesse]), self.command[pos][vit], 0) #* self.grid_sys.dt
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

    def compute_steps(self, print_iteration=True):
        print('there is: ' + str(np.sum(self.terminal_state)) + '/' + str(np.size(self.terminal_state))+', ' +str(np.sum(self.terminal_state)/np.size(self.terminal_state) * 100)+ '%' )

        if print_iteration is True:
            self.print_iter = True
        else:
            self.print_iter = False

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
            axs[i].set_title(self.cf_list_name[i])
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
    
class cost2go_esperance:
    def __init__(self, grid_sys, sys, E, closed_loop_system, cf_list, cf_list_name = list(('g','confort','security','override'))):
        # Dynamic system
        self.grid_sys = grid_sys
        self.sys = sys
        self.cl_sys = closed_loop_system
        
        self.states = grid_sys.x_level
        self.dt = grid_sys.dt
        self.E = E
        self.x_next = np.zeros([len(self.states[0]), len(self.states[1]), len(self.E)])
        self.command = np.zeros([self.grid_sys.x_grid_dim[0], self.grid_sys.x_grid_dim[1], 2])
        self.next_state = np.zeros([self.grid_sys.x_grid_dim[0], self.grid_sys.x_grid_dim[1], len(self.E), 2])
        self.terminal_state = np.zeros([len(self.states[0]), len(self.states[1]), len(self.E)])
        
        self.print_iter = True
        
        ##LISTING setup
        self.cf_list = cf_list
        self.cf_list_name = cf_list_name
        
        self.j_map_list = np.zeros([len(self.cf_list), len(self.states[0]), len(self.states[1]), len(self.E)])
        self.cost2go_map_list = np.zeros([len(self.cf_list), len(self.states[0]), len(self.states[1]), len(self.E)])
        
        self.debug_state_nbr = 0
        self.debug_plot = np.zeros(self.grid_sys.x_grid_dim)
        
        self.nbr_iter = 0
        self.delta_terminal_state = -1
        
        ### Define terminal states
        self.finals_conditions = np.array([20., 0.1])

        
    def create_x_next(self):
        for e in range(len(self.E)):
            for pos in range(len(self.states[0])):
                for vit in range(len(self.states[1])):
                    ### Position and vitesse from grid
                    position = self.states[0][pos]
                    vitesse = self.states[1][vit]
                    
                    self.command[pos][vit] = self.cl_sys.controller.c([position,vitesse],0)
    
                    #Indicates if state is terminal
                    if position >= self.finals_conditions[0]:
                        self.terminal_state[pos][vit][e] = 1
                    if vitesse <= self.finals_conditions[1]:
                        self.terminal_state[pos][vit][e] = 1
         
                    #Sets where the next state is for every states
                    if self.terminal_state[pos][vit][e] == 1:
                        self.next_state[pos][vit][e] = np.array([int(pos),int(vit)])
                    else:
                        x_k0 = self.sys.f([position,vitesse], self.command[pos][vit]) 
                        x_k1 = np.array([position,vitesse]) + x_k0 * self.dt
                        self.next_state[pos][vit][e][0] = int(self.find_nearest(x_k1[0],self.states[0]))
                        self.next_state[pos][vit][e][1] = int(self.find_nearest(x_k1[1],self.states[1]))
                    if self.next_state[pos][vit][e][0] == pos and self.next_state[pos][vit][e][1] == vit:
                        self.debug_state_nbr = self.debug_state_nbr + 1
                        self.terminal_state[pos][vit][e] = 1
                        
                    ### LISTING through CF
                    for i in range(len(self.cf_list)):    
                        self.j_map_list[i][pos][vit][e] = self.cf_list[i]([position, vitesse], self.command[pos][vit], 0) * self.dt
                        self.cost2go_map_list[i][pos][vit][e] = self.j_map_list[i][pos][vit][e]                   
                        if self.terminal_state[pos][vit][e] == 1:
                            self.cost2go_map_list[i][pos][vit][e] = self.cost2go_map_list[i][pos][vit][e] + self.cost2go_map_list[i][int(self.next_state[pos][vit][e][0])][int(self.next_state[pos][vit][e][1])][e] 

        
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

    def compute_steps(self, print_iteration=True):
        print('there is: ' + str(np.sum(self.terminal_state)) + '/' + str(np.size(self.terminal_state))+', ' +str(np.sum(self.terminal_state)/np.size(self.terminal_state) * 100)+ '%' )

        if print_iteration is True:
            self.print_iter = True
        else:
            self.print_iter = False

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
            axs[i].set_title(self.cf_list_name[i])
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
    
class cost2go_list_2:
    def __init__(self, grid_sys, sys, cf, command_law):
        # Dynamic system
        self.grid_sys = grid_sys
        self.sys = sys
        self.cf = cf
        self.command_law = command_law
        self.dim = self.grid_sys.x_grid_dim
        self.print_iter = True
        self.position = self.grid_sys.x_level[0]
        self.vitesse = self.grid_sys.x_level[1]
        
        ### INTERPOLATION
        self.levels = tuple(self.grid_sys.x_level[i] for i in range(self.grid_sys.sys.n))
        self.inter_total = np.nan
        self.inter_override = np.nan
        self.inter_confort = np.nan
        self.inter_security = np.nan
        
        ### Define terminal states
        self.finals_conditions = np.array([0, 0])
        self.total_state = self.dim[0]*self.dim[1]*len(self.sys.driver[0])
        ### MAP des commandes du Contrôleur
        self.command = np.zeros([self.dim[0], self.dim[1], 2])
        
        
        ### STUPID STATE
        self.max_diff = 1000
        self.max_diff_last = 10000
        self.flag_done = False
        self.iter_max = 10000
        
        # J
        self.J_total = np.zeros([self.dim[0], self.dim[1]])
        self.J_override = np.zeros([self.dim[0], self.dim[1]])
        self.J_confort = np.zeros([self.dim[0], self.dim[1]])
        self.J_security = np.zeros([self.dim[0], self.dim[1]])
       
        self.J0_total = np.zeros([self.dim[0], self.dim[1]])
        self.J0_override = np.zeros([self.dim[0], self.dim[1]])
        self.J0_confort = np.zeros([self.dim[0], self.dim[1]])
        self.J0_security = np.zeros([self.dim[0], self.dim[1]])
       
        self.g_total = np.zeros([self.dim[0], self.dim[1]])
        self.g_override = np.zeros([self.dim[0], self.dim[1]])
        self.g_confort = np.zeros([self.dim[0], self.dim[1]])
        self.g_security = np.zeros([self.dim[0], self.dim[1]])
        
        #self.J_confort_list = list()
        
        ### FUNCTION NO ESP      
        if len(self.sys.driver[0]) == 1:
            self.next_state = np.zeros([self.dim[0], self.dim[1], 2])
            self.init_command_and_next_state()
            self.compute_steps()
        else:
            self.next_state = np.zeros([self.dim[0], self.dim[1], 2, len(self.sys.driver[0])])
            ### FUNCTION WITH ESP
            self.init_command_and_next_state_e()
            self.compute_steps_e()
    
        
        
    def init_command_and_next_state(self):
         # print('Initialisation des commandes et des états futures')
         i = 0
         time_debut = t.time()
         for pos_index in range(len(self.position)):
              for vit_index in range(len(self.vitesse)):
                   c = self.command_law([self.position[pos_index], self.vitesse[vit_index]],0)
              
                   self.command[pos_index][vit_index][0] = c[0]
                   self.command[pos_index][vit_index][1] = c[1]
                   
                   dfx1 = self.sys.f([self.position[pos_index],self.vitesse[vit_index]],c,0)
                   fx_pos = self.position[pos_index] + dfx1[0]*self.grid_sys.dt
                   fx_vit = self.vitesse[vit_index] + dfx1[1]*self.grid_sys.dt
                   
                   if fx_pos >= 0:
                        fx_pos = 0
                   if fx_vit < self.finals_conditions[1]:
                        fx_vit = 0
                   
                   self.next_state[pos_index][vit_index][0] = fx_pos
                   self.next_state[pos_index][vit_index][1] = fx_vit
                   
                   self.g_total[pos_index][vit_index] = self.cf.g([self.position[pos_index], self.vitesse[vit_index]], c, 0) * self.grid_sys.dt
                   self.g_override[pos_index][vit_index] = self.cf.g_override([self.position[pos_index], self.vitesse[vit_index]], c, 0) * self.grid_sys.dt
                   self.g_confort[pos_index][vit_index] = self.cf.g_confort([self.position[pos_index], self.vitesse[vit_index]], c, 0) * self.grid_sys.dt
                   self.g_security[pos_index][vit_index] = self.cf.g_security([self.position[pos_index], self.vitesse[vit_index]], c, 0) * self.grid_sys.dt
                   
                   self.J0_total[pos_index][vit_index] = np.copy(self.g_total[pos_index][vit_index])
                   self.J0_override[pos_index][vit_index] = np.copy(self.g_override[pos_index][vit_index])
                   self.J0_confort[pos_index][vit_index] = np.copy(self.g_confort[pos_index][vit_index])
                   self.J0_security[pos_index][vit_index] = np.copy(self.g_security[pos_index][vit_index])
                   
                   i = i+1
                   if i%10000 == 0:
                        time_now = t.time()
                        # print(str(i)+'/'+ str(self.total_state)+ ' in ' + str(time_now-time_debut)+'sec')

    def create_interpole(self):
        self.inter_total = inter.RegularGridInterpolator(self.levels, self.J0_total, 'linear', True, fill_value=np.nan)
        self.inter_override = inter.RegularGridInterpolator(self.levels, self.J0_override, 'linear', True, fill_value=np.nan)
        self.inter_confort = inter.RegularGridInterpolator(self.levels, self.J0_confort, 'linear', True, fill_value=np.nan)
        self.inter_security = inter.RegularGridInterpolator(self.levels, self.J0_security, 'linear', True, fill_value=np.nan)
                        
    def compute_step(self):
        
        # print(np.shape(self.next_state))
        # print(self.levels)
        self.J_total = self.g_total + self.inter_total(self.next_state)
        self.J_security = self.g_security  + self.inter_security (self.next_state)
        self.J_confort = self.g_confort + self.inter_confort(self.next_state)
        self.J_override = self.g_override + self.inter_override(self.next_state)
        
        # if self.max_diff == np.max(self.J_total - self.J0_total):
        #     self.flag_done = True

        if np.max(self.J_total - self.J0_total) < self.max_diff:
            self.max_diff = np.max(self.J_total - self.J0_total)
            #print('Diff Max: ', self.max_diff)
            
        if self.max_diff < 0.005:
            self.flag_done = True
        
        self.J0_total = np.copy(self.J_total)
        self.J0_security = np.copy(self.J_security)
        self.J0_confort = np.copy(self.J_confort)
        self.J0_override = np.copy(self.J_override)
        #self.J_confort_list.append(self.J0_confort)
        
    def compute_steps(self):
        time_debut = t.time()
        i = 0
        while self.flag_done is False and i<self.iter_max:
            self.create_interpole()
            self.compute_step()
            i = i+1
            if i %1000 ==0:
                time_now = t.time()
                print(str(i) + ' in ' + str(time_now-time_debut)+' and max diff: ' + str(self.max_diff))
                
        time_now = t.time()
        print(str(i) + ' in ' + str(time_now-time_debut)+' and max diff: ' + str(self.max_diff))

    def init_command_and_next_state_e(self):
         print('Initialisation des commandes et des états futures')
         i = 0
         time_debut = t.time()
         for pos_index in range(len(self.position)):
              for vit_index in range(len(self.vitesse)):
                   ### COMMAND
                   c = self.command_law([self.position[pos_index], self.vitesse[vit_index]],0)       
                   self.command[pos_index][vit_index][0] = c[0]
                   self.command[pos_index][vit_index][1] = c[1]
                   
                   ### G COST
                   self.g_total[pos_index][vit_index] = self.cf.g([self.position[pos_index], self.vitesse[vit_index]], c, 0) * self.grid_sys.dt
                   self.g_override[pos_index][vit_index] = self.cf.g_override([self.position[pos_index], self.vitesse[vit_index]], c, 0) * self.grid_sys.dt
                   self.g_confort[pos_index][vit_index] = self.cf.g_confort([self.position[pos_index], self.vitesse[vit_index]], c, 0) * self.grid_sys.dt
                   self.g_security[pos_index][vit_index] = self.cf.g_security([self.position[pos_index], self.vitesse[vit_index]], c, 0) * self.grid_sys.dt
                   
                   self.J0_total[pos_index][vit_index] = np.copy(self.g_total[pos_index][vit_index])
                   self.J0_override[pos_index][vit_index] = np.copy(self.g_total[pos_index][vit_index])
                   self.J0_confort[pos_index][vit_index] = np.copy(self.g_total[pos_index][vit_index])
                   self.J0_security[pos_index][vit_index] = np.copy(self.g_total[pos_index][vit_index])
                   
                   index_esp = 0
                   for esp in self.sys.driver[0]:
                       dfx1 = self.sys.f([self.position[pos_index],self.vitesse[vit_index]],c, e = esp[0])
                       fx_pos = self.position[pos_index] + dfx1[0]*self.grid_sys.dt
                       fx_vit = self.vitesse[vit_index] + dfx1[1]*self.grid_sys.dt
                       
                       if fx_pos > self.finals_conditions[0]:
                            fx_pos = 0
                       if fx_vit < self.finals_conditions[1]:
                            fx_vit = 0
            
                       self.next_state[pos_index][vit_index][0][index_esp] = fx_pos
                       self.next_state[pos_index][vit_index][1][index_esp] = fx_vit
                       
                       index_esp = index_esp + 1
                   
                       i = i+1
                       if i%10000 == 0:
                            time_now = t.time()
                            #print(str(i)+'/'+ str(self.total_state)+ ' in ' + str(time_now-time_debut)+'sec')

    def create_interpole_e(self):
        self.inter_total = list()
        self.inter_override = list()
        self.inter_confort = list()
        self.inter_security = list()
        for esp_index in range(len(self.sys.driver[0])):
            self.inter_total.append(inter.RegularGridInterpolator(self.levels, self.J0_total, 'linear', True, fill_value=np.nan))
            self.inter_override.append(inter.RegularGridInterpolator(self.levels, self.J0_override, 'linear', True, fill_value=np.nan))
            self.inter_confort.append(inter.RegularGridInterpolator(self.levels, self.J0_confort, 'linear', True, fill_value=np.nan))
            self.inter_security.append(inter.RegularGridInterpolator(self.levels, self.J0_security, 'linear', True, fill_value=np.nan))
                        
    def compute_step_e(self):
        esp_index = 0
        self.J_total = np.zeros([self.dim[0], self.dim[1]])
        self.J_override = np.zeros([self.dim[0], self.dim[1]])
        self.J_confort = np.zeros([self.dim[0], self.dim[1]])
        self.J_security = np.zeros([self.dim[0], self.dim[1]])
        
        for esp in self.sys.driver[0]:
            self.J_total = self.J_total             + (self.g_total      + self.inter_total[esp_index](self.next_state[:,:,:,esp_index])) * esp[1]
            self.J_security = self.J_security       + (self.g_security   + self.inter_security[esp_index](self.next_state[:,:,:,esp_index])) * esp[1]
            self.J_confort = self.J_confort         + (self.g_confort    + self.inter_confort[esp_index](self.next_state[:,:,:,esp_index])) * esp[1]
            self.J_override = self.J_override       + (self.g_override   + self.inter_override[esp_index](self.next_state[:,:,:,esp_index])) * esp[1]
            esp_index = esp_index+1
        
        if self.max_diff == np.max(self.J_total - self.J0_total):
            self.flag_done = True

        if np.max(self.J_total - self.J0_total) < self.max_diff:
            self.max_diff = np.max(self.J_total - self.J0_total)
            #print('Diff Max: ', self.max_diff)
            
        if self.max_diff < 0.005:
            self.flag_done = True
        
        self.J0_total = np.copy(self.J_total)
        self.J0_security = np.copy(self.J_security)
        self.J0_confort = np.copy(self.J_confort)
        self.J0_override = np.copy(self.J_override)   
    
    def compute_steps_e(self):
        time_debut = t.time()
        i = 0
        while self.flag_done is False and i<self.iter_max:
            self.create_interpole_e()
            self.compute_step_e()
            i = i+1
            if i %1000 ==0:
                time_now = t.time()
                print(str(i) + ' in ' + str(time_now-time_debut)+' and max diff: ' + str(self.max_diff))
                
        time_now = t.time()
        print(str(i) + ' in ' + str(time_now-time_debut)+' and max diff: ' + str(self.max_diff))   