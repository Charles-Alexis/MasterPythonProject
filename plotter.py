# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:29:11 2023

@author: Charles-Alexis
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import date
import datetime

class plotter():
    def __init__(self, metadata, controler, closedloop, cost2go, plot_cost_to_go = False, plot_cost_to_go_all = False):
        self.metadata = metadata
        self.controler = controler
        self.closedloop = closedloop
        self.cost2go = cost2go
        self.plot_cost_to_go = plot_cost_to_go
        self.plot_cost_to_go_all = plot_cost_to_go_all
        self.font= {'family': 'sans-serif',
                    'color':  'black',
                    'weight': 'normal',
                    'size': 10,
        }
        
        self.nbr_roads = len(self.metadata[0])
        self.nbr_drivers = len(self.metadata[1])
        self.nbr_controlers = len(self.metadata[2])
        self.nbr_coefs = len(self.metadata[3])
        
        self.roads_array = ['Asphalte Sec','Ciment Sec','Asphalte Mouillée','Gravier Sec','Gravier Mouillé','Neige','Glace']
        self.dec_array = ['-8.55m/s^2','-7.77m/s^2','-5.28m/s^2','-6.68m/s^2','-2.38m/s^2','-1.29m/s^2','-0.59m/s^2']
        self.mu_array = ['1.17','1.09','0.8','0.97','0.38','0.16','0.05']
        self.drivers_array = ['Bon','Normale','Mauvais','Endormi','Pas d\'espérence']
        self.controlers_array = ['Itération par valeurs','Temps de collision','Distance de frein minimale', 'Modèle Humain']
        
        self.grid = self.metadata[-1][0].x_grid_dim
        
        self.data_array = np.zeros([7, 5, 4, self.grid[0], self.grid[1]])
        self.cost_array = np.zeros([7, 5, 4, self.grid[0], self.grid[1]])
        self.cost_array_override = np.zeros([7, 5, 4, self.grid[0], self.grid[1]])
        self.cost_array_security = np.zeros([7, 5, 4, self.grid[0], self.grid[1]])
        self.cost_array_confort = np.zeros([7, 5, 4, self.grid[0], self.grid[1]])
        self.create_data_array()
        self.create_cost_array()
        
        #self.plotting()
        if self.plot_cost_to_go:
            self.plotting_cost2go()
        if self.plot_cost_to_go_all:
            self.plotting_cost2go_all()
        
    def plot_function(self, fig, axs, grid, data, name= None):
        i = axs.pcolormesh(grid.x_level[0], grid.x_level[1], data, shading='gouraud', cmap = 'plasma')
        axs.axis([grid.x_level[0][0], grid.x_level[0][-1], grid.x_level[1][0], grid.x_level[1][-1]])
        fig.colorbar(i, ax=axs)
        axs.grid()
        axs.set_ylabel('Vitesse')
        axs.set_xlabel('Position')
        if name is not None:
             axs.set_title(name, fontsize=10)
    
    def create_data_array(self):
        for test_ind in range(len(self.metadata[4])):            
            test = self.metadata[4][test_ind]
            for tested_controler in self.metadata[2]:
                if tested_controler == 0 and len(self.controler[0])>0:
                    grid = self.metadata[-2][test_ind]
                    self.data_array[test[0], test[1], 0, :, :] = grid.get_grid_from_array(grid.get_input_from_policy(self.controler[0][test_ind].pi, 0))
                if tested_controler == 1 and len(self.controler[1])>0:
                    self.data_array[test[0], test[1], 1, :, :] = self.controler[1][test_ind].c_array()
                if tested_controler == 2 and len(self.controler[2])>0:
                    self.data_array[test[0], test[1], 2, :, :] = self.controler[2][test_ind].c_array()
                if tested_controler == 3 and len(self.controler[3])>0 :
                    self.data_array[test[0], test[1], 3, :, :] = self.controler[3][test_ind].c_array()

    def create_cost_array(self):
        for test_ind in range(len(self.metadata[4])):            
            test = self.metadata[4][test_ind]
            for tested_controler in self.metadata[2]:
                if tested_controler == 0 and len(self.cost2go[0])>0:
                    self.cost_array[test[0], test[1], 0, :, :] = self.cost2go[0][test_ind].J_total
                    self.cost_array_override[test[0], test[1], 0, :, :] = self.cost2go[0][test_ind].J_override
                    self.cost_array_security[test[0], test[1], 0, :, :] = self.cost2go[0][test_ind].J_security
                    self.cost_array_confort[test[0], test[1], 0, :, :] = self.cost2go[0][test_ind].J_confort
                    
                if tested_controler == 1 and len(self.cost2go[1])>0:
                    self.cost_array[test[0], test[1], 1, :, :] = self.cost2go[1][test_ind].J_total
                    self.cost_array_override[test[0], test[1], 1, :, :] = self.cost2go[1][test_ind].J_override
                    self.cost_array_security[test[0], test[1], 1, :, :] = self.cost2go[1][test_ind].J_security
                    self.cost_array_confort[test[0], test[1], 1, :, :] = self.cost2go[1][test_ind].J_confort
                    
                if tested_controler == 2 and len(self.cost2go[2])>0:
                    self.cost_array[test[0], test[1], 2, :, :] = self.cost2go[2][test_ind].J_total
                    self.cost_array_override[test[0], test[1], 2, :, :] = self.cost2go[2][test_ind].J_override
                    self.cost_array_security[test[0], test[1], 2, :, :] = self.cost2go[2][test_ind].J_security
                    self.cost_array_confort[test[0], test[1], 2, :, :] = self.cost2go[2][test_ind].J_confort
                    
                if tested_controler == 3 and len(self.cost2go[3])>0 :
                    self.cost_array[test[0], test[1], 3, :, :] = self.cost2go[3][test_ind].J_total
                    self.cost_array_override[test[0], test[1], 3, :, :] = self.cost2go[3][test_ind].J_override
                    self.cost_array_security[test[0], test[1], 3, :, :] = self.cost2go[3][test_ind].J_security
                    self.cost_array_confort[test[0], test[1], 3, :, :] = self.cost2go[3][test_ind].J_confort
    
    def plotting(self):
        if self.nbr_roads == 1 and self.nbr_drivers == 1 and self.nbr_controlers == 1 and len(self.metadata[3])==1:
            self.plot_single()
            
        if self.nbr_roads == 1 and self.nbr_drivers == 1 and self.nbr_controlers == 1 and len(self.metadata[3])!=1:
            self.plot_multiple_coef()         
            
        if self.nbr_roads != 1 and self.nbr_drivers == 1 and self.nbr_controlers == 1:
            self.plot_multiple_road()
            
        if self.nbr_roads == 1 and self.nbr_drivers != 1 and self.nbr_controlers == 1:
            self.plot_multiple_driver()
        
        if self.nbr_roads == 1 and self.nbr_drivers == 1 and self.nbr_controlers != 1:
            self.plot_multiple_controler()
            
        if self.nbr_roads != 1 and self.nbr_drivers == 1 and self.nbr_controlers != 1:
            self.plot_multiple_controler_and_road()
            
        if self.nbr_roads == 1 and self.nbr_drivers != 1 and self.nbr_controlers != 1:
            self.plot_multiple_controler_and_driver()
            
            
    def plotting_cost2go(self):
        if self.nbr_roads == 1 and self.nbr_drivers == 1 and self.nbr_controlers == 1 and len(self.metadata[3])==1:
            self.plot_cost_single()
            
        if self.nbr_roads == 1 and self.nbr_drivers == 1 and self.nbr_controlers == 1 and len(self.metadata[3])!=1:
            self.plot_cost_multiple_coef()         
            
        if self.nbr_roads != 1 and self.nbr_drivers == 1 and self.nbr_controlers == 1:
            self.plot_cost_multiple_road()
            
        if self.nbr_roads == 1 and self.nbr_drivers != 1 and self.nbr_controlers == 1:
            self.plot_cost_multiple_driver()
        
        if self.nbr_roads == 1 and self.nbr_drivers == 1 and self.nbr_controlers != 1:
            self.plot_cost_multiple_controler()
            
        if self.nbr_roads != 1 and self.nbr_drivers == 1 and self.nbr_controlers != 1:
            self.plot_cost_multiple_controler_and_road()
            
        if self.nbr_roads == 1 and self.nbr_drivers != 1 and self.nbr_controlers != 1:
            self.plot_cost_multiple_controler_and_driver()
            
    def plot_single(self):
        fig, axs = plt.subplots(1)
        plt.ion()
        fig.suptitle(str(self.metadata[0]) +str(self.metadata[1])+str(self.metadata[2])+str(self.metadata[3]))
        cont = self.metadata[2][0]
        if cont == 0:
            grid = self.metadata[-2][0]
        else:
            grid = self.metadata[-1][0]
            
        for test in self.metadata[4]:
            self.plot_function(fig, axs, grid, self.data_array[test[0],test[1],cont,:,:].T)
            #axs.grid()
        
    def plot_multiple_road(self):
        fig, axs = plt.subplots(1, self.nbr_roads)
        plt.ion()
        fig.suptitle(str(self.metadata[0]) +str(self.metadata[1])+str(self.metadata[2])+str(self.metadata[3]))
        cont = self.metadata[2][0]
        ind = 0
        for test in self.metadata[4]:
            if cont == 0:
                grid = self.metadata[-2][ind]
            else:
                grid = self.metadata[-1][ind]
            self.plot_function(fig, axs[ind], grid, self.data_array[test[0],test[1], cont, :, :].T)
            axs[ind].grid()
            ind = ind+1


    def plot_multiple_driver(self):
        fig, axs = plt.subplots(1, self.nbr_drivers)
        plt.ion()
        fig.suptitle(str(self.metadata[0]) +str(self.metadata[1])+str(self.metadata[2])+str(self.metadata[3]))
        cont = self.metadata[2][0]
        ind = 0
        for test in self.metadata[4]:
            if cont == 0:
                grid = self.metadata[-2][ind]
            else:
                grid = self.metadata[-1][ind]
            self.plot_function(fig, axs[ind], grid, self.data_array[test[0],test[1], cont, :, :].T, name = test)
            #axs[ind].grid()
            ind = ind+1

    def plot_multiple_controler(self):
        fig, axs = plt.subplots(1, self.nbr_controlers)
        plt.ion()
        fig.suptitle(str(self.metadata[0]) +str(self.metadata[1])+str(self.metadata[2])+str(self.metadata[3]))
        ind = 0
        for test in self.metadata[4]:
            for cont in self.metadata[2]:
                if cont == 0:
                    grid = self.metadata[-2][0]
                else:
                    grid = self.metadata[-1][0]                 
                self.plot_function(fig, axs[ind], grid, self.data_array[test[0], test[1], cont, :, :].T, name = test)
                
                ind = ind+1
                
    def plot_multiple_coef(self):
        fig, axs = plt.subplots(1, self.nbr_coefs)
        plt.ion()
        ind = 0
        for test in self.metadata[4]:
            grid = self.metadata[-2][ind]
            data = grid.get_grid_from_array(grid.get_input_from_policy(self.controler[0][ind].pi, 0)).T
            axs[ind].set_title(self.metadata[5][ind])
            self.plot_function(fig, axs[ind], grid, data)
            ind = ind+1  


    def plot_cost_single(self):
        fig, axs = plt.subplots(1,4)
        plt.ion()
        fig.suptitle(self.metadata[-3])
        cont = self.metadata[2][0]
        if cont == 0:
            grid = self.metadata[-2][0]
        else:
            grid = self.metadata[-1][0]
            
        for test in self.metadata[4]:
            axs[0].set_title('Total')
            axs[1].set_title('Confort: '+str(self.metadata[3][0][0]))
            axs[2].set_title('Override: '+str(self.metadata[3][0][1]))
            axs[3].set_title('Security: '+str(self.metadata[3][0][2]))
            
            self.plot_function(fig, axs[0], grid, self.cost_array[test[0],test[1],cont,:,:])
            self.plot_function(fig, axs[1], grid, self.cost_array_confort[test[0],test[1],cont,:,:])
            self.plot_function(fig, axs[2], grid, self.cost_array_override[test[0],test[1],cont,:,:])
            self.plot_function(fig, axs[3], grid, self.cost_array_security[test[0],test[1],cont,:,:])
            
    def plot_cost_multiple_road(self):
        fig, axs = plt.subplots(1, self.nbr_roads)
        plt.ion()
        cont = self.metadata[2][0]
        ind = 0
        for test in self.metadata[4]:
            if cont == 0:
                grid = self.metadata[-2][ind]
            else:
                grid = self.metadata[-1][ind]
            self.plot_function(fig, axs[ind], grid, self.cost_array[test[0],test[1], cont, :, :])
            ind = ind+1


    def plot_cost_multiple_driver(self):
        fig, axs = plt.subplots(1, self.nbr_drivers)
        plt.ion()
        cont = self.metadata[2][0]
        ind = 0
        for test in self.metadata[4]:
            if cont == 0:
                grid = self.metadata[-2][ind]
            else:
                grid = self.metadata[-1][ind]
            self.plot_function(fig, axs[ind], grid, self.cost_array[test[0],test[1], cont, :, :])
            ind = ind+1

    def plot_cost_multiple_controler(self):
        fig, axs = plt.subplots(1, self.nbr_controlers)
        plt.ion()
        ind = 0
        for test in self.metadata[4]:
            for cont in self.metadata[2]:
                if cont == 0:
                    grid = self.metadata[-2][0]
                else:
                    grid = self.metadata[-1][0]                 
                self.plot_function(fig, axs[ind], grid, self.cost_array[test[0],test[1], cont, :, :].T)
                ind = ind+1
                
    def plot_cost_multiple_coef(self):
        fig, axs = plt.subplots(self.nbr_coefs, 4)
        plt.ion()
        ind = 0
        for test in self.metadata[4]:
            grid = self.metadata[-2][ind]
            axs[ind][0].set_title('Total')
            axs[ind][1].set_title('Confort: '+str(self.metadata[3][ind][0]))
            axs[ind][2].set_title('Override: '+str(self.metadata[3][ind][1]))
            axs[ind][3].set_title('Security: '+str(self.metadata[3][ind][2]))
            
            data = np.clip(self.cost2go[0][ind].J_total.T,0,100)
            self.plot_function(fig, axs[ind][0], grid, data)
            data = np.clip(self.cost2go[0][ind].J_confort.T,0,100)
            self.plot_function(fig, axs[ind][1], grid, data)
            data = np.clip(self.cost2go[0][ind].J_override.T,0,100)
            self.plot_function(fig, axs[ind][2], grid, data)
            data = np.clip(self.cost2go[0][ind].J_security.T,0,100)
            self.plot_function(fig, axs[ind][3], grid, data)
            
            ind = ind+1    
            
    def plot_multiple_controler_and_driver(self):
        fig, axs = plt.subplots(self.nbr_drivers, self.nbr_controlers)
        plt.ion()
        name = str(self.metadata[3])
        fig.suptitle(name)
        for r_i in range(self.nbr_roads):
             for d_i in range(self.nbr_drivers):
                  for c_i in range(self.nbr_controlers):
                       grid = self.metadata[-1][0]
                       
                       self.plot_function(fig, axs[d_i][c_i], grid, self.data_array[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T
                                          , name = str([self.roads_array[self.metadata[0][r_i]],self.drivers_array[self.metadata[1][d_i]],self.controlers_array[self.metadata[2][c_i]]])) 
    
    def plot_multiple_controler_and_road(self):
        fig, axs = plt.subplots(self.nbr_controlers,self.nbr_roads)
        plt.ion()
        for r_i in range(self.nbr_roads):
             for d_i in range(self.nbr_drivers):
                  for c_i in range(self.nbr_controlers):
                       grid = self.metadata[-1][0]
                       
                       self.plot_function(fig, axs[c_i][r_i], grid, self.data_array[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T) 


    def plot_cost_multiple_controler_and_driver(self):
        fig, axs = plt.subplots(self.nbr_drivers, self.nbr_controlers)
        plt.ion()
        fig.suptitle('ROAD DRIVER CONTROLER')
        for r_i in range(self.nbr_roads):
             for d_i in range(self.nbr_drivers):
                  for c_i in range(self.nbr_controlers):
                       grid = self.metadata[-1][0]
                       
                       self.plot_function(fig, axs[d_i][c_i], grid, self.cost_array[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T
                                          , name = str([self.roads_array[self.metadata[0][r_i]],self.drivers_array[self.metadata[1][d_i]],self.controlers_array[self.metadata[2][c_i]]])) 
    
    def plot_cost_multiple_controler_and_road(self):
        fig, axs = plt.subplots(self.nbr_controlers,self.nbr_roads)
        plt.ion()
        for r_i in range(self.nbr_roads):
             for d_i in range(self.nbr_drivers):
                  for c_i in range(self.nbr_controlers):
                       grid = self.metadata[-1][0]
                       self.plot_function(fig, axs[c_i][r_i], grid, self.cost_array[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T) 

    def plot_cost_multiple_controler_and_road_param(self):
        for i in range(3):
            fig, axs = plt.subplots(self.nbr_controlers,self.nbr_roads)
            if i == 0:
                fig.suptitle('Coûts à venir pour le paramètre: Sécurité')
            if i == 1:
                fig.suptitle('Coûts à venir pour le paramètre: Confort')
            if i == 2:
                fig.suptitle('Coûts à venir pour le paramètre: Liberté')
            plt.ion()
            for r_i in range(self.nbr_roads):
                 for d_i in range(self.nbr_drivers):
                      for c_i in range(self.nbr_controlers):
                           if c_i == 0:
                               name = self.roads_array[r_i]
                           else:
                               name = None
                           grid = self.metadata[-1][0]
                           if i == 0:
                               self.plot_function(fig, axs[c_i][r_i], grid, self.cost_array_security[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T,name=name) 
                           if i == 1:
                               self.plot_function(fig, axs[c_i][r_i], grid, self.cost_array_confort[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T,name=name) 
                           if i == 2:
                               self.plot_function(fig, axs[c_i][r_i], grid, self.cost_array_override[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T,name=name)
                               
    def plot_cost_multiple_controler_and_road_param_diff(self, roads_to_test):
        for i in range(3):
            fig, axs = plt.subplots(self.nbr_controlers,len(roads_to_test))
            if i == 0:
                fig.suptitle('Coûts à venir pour le paramètre: Sécurité')
            if i == 1:
                fig.suptitle('Coûts à venir pour le paramètre: Confort')
            if i == 2:
                fig.suptitle('Coûts à venir pour le paramètre: Liberté')
            plt.ion()
            for r in range(len(roads_to_test)):
                 for d_i in range(self.nbr_drivers):
                      for c_i in range(self.nbr_controlers):
                           grid = self.metadata[-1][0]
                           r_i = roads_to_test[r]
                           if i == 0:
                               if c_i == 0:
                                   if r == 0:
                                       axs[c_i][r_i].set_ylabel(self.controlers_array[c_i])
                                   name = self.roads_array[r_i]
                                   self.plot_function(fig, axs[c_i][r_i], grid, self.cost_array_security[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T, name=name)
                               else:
                                   if r == 0:
                                       axs[c_i][r_i].set_ylabel(self.controlers_array[c_i])
                                   diff_data = self.cost_array_security[self.metadata[0][r_i],self.metadata[1][d_i],self.metadata[2][0], :, :]- self.cost_array_security[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :]
                                   self.plot_function(fig, axs[c_i][r_i], grid, diff_data.T)
                           if i == 1:
                               if c_i == 0:
                                   if r == 0:
                                       axs[c_i][r_i].set_ylabel(self.controlers_array[c_i])
                                       
                                   name = self.roads_array[r_i]
                                   self.plot_function(fig, axs[c_i][r_i], grid, self.cost_array_confort[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T, name = name) 
                               else:
                                   if r == 0:
                                       axs[c_i][r_i].set_ylabel(self.controlers_array[c_i])
                                   diff_data = self.cost_array_confort[self.metadata[0][r_i],self.metadata[1][d_i],self.metadata[2][0], :, :]- self.cost_array_confort[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :]
                                   self.plot_function(fig, axs[c_i][r_i], grid, diff_data.T)
                           if i == 2:
                               if c_i == 0:
                                   if r == 0:
                                       axs[c_i][r_i].set_ylabel(self.controlers_array[c_i])
                                   name = self.roads_array[r_i]
                                   self.plot_function(fig, axs[c_i][r_i], grid, self.cost_array_override[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T, name = name) 
                               else:
                                   if r == 0:
                                       axs[c_i][r_i].set_ylabel(self.controlers_array[c_i])
                                   name = self.roads_array[r_i]
                                   diff_data = self.cost_array_override[self.metadata[0][r_i],self.metadata[1][d_i],self.metadata[2][0], :, :]- self.cost_array_override[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :]
                                   self.plot_function(fig, axs[c_i][r_i], grid, diff_data.T)

    def plot_cost_multiple_controler_and_single_road_param(self, road_to_test):
        for i in range(3):
            fig, axs = plt.subplots(1, self.nbr_controlers)
            if i == 0:
                fig.suptitle('Coûts à venir pour le paramètre: Sécurité sur la route: '+str(self.roads_array[road_to_test]))
            if i == 1:
                fig.suptitle('Coûts à venir pour le paramètre: Confort sur la route: '+str(self.roads_array[road_to_test]))
            if i == 2:
                fig.suptitle('Coûts à venir pour le paramètre: Liberté sur la route: '+str(self.roads_array[road_to_test]))
            plt.ion()
            for d_i in range(self.nbr_drivers):
                for c_i in range(self.nbr_controlers):
                     grid = self.metadata[-1][0]
                     r_i = road_to_test
                     if i == 0:
                        name = self.controlers_array[c_i]
                        self.plot_function(fig, axs[c_i], grid, self.cost_array_security[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T,name=name)
                        
                     if i == 1:
                        name = self.controlers_array[c_i]
                        self.plot_function(fig, axs[c_i], grid, self.cost_array_confort[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T,name=name)
                        
                     if i == 2:
                        name = self.controlers_array[c_i]
                        self.plot_function(fig, axs[c_i], grid, self.cost_array_override[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T,name=name)


    def plot_cost_multiple_controler_and_single_road_param_diff(self, road_to_test):
        for i in range(3):
            fig, axs = plt.subplots(1, self.nbr_controlers)
            if i == 0:
                fig.suptitle('Coûts à venir pour le paramètre: Sécurité sur la route: '+str(self.roads_array[road_to_test]))
            if i == 1:
                fig.suptitle('Coûts à venir pour le paramètre: Confort sur la route: '+str(self.roads_array[road_to_test]))
            if i == 2:
                fig.suptitle('Coûts à venir pour le paramètre: Liberté sur la route: '+str(self.roads_array[road_to_test]))
            plt.ion()
            for d_i in range(self.nbr_drivers):
                for c_i in range(self.nbr_controlers):
                     grid = self.metadata[-1][0]
                     r_i = road_to_test
                     if i == 0:
                         if c_i == 0:
                             name = self.controlers_array[c_i]
                             self.plot_function(fig, axs[c_i], grid, self.cost_array_security[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T, name = name)
                         else:
                             diff_data = self.cost_array_security[self.metadata[0][r_i],self.metadata[1][d_i],self.metadata[2][0], :, :]- self.cost_array_security[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :]
                             name = self.controlers_array[c_i]
                             self.plot_function(fig, axs[c_i], grid, diff_data.T, name = name)
                     if i == 1:
                         if c_i == 0:
                             name = self.controlers_array[c_i]
                             self.plot_function(fig, axs[c_i], grid, self.cost_array_confort[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T,name=name)
                         else:
                             diff_data = self.cost_array_confort[self.metadata[0][r_i],self.metadata[1][d_i],self.metadata[2][0], :, :]- self.cost_array_confort[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :]
                             name = self.controlers_array[c_i]
                             self.plot_function(fig, axs[c_i], grid, diff_data.T,name=name)
                     if i == 2:
                         if c_i == 0:
                             name = self.controlers_array[c_i]
                             self.plot_function(fig, axs[c_i], grid, self.cost_array_override[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T,name=name)
                         else:
                             diff_data = self.cost_array_override[self.metadata[0][r_i],self.metadata[1][d_i],self.metadata[2][0], :, :]- self.cost_array_override[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :]
                             name = self.controlers_array[c_i]
                             self.plot_function(fig, axs[c_i], grid, diff_data.T,name=name)



    def plot_cost_multiple_controler_and_road_diff(self):
        fig, axs = plt.subplots(self.nbr_controlers,self.nbr_roads)
        plt.ion()
        for r_i in range(self.nbr_roads):
             for d_i in range(self.nbr_drivers):
                  for c_i in range(self.nbr_controlers):
                       grid = self.metadata[-1][0]
                       if c_i == 0:
                           self.plot_function(fig, axs[c_i][r_i], grid, self.cost_array[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T)
                       else:
                           diff_data = self.cost_array[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][0], :, :] - self.cost_array[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :]
                           self.plot_function(fig, axs[c_i][r_i], grid, diff_data.T)

    def plot_cost_multiple_controler_and_road_diff_performance(self):
        fig, axs = plt.subplots(self.nbr_controlers,self.nbr_roads)
        plt.ion()
        for r_i in range(self.nbr_roads):
             for d_i in range(self.nbr_drivers):
                  for c_i in range(self.nbr_controlers):
                       grid = self.metadata[-1][0]
                       if c_i == 0:
                           self.plot_function(fig, axs[c_i][r_i], grid, self.cost_array[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T)
                       else:
                           diff_data = self.cost_array[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][0], :, :] - self.cost_array[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :]
                           self.plot_function(fig, axs[c_i][r_i], grid, diff_data.T)


    def plot_cost_diff_road(self,road):
        fig, axs = plt.subplots(1, self.nbr_controlers)
        plt.ion()
        r_i = road
        d_i = 0
        for c_i in range(self.nbr_controlers):
             grid = self.metadata[-1][0]
             if c_i == 0:
                 self.plot_function(fig, axs[c_i], grid, self.cost_array[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :].T)
             else:
                 diff_data = self.cost_array[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][0], :, :] - self.cost_array[self.metadata[0][r_i],self.metadata[1][d_i], self.metadata[2][c_i], :, :]
                 self.plot_function(fig, axs[c_i], grid, diff_data.T)
            
    def plot_multiple_data(self, size, controler,name = None):
         
        fig, axs = plt.subplots(size[0], size[1])
        if name == None:
            fig.suptitle('Loi de commandes: ' + str(self.controlers_array[controler]) + ' pour différentes routes')
        else:
            fig.suptitle('Loi de commandes: ' + str(self.controlers_array[controler]) + ' pour différentes routes avec une décélération constante de ' + name + 'm/s^2')
        
        plt.ion()
        cont = self.metadata[2][controler] - self.metadata[2][0] 
        ind = 0
        ind2 = 0
        for test in self.metadata[4]:
            grid = self.metadata[-1][ind]
            
            i1 = ind%size[1]
            
            if i1 == 0 and ind != 0:
                 ind2=ind2+1
            
            i2 =  ind2
            self.plot_function(fig, axs[i2][i1], grid, self.data_array[test[0],test[1], cont, :, :])
            axs[i2][i1].set_title('Route: ' + self.roads_array[test[0]])
            #axs[i2][i1].text(-99, 3, 'Route: ' + self.roads_array[test[0]], fontdict = self.font)
            #axs[i2][i1].text(-99, 2, 'Conducteur: '+ self.drivers_array[test[1]], fontdict = self.font)
            #axs[i2][i1].text(-99, 1, 'Loi de commandes: '+self.controlers_array[controler], fontdict = self.font)
            
            ind = ind+1   
            
    def plot_worst_msd_controller(self):
        msd_array = self.controler[2][0].c_array().T
        msd_worst_array = self.controler[2][0].c_array_worst_e().T
        grid = self.metadata[-1][0]
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Différence entre la loi de commandes par distance de freins minimale si l\'espérance est incluse')
        axs[0].set_title('Sans Espérence')
        axs[1].set_title('Avec Espérence')
        self.plot_function(fig, axs[0], grid, msd_array)
        self.plot_function(fig, axs[1], grid, msd_worst_array)
        axs[0].grid()
        axs[1].grid()
        
    def plot_desired(self,road,driver,controler, name=''):
        data = self.data_array[road][driver][controler]
        fig, axs = plt.subplots(1, 1)
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        self.plot_function(fig, axs, grid, data.T)
        
    def plot_desired_data(self, data, name=''):
        fig, axs = plt.subplots(1, 1)
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        self.plot_function(fig, axs, grid, data.T)        

    def plot_desired_multiple_data(self, data, name):
        fig, axs = plt.subplots(1, len(data))
        fig.suptitle('Commandes données par l\'algorithme de temps de collisions pour des décélérations constantes')
        grid = self.metadata[-1][0]
        ind = 0
        for d in data:
            self.plot_function(fig, axs[ind], grid, data[ind].T, name = name[ind])    
            ind = ind+1

    def plot_similarity(self,road,driver,controler, name=''):
        fig, axs = plt.subplots(1,1)
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        
        temp = np.zeros(grid.x_grid_dim)
        for p in range(len(grid.x_level[0])):
            for v in range(len(grid.x_level[1])):
                if self.data_array[road][driver][controler][p][v] == self.data_array[road][driver][3][p][v]:
                    temp[p][v] = 0
                else:
                    temp[p][v] = self.data_array[road][driver][controler][p][v]
        self.plot_function(fig,axs,grid,temp.T)
                      
    def plot_cont(self,road,driver, name=''):
        fig, axs = plt.subplots(2,2)
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        self.plot_function(fig,axs[0][0],grid,self.data_array[road][driver][0].T, name = self.controlers_array[0])
        self.plot_function(fig,axs[0][1],grid,self.data_array[road][driver][1].T, name = self.controlers_array[1])
        self.plot_function(fig,axs[1][0],grid,self.data_array[road][driver][2].T, name = self.controlers_array[2])
        self.plot_function(fig,axs[1][1],grid,self.data_array[road][driver][3].T, name = self.controlers_array[3])

    def plot_cost(self,road,driver, name='',save = False):
        fig, axs = plt.subplots(2,2, figsize=(8,8))
        fig.suptitle('Route: '+ str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver]))
        grid = self.metadata[-1][0]
        self.plot_function(fig,axs[0][0],grid,self.cost_array[road][driver][0].T, name = self.controlers_array[0])
        self.plot_function(fig,axs[0][1],grid,self.cost_array[road][driver][1].T, name = self.controlers_array[1])
        self.plot_function(fig,axs[1][0],grid,self.cost_array[road][driver][2].T, name = self.controlers_array[2])
        self.plot_function(fig,axs[1][1],grid,self.cost_array[road][driver][3].T, name = self.controlers_array[3])
        plt.tight_layout() 
        if save:
            plt.savefig(self.folder_dir+'\\Road_Controlers\\Cost\\'+str(road)+str(driver))
            plt.close('all')

    def plot_cost_driver(self,road,driver, name='',save = False):
        fig, axs = plt.subplots(2,2, figsize=(8,8))
        fig.suptitle('Route: '+ str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver]))
        grid = self.metadata[-1][0]
        self.plot_function(fig,axs[0][0],grid,self.cost_array[road][driver][0].T, name = self.controlers_array[0])
        self.plot_function(fig,axs[0][1],grid,self.cost_array[road][driver][1].T, name = self.controlers_array[1])
        self.plot_function(fig,axs[1][0],grid,self.cost_array[road][driver][2].T, name = self.controlers_array[2])
        self.plot_function(fig,axs[1][1],grid,self.cost_array[road][driver][3].T, name = self.controlers_array[3])
        plt.tight_layout() 
        if save:
            plt.savefig(self.folder_dir+'\\Driver_Controlers\\Cost\\'+str(road)+str(driver))
            plt.close('all')
        
    def plot_c2g_single(self,road,driver,controler):
        fig, axs = plt.subplots(2,2, figsize=(8,8))
        name = 'Coûts à venir pour la loi de commandes: ' + str(self.controlers_array[controler])+ ' sur une route ' + str(self.roads_array[road])
        name = 'Loi de commandes: ' + str(self.controlers_array[controler]) + '\nRoute: ' + str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver])
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        self.plot_function(fig,axs[0][0],grid,self.cost_array[road][driver][controler].T, name = 'Coûts à venir totaux')
        self.plot_function(fig,axs[0][1],grid,self.cost_array_security[road][driver][controler].T, name = 'Coûts à venir pour la sécurité')
        self.plot_function(fig,axs[1][0],grid,self.cost_array_confort[road][driver][controler].T, name = 'Coûts à venir pour le confort')
        self.plot_function(fig,axs[1][1],grid,self.cost_array_override[road][driver][controler].T, name = 'Coûts à venir pour la liberté')
        plt.tight_layout() 
        #plt.savefig('image/cost'+str(road)+str(driver)+str(controler))

    def plot_c2g_single_diff(self,road,driver,controler, controler2):
        fig, axs = plt.subplots(2,2, figsize=(8,8))
        name = 'Loi de commandes: ' + str(self.controlers_array[controler]) + ' - ' + str(self.controlers_array[controler2]) + '\nRoute: ' + str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver])
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        self.plot_function(fig,axs[0][0],grid,self.cost_array[road][driver][controler].T-self.cost_array[road][driver][controler2].T, name = 'Coûts à venir totaux')
        self.plot_function(fig,axs[0][1],grid,self.cost_array_security[road][driver][controler].T-self.cost_array_security[road][driver][controler2].T, name = 'Coûts à venir pour la sécurité')
        self.plot_function(fig,axs[1][0],grid,self.cost_array_confort[road][driver][controler].T-self.cost_array_confort[road][driver][controler2].T, name = 'Coûts à venir pour le confort')
        self.plot_function(fig,axs[1][1],grid,self.cost_array_override[road][driver][controler].T-self.cost_array_override[road][driver][controler2].T, name = 'Coûts à venir pour la liberté')   
        plt.tight_layout() 
        plt.savefig('image/cost_diff'+str(road)+str(driver)+'_'+str(controler)+'-'+str(controler2))


    def plot_c2g_single_diff_stricly_better(self,road,driver,controler, controler2):
        fig, axs = plt.subplots(2,2)
        name = ('Coûts à venir où ' + self.controlers_array[controler] + ' - ' +   self.controlers_array[controler2])
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        
        temp_res = self.cost_array[road][driver][controler]-self.cost_array[road][driver][controler2]
        temp_res[temp_res==0] = 0
        temp_res[temp_res<0] = -1
        temp_res[temp_res>0] = 1
        self.plot_function(fig,axs[0][0],grid,temp_res.T, name = 'Coûts à venir totaux')
        
        temp_res = self.cost_array_security[road][driver][controler]-self.cost_array_security[road][driver][controler2]
        temp_res[temp_res==0] = 0
        temp_res[temp_res<0] = -1
        temp_res[temp_res>0] = 1
        self.plot_function(fig,axs[0][1],grid,temp_res.T, name = 'Coûts à venir pour la sécurité')
        
        temp_res = self.cost_array_confort[road][driver][controler]-self.cost_array_confort[road][driver][controler2]
        temp_res[temp_res==0] = 0
        temp_res[temp_res<0] = -1
        temp_res[temp_res>0] = 1
        self.plot_function(fig,axs[1][0],grid,temp_res.T, name = 'Coûts à venir pour le confort')
        
        temp_res = self.cost_array_override[road][driver][controler]-self.cost_array_override[road][driver][controler2]
        temp_res[temp_res==0] = 0
        temp_res[temp_res<0] = -1
        temp_res[temp_res>0] = 1
        self.plot_function(fig,axs[1][1],grid,temp_res.T, name = 'Coûts à venir pour la Liberté')


                    
    def plot_cost_diff(self,road,driver, name=''):
        fig, axs = plt.subplots(2,2)
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        self.plot_function(fig,axs[0][0],grid,self.cost_array[road][driver][0].T, name = self.controlers_array[0])
        self.plot_function(fig,axs[0][1],grid,self.cost_array[road][driver][1].T-self.cost_array[road][driver][0].T, name = self.controlers_array[1])
        self.plot_function(fig,axs[1][0],grid,self.cost_array[road][driver][2].T-self.cost_array[road][driver][0].T, name = self.controlers_array[2])
        self.plot_function(fig,axs[1][1],grid,self.cost_array[road][driver][3].T-self.cost_array[road][driver][0].T, name = self.controlers_array[3])                    
                    
    def plot_diff_cont(self,road,driver, name=''):
        fig, axs = plt.subplots(2,2)
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        self.plot_function(fig,axs[0][0],grid,self.data_array[road][driver][0].T, name = self.controlers_array[0])
        self.plot_function(fig,axs[0][1],grid,self.data_array[road][driver][1].T-self.data_array[road][driver][0].T, name = self.controlers_array[1])
        self.plot_function(fig,axs[1][0],grid,self.data_array[road][driver][2].T-self.data_array[road][driver][0].T, name = self.controlers_array[2])
        self.plot_function(fig,axs[1][1],grid,self.data_array[road][driver][3].T-self.data_array[road][driver][0].T, name = self.controlers_array[3])                    
        
    def print_resultats_fast(self, road, driver, pos = [0, 150], vit = [0,150], save = False, name_mean = 'All'):
        cost_vi = np.zeros(4)
        cost_ttc = np.zeros(4)
        cost_msd = np.zeros(4)
        cost_human = np.zeros(4)
        
        grid = self.metadata[-1][0]
        cost_vi[0] = np.sum(self.cost_array[road][driver][0][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array[road][driver][0][pos[0]:pos[1],vit[0]:vit[1]])
        cost_vi[1] = np.sum(self.cost_array_security[road][driver][0][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_security[road][driver][0][pos[0]:pos[1],vit[0]:vit[1]])
        cost_vi[2] = np.sum(self.cost_array_confort[road][driver][0][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_confort[road][driver][0][pos[0]:pos[1],vit[0]:vit[1]])
        cost_vi[3] = np.sum(self.cost_array_override[road][driver][0][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_override[road][driver][0][pos[0]:pos[1],vit[0]:vit[1]])
    
        cost_ttc[0] = np.sum(self.cost_array[road][driver][1][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array[road][driver][1][pos[0]:pos[1],vit[0]:vit[1]])
        cost_ttc[1] = np.sum(self.cost_array_security[road][driver][1][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_security[road][driver][1][pos[0]:pos[1],vit[0]:vit[1]])
        cost_ttc[2] = np.sum(self.cost_array_confort[road][driver][1][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_confort[road][driver][1][pos[0]:pos[1],vit[0]:vit[1]])
        cost_ttc[3] = np.sum(self.cost_array_override[road][driver][1][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_override[road][driver][1][pos[0]:pos[1],vit[0]:vit[1]])
        
        cost_msd[0] = np.sum(self.cost_array[road][driver][2][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array[road][driver][2][pos[0]:pos[1],vit[0]:vit[1]])
        cost_msd[1] = np.sum(self.cost_array_security[road][driver][2][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_security[road][driver][2][pos[0]:pos[1],vit[0]:vit[1]])
        cost_msd[2] = np.sum(self.cost_array_confort[road][driver][2][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_confort[road][driver][2][pos[0]:pos[1],vit[0]:vit[1]])
        cost_msd[3] = np.sum(self.cost_array_override[road][driver][2][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_override[road][driver][2][pos[0]:pos[1],vit[0]:vit[1]])
        
        cost_human[0] = np.sum(self.cost_array[road][driver][3][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array[road][driver][3][pos[0]:pos[1],vit[0]:vit[1]])
        cost_human[1] = np.sum(self.cost_array_security[road][driver][3][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_security[road][driver][3][pos[0]:pos[1],vit[0]:vit[1]])
        cost_human[2] = np.sum(self.cost_array_confort[road][driver][3][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_confort[road][driver][3][pos[0]:pos[1],vit[0]:vit[1]])
        cost_human[3] = np.sum(self.cost_array_override[road][driver][3][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_override[road][driver][3][pos[0]:pos[1],vit[0]:vit[1]])

        p_min = grid.x_level[0][pos[0]]
        p_max = grid.x_level[0][pos[1]-1]
        
        v_min = grid.x_level[1][vit[0]]
        v_max = grid.x_level[1][vit[1]-1]
        
        dp = [p_min,p_max]
        dv = [v_min,v_max]
        
        params = ("Total", "Sécurité", "Confort", "Liberté")
        x = np.arange(len(params))  # the label locations
        width = 0.20  # the width of the bars
        multiplier = 0
        
        arr = {
            'Total': (cost_vi[0], cost_ttc[0], cost_msd[0], cost_human[0]),
            'Sécurité': (cost_vi[1], cost_ttc[1], cost_msd[1], cost_human[1]),
            'Confort': (cost_vi[2], cost_ttc[2], cost_msd[2], cost_human[2]),
            'Liberté': (cost_vi[3], cost_ttc[3], cost_msd[3], cost_human[3])
            }
        
        max_val = np.zeros(4)
        max_val[0] = np.max(cost_vi)
        max_val[1] = np.max(cost_ttc)
        max_val[2] = np.max(cost_msd)
        max_val[3] = np.max(cost_human)
        maxmax = np.max(max_val)
        
        fig, ax = plt.subplots(figsize=(10,10))
        name = 'Route: ' + str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver]+'\nEspace d\'état: Position= %2.2f' % dp[0]+ ' à %2.2f'%dp[1] 
                                                                                + ' Vitesse= %2.2f' % dv[0]+ ' à %2.2f'%dv[1])
        fig.suptitle(name)
        
        for attribute, measurement in arr.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=5, label_type='edge',rotation=90)
            multiplier += 1
        
        
        
        ax.set_ylabel('Coûts à venir moyens')
        ax.set_xticks(x + width, self.controlers_array)
        ax.set_ylim([0, maxmax+(0.1*maxmax)])
        ax.legend(loc='upper right')   
        plt.tight_layout() 

        if save:
            plt.savefig(self.folder_dir+'\\MeanValues\\'+str(name_mean)+'\\'+str(road)+str(driver))
            plt.close('all')
        return arr

    def print_resultats_fast_no_plotting(self, road, driver, pos = [0, 150], vit = [0,150], save = False, name_mean = 'All'):
        cost_vi = np.zeros(4)
        cost_ttc = np.zeros(4)
        cost_msd = np.zeros(4)
        cost_human = np.zeros(4)
        
        grid = self.metadata[-1][0]
        cost_vi[0] = np.sum(self.cost_array[road][driver][0][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array[road][driver][0][pos[0]:pos[1],vit[0]:vit[1]])
        cost_vi[1] = np.sum(self.cost_array_security[road][driver][0][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_security[road][driver][0][pos[0]:pos[1],vit[0]:vit[1]])
        cost_vi[2] = np.sum(self.cost_array_confort[road][driver][0][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_confort[road][driver][0][pos[0]:pos[1],vit[0]:vit[1]])
        cost_vi[3] = np.sum(self.cost_array_override[road][driver][0][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_override[road][driver][0][pos[0]:pos[1],vit[0]:vit[1]])
    
        cost_ttc[0] = np.sum(self.cost_array[road][driver][1][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array[road][driver][1][pos[0]:pos[1],vit[0]:vit[1]])
        cost_ttc[1] = np.sum(self.cost_array_security[road][driver][1][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_security[road][driver][1][pos[0]:pos[1],vit[0]:vit[1]])
        cost_ttc[2] = np.sum(self.cost_array_confort[road][driver][1][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_confort[road][driver][1][pos[0]:pos[1],vit[0]:vit[1]])
        cost_ttc[3] = np.sum(self.cost_array_override[road][driver][1][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_override[road][driver][1][pos[0]:pos[1],vit[0]:vit[1]])
        
        cost_msd[0] = np.sum(self.cost_array[road][driver][2][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array[road][driver][2][pos[0]:pos[1],vit[0]:vit[1]])
        cost_msd[1] = np.sum(self.cost_array_security[road][driver][2][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_security[road][driver][2][pos[0]:pos[1],vit[0]:vit[1]])
        cost_msd[2] = np.sum(self.cost_array_confort[road][driver][2][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_confort[road][driver][2][pos[0]:pos[1],vit[0]:vit[1]])
        cost_msd[3] = np.sum(self.cost_array_override[road][driver][2][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_override[road][driver][2][pos[0]:pos[1],vit[0]:vit[1]])
        
        cost_human[0] = np.sum(self.cost_array[road][driver][3][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array[road][driver][3][pos[0]:pos[1],vit[0]:vit[1]])
        cost_human[1] = np.sum(self.cost_array_security[road][driver][3][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_security[road][driver][3][pos[0]:pos[1],vit[0]:vit[1]])
        cost_human[2] = np.sum(self.cost_array_confort[road][driver][3][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_confort[road][driver][3][pos[0]:pos[1],vit[0]:vit[1]])
        cost_human[3] = np.sum(self.cost_array_override[road][driver][3][pos[0]:pos[1],vit[0]:vit[1]])/np.size(self.cost_array_override[road][driver][3][pos[0]:pos[1],vit[0]:vit[1]])
        
        
        tot = [round(cost_vi[0],2),round(cost_ttc[0],2),round(cost_msd[0],2),round(cost_human[0],2)]
        sec = [round(cost_vi[1],2),round(cost_ttc[1],2),round(cost_msd[1],2),round(cost_human[1],2)]
        con = [round(cost_vi[2],2),round(cost_ttc[2],2),round(cost_msd[2],2),round(cost_human[2],2)]
        ove = [round(cost_vi[3],2),round(cost_ttc[3],2),round(cost_msd[3],2),round(cost_human[3],2)]
        
        print('\\textbf{Asphalte sec} & MSD     & TTC     & VI      & Humain   \\\\ \\hline')
        print('Totale & '+str(round(cost_msd[0],2)) + ' & ' +str(round(cost_ttc[0],2)) + ' & '+str(round(cost_vi[0],2)) + ' & '+str(round(cost_human[0],2)) + '\\\\ \\hline')
        print('Sécurité & '+str(round(cost_msd[1],2)) + ' & ' +str(round(cost_ttc[1],2)) + ' & '+str(round(cost_vi[1],2)) + ' & '+str(round(cost_human[1],2)) + '\\\\ \\hline')
        print('Confort & '+str(round(cost_msd[2],2)) + ' & ' +str(round(cost_ttc[2],2)) + ' & '+str(round(cost_vi[2],2)) + ' & '+str(round(cost_human[2],2)) + '\\\\ \\hline')
        print('Liberté & '+str(round(cost_msd[3],2)) + ' & ' +str(round(cost_ttc[3],2)) + ' & '+str(round(cost_vi[3],2)) + ' & '+str(round(cost_human[3],2)) + '\\\\ \\hline \\hline')

        print(np.max(tot) - np.min(tot))
        print(np.max(sec) - np.min(sec))
        print(np.max(con) - np.min(con))
        print(np.max(ove) - np.min(ove))

    def print_resultats_human_braking(self, road, driver, save=False):
        grid = self.metadata[-1][0]
        pos = grid.x_level[0]
        vit = grid.x_level[1]
        counter = 0
        
        cost_vi = np.zeros(4)
        cost_ttc = np.zeros(4)
        cost_msd = np.zeros(4)
        cost_human = np.zeros(4)
        
        human_model = self.data_array[road][driver][3]
        for p_ind in range(len(pos)):
            for v_ind in range(len(vit)):
                if human_model[p_ind][v_ind] != 0:
                    cost_vi[0] = cost_vi[0] + self.cost_array[road][driver][0][p_ind][v_ind]
                    cost_ttc[0] = cost_ttc[0] + self.cost_array[road][driver][1][p_ind][v_ind]
                    cost_msd[0] = cost_msd[0] + self.cost_array[road][driver][2][p_ind][v_ind]
                    cost_human[0] = cost_human[0] + self.cost_array[road][driver][3][p_ind][v_ind]
                    
                    cost_vi[1] = cost_vi[1] + self.cost_array_security[road][driver][0][p_ind][v_ind]
                    cost_ttc[1] = cost_ttc[1] + self.cost_array_security[road][driver][1][p_ind][v_ind]
                    cost_msd[1] = cost_msd[1] + self.cost_array_security[road][driver][2][p_ind][v_ind]
                    cost_human[1] = cost_human[1] + self.cost_array_security[road][driver][3][p_ind][v_ind]
                    
                    cost_vi[2] = cost_vi[2] + self.cost_array_confort[road][driver][0][p_ind][v_ind]
                    cost_ttc[2] = cost_ttc[2] + self.cost_array_confort[road][driver][1][p_ind][v_ind]
                    cost_msd[2] = cost_msd[2] + self.cost_array_confort[road][driver][2][p_ind][v_ind]
                    cost_human[2] = cost_human[2] + self.cost_array_confort[road][driver][3][p_ind][v_ind]
                    
                    cost_vi[3] = cost_vi[3] + self.cost_array_override[road][driver][0][p_ind][v_ind]
                    cost_ttc[3] = cost_ttc[3] + self.cost_array_override[road][driver][1][p_ind][v_ind]
                    cost_msd[3] = cost_msd[3] + self.cost_array_override[road][driver][2][p_ind][v_ind]
                    cost_human[3] = cost_human[3] + self.cost_array_override[road][driver][3][p_ind][v_ind]
                    
                    counter = counter+1
        
        cost_vi = cost_vi/counter
        cost_ttc = cost_ttc/counter
        cost_msd = cost_msd/counter
        cost_human = cost_human/counter
        
        params = ("Total", "Sécurité", "Confort", "Liberté")
        x = np.arange(len(params))  # the label locations
        width = 0.20  # the width of the bars
        multiplier = 0
        
        arr = {
            'Total': (cost_vi[0], cost_ttc[0], cost_msd[0], cost_human[0]),
            'Sécurité': (cost_vi[1], cost_ttc[1], cost_msd[1], cost_human[1]),
            'Confort': (cost_vi[2], cost_ttc[2], cost_msd[2], cost_human[2]),
            'Liberté': (cost_vi[3], cost_ttc[3], cost_msd[3], cost_human[3])
            }
        
        max_val = np.zeros(4)
        max_val[0] = np.max(cost_vi)
        max_val[1] = np.max(cost_ttc)
        max_val[2] = np.max(cost_msd)
        max_val[3] = np.max(cost_human)
        maxmax = np.max(max_val)
        
        fig, ax = plt.subplots(figsize=(10,10))
        name = 'Route: ' + str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver]+'\nLe conducteur décide de freiner')
        fig.suptitle(name)
        
        for attribute, measurement in arr.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=5, label_type='edge', rotation=90)
            multiplier += 1

        ax.set_ylabel('Coûts à venir moyens')
        ax.set_ylim([0, maxmax+(0.1*maxmax)])
        ax.set_xticks(x + width, self.controlers_array)
        ax.legend(loc='upper right')
        plt.tight_layout() 
        
        if save:
            plt.savefig(self.folder_dir+'\\MeanValues\\Braking\\'+str(road)+str(driver))
            plt.close('all')
        
    def print_resultats_human_no_braking(self, road, driver, save=False):
        grid = self.metadata[-1][0]
        pos = grid.x_level[0]
        vit = grid.x_level[1]
        counter = 0
        
        cost_vi = np.zeros(4)
        cost_ttc = np.zeros(4)
        cost_msd = np.zeros(4)
        cost_human = np.zeros(4)
        
        human_model = self.data_array[road][driver][3]
        for p_ind in range(len(pos)):
            for v_ind in range(len(vit)):
                if human_model[p_ind][v_ind] == 0:
                    cost_vi[0] = cost_vi[0] + self.cost_array[road][driver][0][p_ind][v_ind]
                    cost_ttc[0] = cost_ttc[0] + self.cost_array[road][driver][1][p_ind][v_ind]
                    cost_msd[0] = cost_msd[0] + self.cost_array[road][driver][2][p_ind][v_ind]
                    cost_human[0] = cost_human[0] + self.cost_array[road][driver][3][p_ind][v_ind]
                    
                    cost_vi[1] = cost_vi[1] + self.cost_array_security[road][driver][0][p_ind][v_ind]
                    cost_ttc[1] = cost_ttc[1] + self.cost_array_security[road][driver][1][p_ind][v_ind]
                    cost_msd[1] = cost_msd[1] + self.cost_array_security[road][driver][2][p_ind][v_ind]
                    cost_human[1] = cost_human[1] + self.cost_array_security[road][driver][3][p_ind][v_ind]
                    
                    cost_vi[2] = cost_vi[2] + self.cost_array_confort[road][driver][0][p_ind][v_ind]
                    cost_ttc[2] = cost_ttc[2] + self.cost_array_confort[road][driver][1][p_ind][v_ind]
                    cost_msd[2] = cost_msd[2] + self.cost_array_confort[road][driver][2][p_ind][v_ind]
                    cost_human[2] = cost_human[2] + self.cost_array_confort[road][driver][3][p_ind][v_ind]
                    
                    cost_vi[3] = cost_vi[3] + self.cost_array_override[road][driver][0][p_ind][v_ind]
                    cost_ttc[3] = cost_ttc[3] + self.cost_array_override[road][driver][1][p_ind][v_ind]
                    cost_msd[3] = cost_msd[3] + self.cost_array_override[road][driver][2][p_ind][v_ind]
                    cost_human[3] = cost_human[3] + self.cost_array_override[road][driver][3][p_ind][v_ind]
                    
                    counter = counter+1
        
        cost_vi = cost_vi/counter
        cost_ttc = cost_ttc/counter
        cost_msd = cost_msd/counter
        cost_human = cost_human/counter
        
        params = ("Total", "Sécurité", "Confort", "Liberté")
        x = np.arange(len(params))  # the label locations
        width = 0.20  # the width of the bars
        multiplier = 0
        
        arr = {
            'Total': (cost_vi[0], cost_ttc[0], cost_msd[0], cost_human[0]),
            'Sécurité': (cost_vi[1], cost_ttc[1], cost_msd[1], cost_human[1]),
            'Confort': (cost_vi[2], cost_ttc[2], cost_msd[2], cost_human[2]),
            'Liberté': (cost_vi[3], cost_ttc[3], cost_msd[3], cost_human[3])
            }
        max_val = np.zeros(4)
        max_val[0] = np.max(cost_vi)
        max_val[1] = np.max(cost_ttc)
        max_val[2] = np.max(cost_msd)
        max_val[3] = np.max(cost_human)
        maxmax = np.max(max_val)
            
        
        fig, ax = plt.subplots(figsize=(10,10))
        name = 'Route: ' + str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver]+'\nLe conducteur décide de ne pas freiner')
        fig.suptitle(name)
        for attribute, measurement in arr.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=5, label_type='edge', rotation=90)
            multiplier += 1
            
        ax.set_ylabel('Coûts à venir moyens')
        ax.set_ylim([0, maxmax+(0.1*maxmax)])
        ax.set_xticks(x + width, self.controlers_array)
        ax.legend(loc='upper left')  
        plt.tight_layout() 
        if save:
            plt.savefig(self.folder_dir+'\\MeanValues\\NoBraking\\'+str(road)+str(driver))
            plt.close('all')          
        
        
    def plot_single_road_multiple_controler(self,road,driver, save = False):
        fig, axs = plt.subplots(2,2, figsize=(8,8))
        plt.ion()
        name = 'Route: ' + str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver])
        fig.suptitle(name)
        ind = 0
        grid = self.metadata[-2][0]
        
        self.plot_function(fig,axs[0][0], grid, self.data_array[road][driver][0].T, name=self.controlers_array[0])
        self.plot_function(fig,axs[0][1], grid, self.data_array[road][driver][1].T, name=self.controlers_array[1])
        self.plot_function(fig,axs[1][0], grid, self.data_array[road][driver][2].T, name=self.controlers_array[2])
        self.plot_function(fig,axs[1][1], grid, self.data_array[road][driver][3].T, name=self.controlers_array[3])
        
        plt.tight_layout()
        if save:
            plt.savefig(self.folder_dir+'\\Road_Controlers\\Data\\'+str(road)+str(driver))
            plt.close('all')

    def plot_single_driver_multiple_controler(self,road, driver,save= False):
        fig, axs = plt.subplots(2,2, figsize=(8,8))
        plt.ion()
        name = 'Route: ' + str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver])
        fig.suptitle(name)
        ind = 0
        grid = self.metadata[-2][0]
        
        self.plot_function(fig,axs[0][0], grid, self.data_array[road][driver][0].T, name=self.controlers_array[0])
        self.plot_function(fig,axs[0][1], grid, self.data_array[road][driver][1].T, name=self.controlers_array[1])
        self.plot_function(fig,axs[1][0], grid, self.data_array[road][driver][2].T, name=self.controlers_array[2])
        self.plot_function(fig,axs[1][1], grid, self.data_array[road][driver][3].T, name=self.controlers_array[3])
        
        plt.tight_layout()
        if save:
            plt.savefig(self.folder_dir+'\\Driver_Controlers\\Data\\'+str(road)+str(driver))
            plt.close('all')
        
    def plot_single_controler_multiple_road(self, controler, driver):
        fig, axs = plt.subplots(3,2, figsize=(8,8))
        plt.ion()
        name = 'Loi de commandes: ' + str(self.controlers_array[controler]) + '\nConducteur: ' + str(self.drivers_array[driver])
        fig.suptitle(name)
        ind = 0
        grid = self.metadata[-2][0]
        
        self.plot_function(fig,axs[0][0], grid, self.data_array[0][driver][controler].T, name=self.roads_array[0])
        self.plot_function(fig,axs[1][0], grid, self.data_array[1][driver][controler].T, name=self.roads_array[1])
        self.plot_function(fig,axs[2][0], grid, self.data_array[2][driver][controler].T, name=self.roads_array[2])
        self.plot_function(fig,axs[0][1], grid, self.data_array[3][driver][controler].T, name=self.roads_array[3])
        self.plot_function(fig,axs[1][1], grid, self.data_array[4][driver][controler].T, name=self.roads_array[4])
        self.plot_function(fig,axs[2][1], grid, self.data_array[5][driver][controler].T, name=self.roads_array[5])
        
        plt.tight_layout()
        plt.savefig('image/DATA'+str(controler)+str(driver))   
        
    def plot_every_cont_for_evey_roads(self, driver,controler, save = False):
        for i in range(self.nbr_roads):
            fig, axs = plt.subplots(1,1, figsize=(8,8))
            plt.ion()
            name = 'Route: ' + str(self.roads_array[i]) + '\nLoi de commandes: '+ str(self.controlers_array[controler])
            fig.suptitle(name)
            grid = self.metadata[-2][0]
            
            self.plot_function(fig,axs, grid, self.data_array[i][driver][controler].T)
            
            plt.tight_layout()
            
            if save:
                plt.savefig(self.folder_dir+'\\SingleRoad_SingleControlers\\Data\\'+str(i)+str(driver)+str(controler))
                plt.close('all')

        
    def plot_every_c2g_for_evey_roads(self,driver,controler,save=False):
         for i in range(self.nbr_roads): 
             fig, axs = plt.subplots(2,2, figsize=(8,8))
             plt.ion()
             name = 'Route: ' + str(self.roads_array[i]) + '\nLoi de commandes: '+ str(self.controlers_array[controler])
             fig.suptitle(name)
             grid = self.metadata[-1][0]
             self.plot_function(fig,axs[0][0],grid,self.cost_array[i][driver][controler].T, name = 'Coûts à venir totaux')
             self.plot_function(fig,axs[0][1],grid,self.cost_array_security[i][driver][controler].T, name = 'Coûts à venir pour la sécurité')
             self.plot_function(fig,axs[1][0],grid,self.cost_array_confort[i][driver][controler].T, name = 'Coûts à venir pour le confort')
             self.plot_function(fig,axs[1][1],grid,self.cost_array_override[i][driver][controler].T, name = 'Coûts à venir pour la liberté')
             plt.tight_layout() 
             if save:
                 plt.savefig(self.folder_dir+'\\SingleRoad_SingleControlers\\Cost\\'+str(i)+str(driver)+str(controler))
                 plt.close('all')

    def plot_every_c2g_for_evey_roads_diff(self,driver,controler, controler2, save = False):
         for i in range(self.nbr_roads): 
             fig, axs = plt.subplots(2,2, figsize=(8,8))
             plt.ion()
             name = 'Route: ' + str(self.roads_array[i]) + '\nLoi de commandes: '+ str(self.controlers_array[controler]) + ' - ' + str(self.controlers_array[controler2]) + '\nSi Valeur positive: ' + str(self.controlers_array[controler]) + ' est moins bon que ' + str(self.controlers_array[controler2]) 
             fig.suptitle(name)
             grid = self.metadata[-1][0]             
             self.plot_function(fig,axs[0][0],grid,self.cost_array[i][driver][controler].T-self.cost_array[i][driver][controler2].T, name = 'Coûts à venir totaux')
             self.plot_function(fig,axs[0][1],grid,self.cost_array_security[i][driver][controler].T-self.cost_array_security[i][driver][controler2].T, name = 'Coûts à venir pour la sécurité')
             self.plot_function(fig,axs[1][0],grid,self.cost_array_confort[i][driver][controler].T-self.cost_array_confort[i][driver][controler2].T, name = 'Coûts à venir pour le confort')
             self.plot_function(fig,axs[1][1],grid,self.cost_array_override[i][driver][controler].T-self.cost_array_override[i][driver][controler2].T, name = 'Coûts à venir pour la liberté') 
             
             plt.tight_layout() 
             if save:
                 plt.savefig(self.folder_dir+'\\SingleRoad_SingleControlers\\CostDiff\\'+str(i)+str(driver)+'_'+str(controler)+'-'+str(controler2))
                 plt.close('all')

    def plot_every_c2g_for_evey_driver_diff(self,road, controler, controler2, save = False):
         for i in range(self.nbr_drivers): 
             fig, axs = plt.subplots(2,2, figsize=(8,8))
             plt.ion()
             name = 'Driver: ' + str(self.drivers_array[i]) + '\nLoi de commandes: '+ str(self.controlers_array[controler]) + ' - ' + str(self.controlers_array[controler2]) + '\nSi Valeur positive: ' + str(self.controlers_array[controler]) + ' est moins bon que ' + str(self.controlers_array[controler2]) 
             fig.suptitle(name)
             grid = self.metadata[-1][0]             
             self.plot_function(fig,axs[0][0],grid,self.cost_array[road][i][controler].T-self.cost_array[road][i][controler2].T, name = 'Coûts à venir totaux')
             self.plot_function(fig,axs[0][1],grid,self.cost_array_security[road][i][controler].T-self.cost_array_security[road][i][controler2].T, name = 'Coûts à venir pour la sécurité')
             self.plot_function(fig,axs[1][0],grid,self.cost_array_confort[road][i][controler].T-self.cost_array_confort[road][i][controler2].T, name = 'Coûts à venir pour le confort')
             self.plot_function(fig,axs[1][1],grid,self.cost_array_override[road][i][controler].T-self.cost_array_override[road][i][controler2].T, name = 'Coûts à venir pour la liberté') 
             
             plt.tight_layout() 
             if save:
                 plt.savefig(self.folder_dir+'\\Driver_Controlers\\CostDiff\\'+str(road)+str(i)+'_'+str(controler)+'-'+str(controler2))
                 plt.close('all')
            
    def plot_single_controler_multiple_road(self, controler, driver, save = False):
        fig, axs = plt.subplots(3,2, figsize=(8,8))
        plt.ion()
        name = 'Loi de commandes: ' + str(self.controlers_array[controler]) + '\nConducteur: ' + str(self.drivers_array[driver])
        fig.suptitle(name)
        ind = 0
        grid = self.metadata[-2][0]
        
        self.plot_function(fig, axs[0,0], grid, self.data_array[0][driver][controler].T, name= self.roads_array[0] + ' (\u03BC: ' + str(self.mu_array[0]) + ' Décel Max: '+str(self.dec_array[0]) + ')')
        self.plot_function(fig, axs[0,1], grid, self.data_array[1][driver][controler].T, name= self.roads_array[1] + ' (\u03BC: ' + str(self.mu_array[1]) + ' Décel Max: '+str(self.dec_array[1]) + ')')        
        self.plot_function(fig, axs[1,0], grid, self.data_array[2][driver][controler].T, name= self.roads_array[2] + ' (\u03BC: ' + str(self.mu_array[2]) + ' Décel Max: '+str(self.dec_array[2]) + ')')
        self.plot_function(fig, axs[1,1], grid, self.data_array[4][driver][controler].T, name= self.roads_array[4] + ' (\u03BC: ' + str(self.mu_array[4]) + ' Décel Max: '+str(self.dec_array[4]) + ')')
        self.plot_function(fig, axs[2,0], grid, self.data_array[5][driver][controler].T, name= self.roads_array[5] + ' (\u03BC: ' + str(self.mu_array[5]) + ' Décel Max: '+str(self.dec_array[5]) + ')')
        self.plot_function(fig, axs[2,1], grid, self.data_array[6][driver][controler].T, name= self.roads_array[6] + ' (\u03BC: ' + str(self.mu_array[6]) + ' Décel Max: '+str(self.dec_array[6]) + ')')

        plt.tight_layout()
        
        if save:
            plt.savefig(self.folder_dir+'\\AllRoads_singleControlers\\Data\\'+str(driver)+str(controler))
            plt.close('all')
   
    def plot_c2g_single_param(self,road,driver,save = False):
        fig, axs = plt.subplots(2,2, figsize=(8,8))
        name = 'Paramètre: Sécurité ' + '\nRoute: ' + str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver])
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        self.plot_function(fig,axs[0][0],grid,(self.cost_array_security[road][driver][0]).T, name = self.controlers_array[0])
        self.plot_function(fig,axs[0][1],grid,(self.cost_array_security[road][driver][1]-self.cost_array_security[road][driver][0]).T, name = self.controlers_array[1] + ' - ' + self.controlers_array[0])
        self.plot_function(fig,axs[1][0],grid,(self.cost_array_security[road][driver][2]-self.cost_array_security[road][driver][0]).T, name = self.controlers_array[2] + ' - ' + self.controlers_array[0])
        self.plot_function(fig,axs[1][1],grid,(self.cost_array_security[road][driver][3]-self.cost_array_security[road][driver][0]).T, name = self.controlers_array[3] + ' - ' + self.controlers_array[0])
        plt.tight_layout()
        if save:
            plt.savefig(self.folder_dir+'\\Road_Controlers\\CostDiff\\'+str(road)+str(driver)+'_sécuité')
            #plt.savefig(str(road)+str(driver)+'_sécurité')
            plt.close('all')

        fig, axs = plt.subplots(2,2, figsize=(8,8))
        name = 'Paramètre: Confort ' + '\nRoute: ' + str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver])
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        self.plot_function(fig,axs[0][0],grid,self.cost_array_confort[road][driver][0].T, name = self.controlers_array[0])
        self.plot_function(fig,axs[0][1],grid,(self.cost_array_confort[road][driver][1]-self.cost_array_confort[road][driver][0]).T, name = self.controlers_array[1] + ' - ' + self.controlers_array[0])
        self.plot_function(fig,axs[1][0],grid,(self.cost_array_confort[road][driver][2]-self.cost_array_confort[road][driver][0]).T, name = self.controlers_array[2] + ' - ' + self.controlers_array[0])
        self.plot_function(fig,axs[1][1],grid,(self.cost_array_confort[road][driver][3]-self.cost_array_confort[road][driver][0]).T, name = self.controlers_array[3] + ' - ' + self.controlers_array[0])
        plt.tight_layout() 
        if save:
            plt.savefig(self.folder_dir+'\\Road_Controlers\\CostDiff\\'+str(road)+str(driver)+'_confort')
            #plt.savefig(str(road)+str(driver)+'_confort')
            plt.close('all')
            
        fig, axs = plt.subplots(2,2, figsize=(8,8))
        name = 'Paramètre: Liberté ' + '\nRoute: ' + str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver])
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        self.plot_function(fig,axs[0][0],grid,self.cost_array_override[road][driver][0].T, name = self.controlers_array[0])
        self.plot_function(fig,axs[0][1],grid,(self.cost_array_override[road][driver][1]-self.cost_array_override[road][driver][0]).T, name = self.controlers_array[1] + ' - ' + self.controlers_array[0])
        self.plot_function(fig,axs[1][0],grid,(self.cost_array_override[road][driver][2]-self.cost_array_override[road][driver][0]).T, name = self.controlers_array[2] + ' - ' + self.controlers_array[0])
        self.plot_function(fig,axs[1][1],grid,(self.cost_array_override[road][driver][3]-self.cost_array_override[road][driver][0]).T, name = self.controlers_array[3] + ' - ' + self.controlers_array[0])
        plt.tight_layout() 
        if save:
            plt.savefig(self.folder_dir+'\\Road_Controlers\\CostDiff\\'+str(road)+str(driver)+'_liberté')
            #plt.savefig(str(road)+str(driver)+'_liberté')
            plt.close('all')

        fig, axs = plt.subplots(2,2, figsize=(8,8))
        name = 'Paramètre: Tous les paramètres ' + '\nRoute: ' + str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver])
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        self.plot_function(fig,axs[0][0],grid,self.cost_array[road][driver][0].T, name = self.controlers_array[0])
        self.plot_function(fig,axs[0][1],grid,(self.cost_array[road][driver][1]-self.cost_array[road][driver][0]).T, name = self.controlers_array[1] + ' - ' + self.controlers_array[0])
        self.plot_function(fig,axs[1][0],grid,(self.cost_array[road][driver][2]-self.cost_array[road][driver][0]).T, name = self.controlers_array[2] + ' - ' + self.controlers_array[0])
        self.plot_function(fig,axs[1][1],grid,(self.cost_array[road][driver][3]-self.cost_array[road][driver][0]).T, name = self.controlers_array[3] + ' - ' + self.controlers_array[0])
        plt.tight_layout() 
        if save:
            plt.savefig(self.folder_dir+'\\Road_Controlers\\CostDiff\\'+str(road)+str(driver)+'_all')
            #plt.savefig(str(road)+str(driver)+'_all')
            plt.close('all')

    def plot_c2g_single_param_driver(self,road,driver,save = False):
        fig, axs = plt.subplots(2,2, figsize=(8,8))
        name = 'Paramètre: Sécurité ' + '\nRoute: ' + str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver])
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        self.plot_function(fig,axs[0][0],grid,(self.cost_array_security[road][driver][0]).T, name = self.controlers_array[0])
        self.plot_function(fig,axs[0][1],grid,(self.cost_array_security[road][driver][1]-self.cost_array_security[road][driver][0]).T, name = self.controlers_array[1] + ' - ' + self.controlers_array[0])
        self.plot_function(fig,axs[1][0],grid,(self.cost_array_security[road][driver][2]-self.cost_array_security[road][driver][0]).T, name = self.controlers_array[2] + ' - ' + self.controlers_array[0])
        self.plot_function(fig,axs[1][1],grid,(self.cost_array_security[road][driver][3]-self.cost_array_security[road][driver][0]).T, name = self.controlers_array[3] + ' - ' + self.controlers_array[0])
        plt.tight_layout()
        if save:
            plt.savefig(self.folder_dir+'\\Driver_Controlers\\CostDiff\\'+str(road)+str(driver)+'_sécuité')
            plt.close('all')

        fig, axs = plt.subplots(2,2, figsize=(8,8))
        name = 'Paramètre: Confort ' + '\nRoute: ' + str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver])
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        self.plot_function(fig,axs[0][0],grid,self.cost_array_confort[road][driver][0].T, name = self.controlers_array[0])
        self.plot_function(fig,axs[0][1],grid,(self.cost_array_confort[road][driver][1]-self.cost_array_confort[road][driver][0]).T, name = self.controlers_array[1] + ' - ' + self.controlers_array[0])
        self.plot_function(fig,axs[1][0],grid,(self.cost_array_confort[road][driver][2]-self.cost_array_confort[road][driver][0]).T, name = self.controlers_array[2] + ' - ' + self.controlers_array[0])
        self.plot_function(fig,axs[1][1],grid,(self.cost_array_confort[road][driver][3]-self.cost_array_confort[road][driver][0]).T, name = self.controlers_array[3] + ' - ' + self.controlers_array[0])
        plt.tight_layout() 
        if save:
            plt.savefig(self.folder_dir+'\\Driver_Controlers\\CostDiff\\'+str(road)+str(driver)+'_confort')
            plt.close('all')
            
        fig, axs = plt.subplots(2,2, figsize=(8,8))
        name = 'Paramètre: Liberté ' + '\nRoute: ' + str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver])
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        self.plot_function(fig,axs[0][0],grid,self.cost_array_override[road][driver][0].T, name = self.controlers_array[0])
        self.plot_function(fig,axs[0][1],grid,(self.cost_array_override[road][driver][1]-self.cost_array_override[road][driver][0]).T, name = self.controlers_array[1] + ' - ' + self.controlers_array[0])
        self.plot_function(fig,axs[1][0],grid,(self.cost_array_override[road][driver][2]-self.cost_array_override[road][driver][0]).T, name = self.controlers_array[2] + ' - ' + self.controlers_array[0])
        self.plot_function(fig,axs[1][1],grid,(self.cost_array_override[road][driver][3]-self.cost_array_override[road][driver][0]).T, name = self.controlers_array[3] + ' - ' + self.controlers_array[0])
        plt.tight_layout() 
        if save:
            plt.savefig(self.folder_dir+'\\Driver_Controlers\\CostDiff\\'+str(road)+str(driver)+'_liberté')
            plt.close('all')

        fig, axs = plt.subplots(2,2, figsize=(8,8))
        name = 'Paramètre: Tous les paramètres ' + '\nRoute: ' + str(self.roads_array[road]) + '\nConducteur: ' + str(self.drivers_array[driver])
        fig.suptitle(name)
        grid = self.metadata[-1][0]
        self.plot_function(fig,axs[0][0],grid,self.cost_array[road][driver][0].T, name = self.controlers_array[0])
        self.plot_function(fig,axs[0][1],grid,(self.cost_array[road][driver][1]-self.cost_array[road][driver][0]).T, name = self.controlers_array[1] + ' - ' + self.controlers_array[0])
        self.plot_function(fig,axs[1][0],grid,(self.cost_array[road][driver][2]-self.cost_array[road][driver][0]).T, name = self.controlers_array[2] + ' - ' + self.controlers_array[0])
        self.plot_function(fig,axs[1][1],grid,(self.cost_array[road][driver][3]-self.cost_array[road][driver][0]).T, name = self.controlers_array[3] + ' - ' + self.controlers_array[0])
        plt.tight_layout() 
        if save:
            plt.savefig(self.folder_dir+'\\Driver_Controlers\\CostDiff\\'+str(road)+str(driver)+'_all')
            plt.close('all')


    
    def plot_drivers(self, road, controler):
        fig, axs = plt.subplots(3,3, figsize=(8,12))
        plt.ion()
        ind = 0
        grid = self.metadata[-2][0]
        self.plot_function(fig, axs[0][0], grid, (self.cost_array_security[road][0][controler] - self.cost_array_security[road][4][controler]).T, name=self.drivers_array[0]+' Sécurité')
        self.plot_function(fig, axs[0][1], grid, (self.cost_array_confort[road][0][controler] - self.cost_array_confort[road][4][controler]).T, name=self.drivers_array[0]+' Confort')
        self.plot_function(fig, axs[0][2], grid, (self.cost_array_override[road][0][controler] - self.cost_array_override[road][4][controler]).T, name=self.drivers_array[0]+' Liberté')

        self.plot_function(fig, axs[1][0], grid, (self.cost_array_security[road][1][controler] - self.cost_array_security[road][4][controler]).T, name=self.drivers_array[1]+' Sécurité')
        self.plot_function(fig, axs[1][1], grid, (self.cost_array_confort[road][1][controler] - self.cost_array_confort[road][4][controler]).T, name=self.drivers_array[1]+' Confort')
        self.plot_function(fig, axs[1][2], grid, (self.cost_array_override[road][1][controler] - self.cost_array_override[road][4][controler]).T, name=self.drivers_array[1]+' Liberté')

        self.plot_function(fig, axs[2][0], grid, (self.cost_array_security[road][2][controler] - self.cost_array_security[road][4][controler]).T, name=self.drivers_array[2]+' Sécurit.')
        self.plot_function(fig, axs[2][1], grid, (self.cost_array_confort[road][2][controler] - self.cost_array_confort[road][4][controler]).T, name=self.drivers_array[2]+' Confort')
        self.plot_function(fig, axs[2][2], grid, (self.cost_array_override[road][2][controler] - self.cost_array_override[road][4][controler]).T, name=self.drivers_array[2]+' Liberté')

        
        plt.tight_layout()

    def plot_drivers_gnb(self, road):
        fig, axs = plt.subplots(1,3, figsize=(12,12))
        plt.ion()
        ind = 0
        grid = self.metadata[-2][0]
        self.plot_function(fig, axs[0], grid, self.data_array[road][0][0].T, name=self.drivers_array[0])
        self.plot_function(fig, axs[1], grid, self.data_array[road][1][0].T, name=self.drivers_array[1])
        self.plot_function(fig, axs[2], grid, self.data_array[road][2][0].T, name=self.drivers_array[2])
        plt.tight_layout()
        
    def save_everything_roads(self):
        path_desktop = 'C:\\Users\\Charles-Alexis\\Desktop\\Image'
        now = datetime.datetime.now()
        directory = now.strftime("ROADS_%m_%d_%Y_%H_%M_%S")
        self.folder_dir = os.path.join(path_desktop, directory)
        os.mkdir(self.folder_dir)
        
        os.mkdir(self.folder_dir+'\\Road_Controlers')
        os.mkdir(self.folder_dir+'\\Road_Controlers\\Data')
        os.mkdir(self.folder_dir+'\\Road_Controlers\\Cost')
        os.mkdir(self.folder_dir+'\\Road_Controlers\\CostDiff')
        self.plot_single_road_multiple_controler(0,4,save = True)
        self.plot_single_road_multiple_controler(1,4,save = True)
        self.plot_single_road_multiple_controler(2,4,save = True)
        self.plot_single_road_multiple_controler(3,4,save = True)
        self.plot_single_road_multiple_controler(4,4,save = True)
        self.plot_single_road_multiple_controler(5,4,save = True)
        self.plot_single_road_multiple_controler(6,4,save = True)
        self.plot_cost(0,4,save = True)
        self.plot_cost(1,4,save = True)
        self.plot_cost(2,4,save = True)
        self.plot_cost(3,4,save = True)
        self.plot_cost(4,4,save = True)
        self.plot_cost(5,4,save = True)
        self.plot_cost(6,4,save = True)
        self.plot_c2g_single_param(0,4,save = True)
        self.plot_c2g_single_param(1,4,save = True)
        self.plot_c2g_single_param(2,4,save = True)
        self.plot_c2g_single_param(3,4,save = True)
        self.plot_c2g_single_param(4,4,save = True)
        self.plot_c2g_single_param(5,4,save = True)
        self.plot_c2g_single_param(6,4,save = True)
        
        os.mkdir(self.folder_dir+'\\AllRoads_singleControlers')
        os.mkdir(self.folder_dir+'\\AllRoads_singleControlers\\Data')
        os.mkdir(self.folder_dir+'\\AllRoads_singleControlers\\Cost')
        self.plot_single_controler_multiple_road(0,4,save = True)
        self.plot_single_controler_multiple_road(1,4,save = True)
        self.plot_single_controler_multiple_road(2,4,save = True)
        self.plot_single_controler_multiple_road(3,4,save = True)

        os.mkdir(self.folder_dir+'\\SingleRoad_SingleControlers')
        os.mkdir(self.folder_dir+'\\SingleRoad_SingleControlers\\Data')
        os.mkdir(self.folder_dir+'\\SingleRoad_SingleControlers\\Cost')
        os.mkdir(self.folder_dir+'\\SingleRoad_SingleControlers\\CostDiff')
        self.plot_every_cont_for_evey_roads(4,0,save = True)
        self.plot_every_cont_for_evey_roads(4,1,save = True)
        self.plot_every_cont_for_evey_roads(4,2,save = True)
        self.plot_every_cont_for_evey_roads(4,3,save = True)
        
        self.plot_every_c2g_for_evey_roads(4,0,save = True)
        self.plot_every_c2g_for_evey_roads(4,1,save = True)
        self.plot_every_c2g_for_evey_roads(4,2,save = True)
        self.plot_every_c2g_for_evey_roads(4,3,save = True)

        self.plot_every_c2g_for_evey_roads_diff(4,0,1,save = True)
        self.plot_every_c2g_for_evey_roads_diff(4,0,2,save = True)
        self.plot_every_c2g_for_evey_roads_diff(4,0,3,save = True)
        self.plot_every_c2g_for_evey_roads_diff(4,1,3,save = True)
        self.plot_every_c2g_for_evey_roads_diff(4,2,3,save = True)
 
        os.mkdir(self.folder_dir+'\\MeanValues')
        os.mkdir(self.folder_dir+'\\MeanValues\\All')
        os.mkdir(self.folder_dir+'\\MeanValues\\Fast')
        os.mkdir(self.folder_dir+'\\MeanValues\\Close')
        os.mkdir(self.folder_dir+'\\MeanValues\\Braking')
        os.mkdir(self.folder_dir+'\\MeanValues\\NoBraking')
        
        self.print_resultats_fast(0,4, save=True, name_mean='All')
        self.print_resultats_fast(1,4, save=True, name_mean='All')
        self.print_resultats_fast(2,4, save=True, name_mean='All')
        self.print_resultats_fast(3,4, save=True, name_mean='All')
        self.print_resultats_fast(4,4, save=True, name_mean='All')
        self.print_resultats_fast(5,4, save=True, name_mean='All')
        self.print_resultats_fast(6,4, save=True, name_mean='All')
        
        self.print_resultats_fast(0,4,vit=[120,150], save=True, name_mean='Fast')
        self.print_resultats_fast(1,4,vit=[120,150], save=True, name_mean='Fast')
        self.print_resultats_fast(2,4,vit=[120,150], save=True, name_mean='Fast')
        self.print_resultats_fast(3,4,vit=[120,150], save=True, name_mean='Fast')
        self.print_resultats_fast(4,4,vit=[120,150], save=True, name_mean='Fast')
        self.print_resultats_fast(5,4,vit=[120,150], save=True, name_mean='Fast')
        self.print_resultats_fast(6,4,vit=[120,150], save=True, name_mean='Fast')
        
        self.print_resultats_fast(0,4,pos=[120,150], save=True, name_mean='Close')
        self.print_resultats_fast(1,4,pos=[120,150], save=True, name_mean='Close')
        self.print_resultats_fast(2,4,pos=[120,150], save=True, name_mean='Close')
        self.print_resultats_fast(3,4,pos=[120,150], save=True, name_mean='Close')
        self.print_resultats_fast(4,4,pos=[120,150], save=True, name_mean='Close')
        self.print_resultats_fast(5,4,pos=[120,150], save=True, name_mean='Close')
        self.print_resultats_fast(6,4,pos=[120,150], save=True, name_mean='Close')
        
        self.print_resultats_human_no_braking(0,4,save=True)
        self.print_resultats_human_no_braking(1,4,save=True)
        self.print_resultats_human_no_braking(2,4,save=True)
        self.print_resultats_human_no_braking(3,4,save=True)
        self.print_resultats_human_no_braking(4,4,save=True)
        self.print_resultats_human_no_braking(5,4,save=True)
        self.print_resultats_human_no_braking(6,4,save=True)
        
        self.print_resultats_human_braking(0,4,save=True)
        self.print_resultats_human_braking(1,4,save=True)
        self.print_resultats_human_braking(2,4,save=True)
        self.print_resultats_human_braking(3,4,save=True)
        self.print_resultats_human_braking(4,4,save=True)
        self.print_resultats_human_braking(5,4,save=True)
        self.print_resultats_human_braking(6,4,save=True)        
        
    def save_everything_driver(self, road):
        path_desktop = 'C:\\Users\\Charles-Alexis\\Desktop\\Image'
        now = datetime.datetime.now()
        directory = now.strftime("DRIVER"+str(road)+"_%m_%d_%Y_%H_%M_%S")
        self.folder_dir = os.path.join(path_desktop, directory)
        os.mkdir(self.folder_dir)
        
        os.mkdir(self.folder_dir+'\\Driver_Controlers')
        os.mkdir(self.folder_dir+'\\Driver_Controlers\\Data')
        os.mkdir(self.folder_dir+'\\Driver_Controlers\\Cost')
        os.mkdir(self.folder_dir+'\\Driver_Controlers\\CostDiff')
        self.plot_single_driver_multiple_controler(road,0,save=True)
        self.plot_single_driver_multiple_controler(road,1,save=True)
        self.plot_single_driver_multiple_controler(road,2,save=True)
        self.plot_single_driver_multiple_controler(road,3,save=True)
        self.plot_cost_driver(road, 0,save=True)
        self.plot_cost_driver(road, 1,save=True)
        self.plot_cost_driver(road, 2,save=True)
        self.plot_cost_driver(road, 3,save=True)
        self.plot_every_c2g_for_evey_driver_diff(road, 0, 1,save=True)
        self.plot_every_c2g_for_evey_driver_diff(road, 0, 2,save=True)
        self.plot_every_c2g_for_evey_driver_diff(road, 0, 3,save=True)
        self.plot_every_c2g_for_evey_driver_diff(road, 1, 3,save=True)
        self.plot_every_c2g_for_evey_driver_diff(road, 2, 3,save=True)
        self.plot_c2g_single_param_driver(road, 0, save=True)
        self.plot_c2g_single_param_driver(road, 1, save=True)
        self.plot_c2g_single_param_driver(road, 2, save=True)
        self.plot_c2g_single_param_driver(road, 3, save=True)
        self.plot_c2g_single_param_driver(road, 4, save=True)
        
        
        os.mkdir(self.folder_dir+'\\MeanValues')
        os.mkdir(self.folder_dir+'\\MeanValues\\All')
        os.mkdir(self.folder_dir+'\\MeanValues\\Fast')
        os.mkdir(self.folder_dir+'\\MeanValues\\Close')
        os.mkdir(self.folder_dir+'\\MeanValues\\Braking')
        os.mkdir(self.folder_dir+'\\MeanValues\\NoBraking')
        
        self.print_resultats_fast(road,0, save=True, name_mean='All')
        self.print_resultats_fast(road,1, save=True, name_mean='All')
        self.print_resultats_fast(road,2, save=True, name_mean='All')
        self.print_resultats_fast(road,3, save=True, name_mean='All')
        self.print_resultats_fast(road,4, save=True, name_mean='All')
        
        self.print_resultats_fast(road,0,vit=[120,150], save=True, name_mean='Fast')
        self.print_resultats_fast(road,1,vit=[120,150], save=True, name_mean='Fast')
        self.print_resultats_fast(road,2,vit=[120,150], save=True, name_mean='Fast')
        self.print_resultats_fast(road,3,vit=[120,150], save=True, name_mean='Fast')
        self.print_resultats_fast(road,4,vit=[120,150], save=True, name_mean='Fast')
        
        self.print_resultats_fast(road,0,pos=[120,150], save=True, name_mean='Close')
        self.print_resultats_fast(road,1,pos=[120,150], save=True, name_mean='Close')
        self.print_resultats_fast(road,2,pos=[120,150], save=True, name_mean='Close')
        self.print_resultats_fast(road,3,pos=[120,150], save=True, name_mean='Close')
        self.print_resultats_fast(road,4,pos=[120,150], save=True, name_mean='Close')
        
        self.print_resultats_human_no_braking(road,0,save=True)
        self.print_resultats_human_no_braking(road,1,save=True)
        self.print_resultats_human_no_braking(road,2,save=True)
        self.print_resultats_human_no_braking(road,3,save=True)
        self.print_resultats_human_no_braking(road,4,save=True)
        
        self.print_resultats_human_braking(road,0,save=True) 
        self.print_resultats_human_braking(road,1,save=True) 
        self.print_resultats_human_braking(road,2,save=True) 
        self.print_resultats_human_braking(road,3,save=True) 
        self.print_resultats_human_braking(road,4,save=True)      
        
        