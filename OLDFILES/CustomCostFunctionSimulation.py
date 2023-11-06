# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:02:23 2022

@author: Charles-Alexis
"""
import numpy as np
import matplotlib.pyplot as plt
from pyro.analysis import costfunction

class CustomCostFunctionSimulation():
    ############################
    def __init__(self, traj, traj2, cost_function):
        self.traj = traj
        self.traj2 = traj2
        self.traj.g = np.zeros((3, np.size(self.traj.t)))
        self.traj2.g = np.zeros((3, np.size(self.traj2.t)))
        self.cost_function = cost_function
        self.x0_lim = 100.
        self.x1_lim = 1.
        
    def compute_cost_function(self):
        i = 0
        for time in self.traj.t:
            self.traj.g[0][i] = self.cost_function.g_security(self.traj.x[i],self.traj.u[i],0,0)
            self.traj.g[1][i] = self.cost_function.g_confort(self.traj.x[i],self.traj.u[i],0,0)
            self.traj.g[2][i] = self.cost_function.g_override(self.traj.x[i],self.traj.u[i],0,0)
            i = i + 1
        i = 0
        for time in self.traj2.t:            
            self.traj2.g[0][i] = self.cost_function.g_security(self.traj2.x[i],self.traj2.u[i],0,0)
            self.traj2.g[1][i] = self.cost_function.g_confort(self.traj2.x[i],self.traj2.u[i],0,0)
            self.traj2.g[2][i] = self.cost_function.g_override(self.traj2.x[i],self.traj2.u[i],0,0)
            i = i + 1

        
    def plot_multiple_g(self):
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(self.traj.t, self.traj.g[0], label='Security')
        axs[0, 0].plot(self.traj.t, self.traj.g[1], label='Confort')
        axs[0, 0].plot(self.traj.t, self.traj.g[2], label='Override')
        axs[0, 0].plot(self.traj.t, self.traj.J, label='J')
        axs[0, 0].set_title('TTC')
        axs[0, 0].legend()
        
        axs[1, 0].plot(self.traj.t, self.traj.x[:,0], label='Position')
        axs[1, 0].plot(self.traj.t, self.traj.dx[:,0], label='Vitesse')
        axs[1, 0].plot(self.traj.t, self.traj.dx[:,1], label='Acc')
        axs[1, 0].plot(self.traj.t, self.traj.J, label='J')
        axs[1, 0].set_title('TTC')
        axs[1, 0].legend()
        
        axs[0, 1].plot(self.traj.t, self.traj2.g[0], label='Security')
        axs[0, 1].plot(self.traj.t, self.traj2.g[1], label='Confort')
        axs[0, 1].plot(self.traj.t, self.traj2.g[2], label='Override')
        axs[0, 1].plot(self.traj.t, self.traj2.J, label='J')
        axs[0, 1].set_title('VI')
        axs[0, 1].legend()
        
        axs[1, 1].plot(self.traj.t, self.traj2.x[:,0], label='Position')
        axs[1, 1].plot(self.traj.t, self.traj2.dx[:,0], label='Vitesse')
        axs[1, 1].plot(self.traj.t, self.traj2.dx[:,1], label='Acc')
        axs[1, 1].plot(self.traj.t, self.traj2.J, label='J')
        axs[1, 1].set_title('VI')
        axs[1, 1].legend()        
        
        # plt.figure()
        # ax1 = plt.subplot(411)
        # ax1.title.set_text('TTC')
        # plt.plot(self.traj.t, self.traj.g[0], label='Security')
        # plt.plot(self.traj.t, self.traj.g[1], label='Confort')
        # plt.plot(self.traj.t, self.traj.g[2], label='Override')
        # plt.plot(self.traj.t, self.traj.J, label='J')
        # ax1.legend()
        
        # ax2 = plt.subplot(412)
        # plt.plot(self.traj.t, self.traj.x[:,0], label='Position')
        # plt.plot(self.traj.t, self.traj.dx[:,0], label='Vitesse')
        # plt.plot(self.traj.t, self.traj.dx[:,1], label='Acc')
        # plt.plot(self.traj.t, self.traj.J, label='J')
        # ax2.legend()
        # plt.show()
        
        # ax3 = plt.subplot(423)
        # ax3.title.set_text('Vi')
        # plt.plot(self.traj.t, self.traj2.g[0], label='Security')
        # plt.plot(self.traj.t, self.traj2.g[1], label='Confort')
        # plt.plot(self.traj.t, self.traj2.g[2], label='Override')
        # plt.plot(self.traj.t, self.traj2.J, label='J')
        # ax3.legend()
        
        # ax4 = plt.subplot(424)
        # plt.plot(self.traj.t, self.traj2.x[:,0], label='Position')
        # plt.plot(self.traj.t, self.traj2.dx[:,0], label='Vitesse')
        # plt.plot(self.traj.t, self.traj2.dx[:,1], label='Acc')
        # plt.plot(self.traj.t, self.traj2.J, label='J')
        # ax4.legend()
        # plt.show()

    def plot_cost_functions(self):
        plt.figure()
        ax1 = plt.subplot(411)
        ax1.title.set_text('J')
        plt.plot(self.traj.t, self.traj.J, label='TTC')
        plt.plot(self.traj2.t, self.traj2.J, label='VI')
        ax2 = plt.subplot(412)
        ax2.title.set_text('dJ')
        plt.plot(self.traj.t, self.traj.dJ)
        plt.plot(self.traj2.t, self.traj2.dJ)
        ax3 = plt.subplot(413)
        ax3.title.set_text('Position')
        plt.plot(self.traj.t, self.traj.x[:,0])
        plt.plot(self.traj2.t, self.traj2.x[:,0])
        ax3 = plt.subplot(414)
        ax3.title.set_text('Vitesse')
        plt.plot(self.traj.t, self.traj.x[:,1])
        plt.plot(self.traj2.t, self.traj2.x[:,1])
        ax1.legend()
        plt.show()
      
    def under_x(self, array, x):
        i = 0
        flag = False
        while flag is False:
            if array[i] < x:
                flag = True
            i=i+1
        return i
      
    def plot_multiple_g_add(self):
        g_0 = np.cumsum(self.traj.g[0])*self.traj.tf/self.traj.steps
        g_01 = np.cumsum(self.traj.g[0]+self.traj.g[1])*self.traj.tf/self.traj.steps
        g_012 = np.cumsum(self.traj.g[0]+self.traj.g[1]+self.traj.g[2])*self.traj.tf/self.traj.steps
        
        g2_0 = np.cumsum(self.traj2.g[0])*self.traj2.tf/self.traj2.steps
        g2_01 = np.cumsum(self.traj2.g[0]+self.traj2.g[1])*self.traj2.tf/self.traj2.steps
        g2_012 = np.cumsum(self.traj2.g[0]+self.traj2.g[1]+self.traj2.g[2])*self.traj2.tf/self.traj2.steps
        
        v_0 = self.under_x(self.traj.x[:,1],self.x1_lim)
        v2_0 = self.under_x(self.traj.x[:,1],self.x1_lim)
        
      
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].fill_between(self.traj.t, g_012, label='Override')
        axs[0, 0].plot(self.traj.t, g_012, drawstyle="steps")
        axs[0, 0].fill_between(self.traj.t, g_01, label='Confort')
        axs[0, 0].plot(self.traj.t, g_01, drawstyle="steps")
        axs[0, 0].fill_between(self.traj.t, g_0, label='Security')
        axs[0, 0].plot(self.traj.t, g_0, drawstyle="steps")
        axs[0, 0].plot(self.traj.t, self.traj.J, label='J')
        
        axs[0, 0].axvline(self.traj.t[v_0], color='black')
        axs[0, 0].axhline(self.traj.J[v_0], color='black')
        
        axs[0, 0].set_title('TTC')
        axs[0, 0].legend()
        
        axs[1, 0].plot(self.traj.t, self.traj.x[:,0], label='Position')
        axs[1, 0].plot(self.traj.t, self.traj.dx[:,0], label='Vitesse')
        axs[1, 0].plot(self.traj.t, self.traj.dx[:,1], label='Acc')
        
        axs10_2 = axs[1, 0].twinx()
        axs10_2.plot(self.traj.t, self.traj.J, label='J', color='red')
        axs[1, 0].set_title('TTC')
        axs[1, 0].legend()
        axs10_2.legend()
        
        
        axs[0, 1].fill_between(self.traj2.t, g2_012, label='Override')
        axs[0, 1].plot(self.traj2.t, g2_012, drawstyle="steps")
        axs[0, 1].fill_between(self.traj2.t, g2_01, label='Confort')
        axs[0, 1].plot(self.traj2.t, g2_01, drawstyle="steps")
        axs[0, 1].fill_between(self.traj2.t, g2_0, label='Security')
        axs[0, 1].plot(self.traj2.t, g2_0, drawstyle="steps")
        axs[0, 1].plot(self.traj2.t, self.traj2.J, label='J')
        
        axs[0, 1].axvline(self.traj2.t[v_0], color='black')
        axs[0, 1].axhline(self.traj2.J[v_0], color='black')
        axs[0, 1].set_title('VI')
        axs[0, 1].legend()
        
        axs[1, 1].plot(self.traj2.t, self.traj2.x[:,0], label='Position')
        axs[1, 1].plot(self.traj2.t, self.traj2.dx[:,0], label='Vitesse')
        axs[1, 1].plot(self.traj2.t, self.traj2.dx[:,1], label='Acc')
                
        axs11_2 = axs[1, 1].twinx()
        axs11_2.plot(self.traj2.t, self.traj2.J, label='J', color='red')
        axs[1, 1].set_title('VI')
        axs[1, 1].legend()
        
class CustomCostFunctionSimulationSim():
    ############################
    def __init__(self, traj, traj2, cost_function):
        self.traj = traj
        self.traj2 = traj2
        self.traj.g = np.zeros((3, np.size(self.traj.t)))
        self.traj2.g = np.zeros((3, np.size(self.traj2.t)))
        self.cost_function = cost_function
        self.x0_lim = 100.
        self.x1_lim = 1.
        
    def compute_cost_function(self):
        i = 0
        for time in self.traj.t:
            self.traj.g[0][i] = self.cost_function.g_security(self.traj.x[i],self.traj.u[i],0,0)
            self.traj.g[1][i] = self.cost_function.g_confort(self.traj.x[i],self.traj.u[i],0,0)
            self.traj.g[2][i] = self.cost_function.g_override(self.traj.x[i],self.traj.u[i],0,0)
            i = i + 1
        i = 0
        for time in self.traj2.t:            
            self.traj2.g[0][i] = self.cost_function.g_security(self.traj2.x[i],self.traj2.u[i],0,0)
            self.traj2.g[1][i] = self.cost_function.g_confort(self.traj2.x[i],self.traj2.u[i],0,0)
            self.traj2.g[2][i] = self.cost_function.g_override(self.traj2.x[i],self.traj2.u[i],0,0)
            i = i + 1

        
    def plot_multiple_g(self):
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(self.traj.t, self.traj.g[0], label='Security')
        axs[0, 0].plot(self.traj.t, self.traj.g[1], label='Confort')
        axs[0, 0].plot(self.traj.t, self.traj.g[2], label='Override')
        axs[0, 0].plot(self.traj.t, self.traj.J, label='J')
        axs[0, 0].set_title('TTC')
        axs[0, 0].legend()
        
        axs[1, 0].plot(self.traj.t, self.traj.x[:,0], label='Position')
        axs[1, 0].plot(self.traj.t, self.traj.dx[:,0], label='Vitesse')
        axs[1, 0].plot(self.traj.t, self.traj.dx[:,1], label='Acc')
        axs[1, 0].plot(self.traj.t, self.traj.J, label='J')
        axs[1, 0].set_title('TTC')
        axs[1, 0].legend()
        
        axs[0, 1].plot(self.traj2.t, self.traj2.g[0], label='Security')
        axs[0, 1].plot(self.traj2.t, self.traj2.g[1], label='Confort')
        axs[0, 1].plot(self.traj2.t, self.traj2.g[2], label='Override')
        axs[0, 1].plot(self.traj2.t, self.traj2.J, label='J')
        axs[0, 1].set_title('VI')
        axs[0, 1].legend()
        
        axs[1, 1].plot(self.traj2.t, self.traj2.x[:,0], label='Position')
        axs[1, 1].plot(self.traj2.t, self.traj2.dx[:,0], label='Vitesse')
        axs[1, 1].plot(self.traj2.t, self.traj2.dx[:,1], label='Acc')
        axs[1, 1].plot(self.traj2.t, self.traj2.J, label='J')
        axs[1, 1].set_title('VI')
        axs[1, 1].legend()        
        

    def plot_cost_functions(self):
        plt.figure()
        ax1 = plt.subplot(411)
        ax1.title.set_text('J')
        plt.plot(self.traj.t, self.traj.J, label='TTC')
        plt.plot(self.traj2.t, self.traj2.J, label='VI')
        ax2 = plt.subplot(412)
        ax2.title.set_text('dJ')
        plt.plot(self.traj.t, self.traj.dJ)
        plt.plot(self.traj2.t, self.traj2.dJ)
        ax3 = plt.subplot(413)
        ax3.title.set_text('Position')
        plt.plot(self.traj.t, self.traj.x[:,0])
        plt.plot(self.traj2.t, self.traj2.x[:,0])
        ax3 = plt.subplot(414)
        ax3.title.set_text('Vitesse')
        plt.plot(self.traj.t, self.traj.x[:,1])
        plt.plot(self.traj2.t, self.traj2.x[:,1])
        ax1.legend()
        plt.show()
      
    def plot_multiple_g_add(self):
        g_0 = np.cumsum(self.traj.g[0])*self.traj.tf/self.traj.steps
        g_01 = np.cumsum(self.traj.g[0]+self.traj.g[1])*self.traj.tf/self.traj.steps
        g_012 = np.cumsum(self.traj.g[0]+self.traj.g[1]+self.traj.g[2])*self.traj.tf/self.traj.steps
        
        g2_0 = np.cumsum(self.traj2.g[0])*self.traj2.tf/self.traj2.steps
        g2_01 = np.cumsum(self.traj2.g[0]+self.traj2.g[1])*self.traj2.tf/self.traj2.steps
        g2_012 = np.cumsum(self.traj2.g[0]+self.traj2.g[1]+self.traj2.g[2])*self.traj2.tf/self.traj2.steps
        
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].fill_between(self.traj.t, g_012, label='Override')
        axs[0, 0].plot(self.traj.t, g_012, drawstyle="steps")
        axs[0, 0].fill_between(self.traj.t, g_01, label='Confort')
        axs[0, 0].plot(self.traj.t, g_01, drawstyle="steps")
        axs[0, 0].fill_between(self.traj.t, g_0, label='Security')
        axs[0, 0].plot(self.traj.t, g_0, drawstyle="steps")
        axs[0, 0].plot(self.traj.t, self.traj.J, label='J')
        
        axs[0, 0].set_title('TTC')
        axs[0, 0].legend()
        
        axs[1, 0].plot(self.traj.t, self.traj.x[:,0], label='Position')
        axs[1, 0].plot(self.traj.t, self.traj.dx[:,0], label='Vitesse')
        axs[1, 0].plot(self.traj.t, self.traj.dx[:,1], label='Acc')
        
        axs10_2 = axs[1, 0].twinx()
        axs10_2.plot(self.traj.t, self.traj.J, label='J', color='red')
        axs[1, 0].set_title('TTC')
        axs[1, 0].legend()
        axs10_2.legend()
        
        
        axs[0, 1].fill_between(self.traj2.t, g2_012, label='Override')
        axs[0, 1].plot(self.traj2.t, g2_012, drawstyle="steps")
        axs[0, 1].fill_between(self.traj2.t, g2_01, label='Confort')
        axs[0, 1].plot(self.traj2.t, g2_01, drawstyle="steps")
        axs[0, 1].fill_between(self.traj2.t, g2_0, label='Security')
        axs[0, 1].plot(self.traj2.t, g2_0, drawstyle="steps")
        axs[0, 1].plot(self.traj2.t, self.traj2.J, label='J')
        axs[0, 1].set_title('VI')
        axs[0, 1].legend()
        
        axs[1, 1].plot(self.traj2.t, self.traj2.x[:,0], label='Position')
        axs[1, 1].plot(self.traj2.t, self.traj2.dx[:,0], label='Vitesse')
        axs[1, 1].plot(self.traj2.t, self.traj2.dx[:,1], label='Acc')
                
        axs11_2 = axs[1, 1].twinx()
        axs11_2.plot(self.traj2.t, self.traj2.J, label='J', color='red')
        axs[1, 1].set_title('VI')
        axs[1, 1].legend()
        