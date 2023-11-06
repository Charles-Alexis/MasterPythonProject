# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:54:16 2022

@author: Charles-Alexis
"""

import numpy as np
import matplotlib.pyplot as plt
from pyro.analysis import simulation
import costfunction

class SimulatorV2:
    """Simulation Class for open-loop ContinuousDynamicalSystem

    Parameters
    -----------
    cds    : Instance of ContinuousDynamicSystem
    tf     : float : final time for simulation
    n      : int   : number of time steps
    solver : {'ode', 'euler'}
    """
    
    ############################
    def __init__(
        self, ClosedLoopDynamicSystem, x0_end = 0, x1_end = 0, tf=100, n=10001, vi_controller_flag = True):

        self.cds    = ClosedLoopDynamicSystem
        self.t0     = 0
        self.tf     = tf
        self.n      = int(n)
        self.dt     = ( tf + 0.0 - self.t0 ) / ( n - 1 )
        self.x0     = self.cds.x0
        self.cf     = self.cds.cost_function 
        self.x      = np.nan
        self.x0_end = x0_end
        self.x1_end = x1_end
        self.traj = 0
        self.compute_2()
        # Check Initial condition state-vector
        if self.x0.size != self.cds.n:
            raise ValueError(
                "Number of elements in x0 must be equal to number of states"
            )
            
        self.traj_J = np.zeros(np.shape(self.traj.t)[0])
        self.traj_dJ_t = np.zeros(np.shape(self.traj.t)[0])

        self.vi_controller_flag = vi_controller_flag
        if self.vi_controller_flag is True:
            self.traj_dJ_confort = np.zeros(np.shape(self.traj.t)[0])
            self.traj_dJ_security = np.zeros(np.shape(self.traj.t)[0])
            self.traj_dJ_override = np.zeros(np.shape(self.traj.t)[0])
            self.compute_costs()   
        else:
            self.traj_dJ_collision = np.zeros(np.shape(self.traj.t)[0])
            self.traj_dJ_risk = np.zeros(np.shape(self.traj.t)[0])
            self.traj_dJ_acceleration = np.zeros(np.shape(self.traj.t)[0])
            self.traj_dJ_velocity = np.zeros(np.shape(self.traj.t)[0])
            self.compute_costs_human()
        #self.plot_trajectories(self.cds.plant.road[-1])

    ##############################
    def compute_2(self, array = False):
        """ Integrate trought time """

        t  = np.linspace( self.t0 , self.tf , self.n )

        x_sol  = np.zeros((self.n,self.cds.n))
        dx_sol = np.zeros((self.n,self.cds.n))
        u_sol  = np.zeros((self.n,self.cds.m))
        y_sol  = np.zeros((self.n,self.cds.p))
        j      = np.zeros(self.n)
        
        i = 0
        self.x = x_sol[0,:] = self.x0
        dt = ( self.tf + 0.0 - self.t0 ) / ( self.n - 1 )
        while self.x[0] <= self.x0_end and self.x[1] >= self.x1_end and i<self.n:
            ti = t[i]
            self.x = x_sol[i,:]
            ui = self.cds.controller.c(self.x, 0)
            
            j[i] = self.cds.cost_function.g(self.x, ui, 0)
            
            if i+1<self.n:
                dx_sol[i]    = self.cds.f( self.x , ui , ti )
                x_sol[i+1,:] = dx_sol[i] * dt + self.x
            
            y_sol[i,:] = self.cds.h( self.x , ui , ti )
            u_sol[i,:] = ui
            i = i + 1

        traj = simulation.Trajectory(
          x = x_sol[:i,:],
          u = u_sol[:i,:],
          t = t[:i],
          dx= dx_sol[:i,:],
          y = y_sol[:i,:]
        )
        #########################
        traj.tf = i*dt
        traj.steps = i
        self.traj = traj
        
        # Compute Cost function
        if self.cf is not None :
            traj = self.cf.trajectory_evaluation( traj )
        
        return traj

    ##############################
    def compute(self, array = False):
        """ Integrate trought time """

        t  = np.linspace( self.t0 , self.tf , self.n )

        x_sol  = np.zeros((self.n,self.cds.n))
        dx_sol = np.zeros((self.n,self.cds.n))
        u_sol  = np.zeros((self.n,self.cds.m))
        y_sol  = np.zeros((self.n,self.cds.p))
        j      = np.zeros(self.n)
        
        i = 0
        self.x = x_sol[0,:] = self.x0
        dt = ( self.tf + 0.0 - self.t0 ) / ( self.n - 1 )
        while self.x[0] <= self.x0_end and self.x[1] >= self.x1_end and i<self.n:
            ti = t[i]
            self.x = x_sol[i,:]
            ui = self.cds.controller.c(self.x, 0)
            
            j[i] = self.cds.cost_function.g(self.x, ui, 0)
            
            if i+1<self.n:
                dx_sol[i]    = self.cds.f( self.x , ui , ti )
                x_sol[i+1,:] = dx_sol[i] * dt + self.x
            
            y_sol[i,:] = self.cds.h( self.x , ui , ti )
            
            u_sol[i,:] = ui
            i = i + 1

        traj = simulation.Trajectory(
          x = x_sol[:i,:],
          u = u_sol[:i,:],
          t = t[:i],
          dx= dx_sol[:i,:],
          y = y_sol[:i,:]
        )
        #########################
        traj.tf = i*dt
        traj.steps = i
        self.traj = traj
        
        # Compute Cost function
        if self.cf is not None :
            traj = self.cf.trajectory_evaluation( traj )
            
        return traj 
    
    def compute_costs(self):  
        for i in range(np.shape(self.traj.t)[0]):
            self.traj_dJ_confort[i] = self.cf.g_confort(np.array([self.traj.x[i,0],self.traj.x[i,1]]), np.array([self.traj.u[i,0],self.traj.u[i,1]]),0)
            self.traj_dJ_override[i] = self.cf.g_override(np.array([self.traj.x[i,0],self.traj.x[i,1]]), np.array([self.traj.u[i,0],self.traj.u[i,1]]),0)
            self.traj_dJ_security[i] = self.cf.g_security(np.array([self.traj.x[i,0],self.traj.x[i,1]]), np.array([self.traj.u[i,0],self.traj.u[i,1]]),0)
            self.traj_dJ_t[i] =  self.traj_dJ_security[i]*self.dt + self.traj_dJ_override[i]*self.dt + self.traj_dJ_confort[i]*self.dt 
        self.traj_J[0:i] = np.cumsum(self.traj_dJ_t[0:i])   

    def compute_costs_human(self):  
        for i in range(np.shape(self.traj.t)[0]):
            self.traj_dJ_collision[i] = self.cf.g_collision(np.array([self.traj.x[i,0],self.traj.x[i,1]]), np.array([self.traj.u[i,0],self.traj.u[i,1]]),0)
            self.traj_dJ_risk[i] = self.cf.g_risk(np.array([self.traj.x[i,0],self.traj.x[i,1]]), np.array([self.traj.u[i,0],self.traj.u[i,1]]),0)
            self.traj_dJ_acceleration[i] = self.cf.g_acceleration(np.array([self.traj.x[i,0],self.traj.x[i,1]]), np.array([self.traj.u[i,0],self.traj.u[i,1]]),0)
            self.traj_dJ_velocity[i] = self.cf.g_velocity(np.array([self.traj.x[i,0],self.traj.x[i,1]]), np.array([self.traj.u[i,0],self.traj.u[i,1]]),0)
            self.traj_dJ_t[i] =  self.traj_dJ_collision[i]*self.dt + self.traj_dJ_risk[i]*self.dt + self.traj_dJ_acceleration[i]*self.dt + self.traj_dJ_velocity[i]*self.dt 
        self.traj_J[0:i] = np.cumsum(self.traj_dJ_t[0:i])  
        
    def plot_trajectories(self, name):
        fig, axs = plt.subplots(9, 1)
        axs[0].set_title('Value Iteration: ' + name)
        axs[0].plot(self.traj.t, self.traj.x[:,0], label='position')  
        axs[1].plot(self.traj.t, self.traj.x[:,1], label='vitesse')  
        axs[2].plot(self.traj.t, self.traj.dx[:,1], label='acceleration')            
        axs[3].plot(self.traj.t, self.traj.u[:,1], label='override')    
        axs[4].plot(self.traj.t, self.traj.u[:,0], label='slip')  
        axs[5].plot(self.traj.t, self.traj_J, label='J')  
        axs[6].plot(self.traj.t, self.traj_dJ_confort, label='confort')
        axs[7].plot(self.traj.t, self.traj_dJ_security, label='security')  
        axs[8].plot(self.traj.t, self.traj_dJ_override, label='override') 
        for p in range(len(axs)):
            axs[p].legend()
            
    def plot_trajectories_new_figure(self, name, axes, print_label = False):
        axes[0].set_title('Controller: ' + name)
        axes[0].plot(self.traj.t, self.traj.x[:,0], label='position')  
        axes[1].plot(self.traj.t, self.traj.x[:,1], label='vitesse')  
        axes[2].plot(self.traj.t, self.traj.dx[:,1], label='acceleration')            
        axes[3].plot(self.traj.t, self.traj.u[:,1], label='override')    
        axes[4].plot(self.traj.t, self.traj.u[:,0], label='slip')  
        axes[5].plot(self.traj.t, self.traj_J, label='J')  
        axes[6].plot(self.traj.t, self.traj_dJ_confort, label='confort')
        axes[7].plot(self.traj.t, self.traj_dJ_security, label='security')  
        axes[8].plot(self.traj.t, self.traj_dJ_override, label='override')
        if print_label == True:
            for p in range(len(axes)):
                axes[p].legend() 
                
    def plot_trajectories_human(self, name, axes, print_label = False):
        axes[0].set_title('Controller: ' + name)
        axes[0].plot(self.traj.t, self.traj.x[:,0], label='position')  
        axes[1].plot(self.traj.t, self.traj.x[:,1], label='vitesse')  
        axes[2].plot(self.traj.t, self.traj.dx[:,1], label='acceleration')    
        axes[3].plot(self.traj.t, self.traj.u[:,0], label='slip')  
        axes[4].plot(self.traj.t, self.traj_J, label='J')  
        axes[5].plot(self.traj.t, self.traj_dJ_collision, label='Collision')
        axes[6].plot(self.traj.t, self.traj_dJ_acceleration, label='Acceleration')  
        axes[7].plot(self.traj.t, self.traj_dJ_risk, label='Risk')  
        axes[8].plot(self.traj.t, self.traj_dJ_velocity, label='Velocity')
        if print_label == True:
            for p in range(len(axes)):
                axes[p].legend() 
    
    def traj_to_args(self, traj):
        args = np.zeros([len(traj.u),4])
        for t in range(len(args)):
          if traj.u[t,1] > 0:
              args[t,3] = traj.x[t,1]
              args[t,2] = traj.x[t,0]
              args[t,1] = np.nan
              args[t,0] = np.nan
          else:
              args[t,1] = traj.x[t,1]
              args[t,0] = traj.x[t,0]
              args[t,2] = np.nan
              args[t,3] = np.nan  
        return args

class SimulatorSeminaire:
    ############################
    def __init__(self, ClosedLoopDynamicSystem, costfunction, x0 = [-80.0,20.0], x_end = [0.0001,0.0001]):

        self.cds    = ClosedLoopDynamicSystem
        self.t0     = 0
        self.tf     = 1000
        self.n      = 10000
        self.dt     = 0.01
        self.x0     = x0
        self.cf     = costfunction
        self.x0_end = x_end[0]
        self.x1_end = x_end[1]
        self.traj = 0
        self.compute()


    ##############################
    def compute(self, array = False):
        """ Integrate trought time """

        self.t = np.arange(self.t0,self.tf,self.dt)
        self.n = np.size(self.t)
        
        self.x_sol  = np.zeros((self.n,self.cds.n))
        self.dx_sol = np.zeros((self.n,self.cds.n))
        self.u_sol  = np.zeros((self.n,self.cds.m))
        self.y_sol  = np.zeros((self.n,self.cds.p))
        self.g      = np.zeros(self.n)
        self.g_security      = np.zeros(self.n)
        self.g_confort      = np.zeros(self.n)
        self.g_override      = np.zeros(self.n)

        self.dJ      = np.zeros(self.n)
        self.dJ_security      = np.zeros(self.n)
        self.dJ_confort      = np.zeros(self.n)
        self.dJ_override      = np.zeros(self.n)
        
        self.J      = np.zeros(self.n)
        self.J_security      = np.zeros(self.n)
        self.J_confort      = np.zeros(self.n)
        self.J_override      = np.zeros(self.n)
        
        i = 0
        
        #État initial
        self.x = self.x_sol[0,:] = self.x0
        dt = self.dt
        
        while self.x[0] <= self.x0_end and self.x[1] >= self.x1_end and i<self.n:
            ti = self.t[i]
            self.x = self.x_sol[i,:]
            ui = self.cds.controller.c(self.x, 0)
            
            self.g[i] = self.cf.g(self.x, ui, 0)
            self.g_security[i] = self.cf.g_security(self.x, ui, 0)
            self.g_confort[i] = self.cf.g_confort(self.x, ui, 0)
            self.g_override[i] = self.cf.g_override(self.x, ui, 0)
            
            self.dJ[i] = self.g[i]*self.dt
            self.dJ_security[i] = self.g_security[i]*self.dt
            self.dJ_confort[i] = self.g_confort[i]*self.dt
            self.dJ_override[i] = self.g_override[i]*self.dt
            
            self.J[i] = self.dJ[i] + self.J[i-1]
            self.J_security[i] = self.dJ_security[i] + self.J_security[i-1]
            self.J_confort[i] = self.dJ_confort[i] + self.J_confort[i-1]
            self.J_override[i] = self.dJ_override[i] + self.J_override[i-1]
            
            if i+1<self.n:
                self.dx_sol[i]    = self.cds.f( self.x , ui , 0)
                self.x_sol[i+1,:] = self.dx_sol[i] * dt + self.x
            
            self.y_sol[i,:] = self.cds.h( self.x , ui , ti )
            self.u_sol[i,:] = ui
            i = i + 1
        self.x_sol = self.x_sol[:i,:]
        self.u_sol = self.u_sol[:i,:]
        self.t_sol = self.t[:i]
        self.dx_sol= self.dx_sol[:i,:]
        self.y_sol = self.y_sol[:i,:]
        
        self.g      = self.g[:i]
        self.g_security      = self.g_security[:i]
        self.g_confort      = self.g_confort[:i]
        self.g_override      = self.g_override[:i]

        self.dJ      = self.dJ[:i]
        self.dJ_security      = self.dJ_security[:i]
        self.dJ_confort      = self.dJ_confort[:i]
        self.dJ_override      = self.dJ_override[:i]
        
        self.J      = self.J[:i]
        self.J_security      = self.J_security[:i]
        self.J_confort      = self.J_confort[:i]
        self.J_override      = self.J_override[:i]
        
    def plot_data(self):
        fig, axs = plt.subplots(1,3, figsize=(8,8))
        plt.ion()
        axs[0].plot(self.x_sol[:,0],self.x_sol[:,1])
        axs[0].grid()
        axs[0].set_ylabel('Vitesse')
        axs[0].set_xlabel('Position')
        
        axs[1].plot(self.t_sol,self.dJ)
        axs[1].plot(self.t_sol,self.dJ_security)
        axs[1].plot(self.t_sol,self.dJ_confort)
        axs[1].plot(self.t_sol,self.dJ_override)
        axs[1].grid()
        axs[1].set_ylabel('Vitesse')
        axs[1].set_xlabel('Position')
        
        axs[2].plot(self.t_sol,self.J)
        axs[2].plot(self.t_sol,self.J_security)
        axs[2].plot(self.t_sol,self.J_confort)
        axs[2].plot(self.t_sol,self.J_override)
        axs[2].grid()
        axs[2].set_ylabel('Vitesse')
        axs[2].set_xlabel('Position')
        
        plt.tight_layout()
        
class SimulatorV2_vi_vs_ttc:
    """Simulation Class for open-loop ContinuousDynamicalSystem

    Parameters
    -----------
    cds    : Instance of ContinuousDynamicSystem
    tf     : float : final time for simulation
    n      : int   : number of time steps
    solver : {'ode', 'euler'}
    """
    
    ############################
    def __init__(
        self, ClosedLoopDynamicSystem_vi, ClosedLoopDynamicSystem_ttc, x0_end = 100, x1_end = 0.05, tf=100, n=10001):

        self.cds_vi    = ClosedLoopDynamicSystem_vi
        self.cds_ttc    = ClosedLoopDynamicSystem_ttc
        self.t0     = 0
        self.tf     = tf
        self.n      = int(n)
        self.dt     = ( tf + 0.0 - self.t0 ) / ( n - 1 )
        self.x0     = self.cds_vi.x0
        self.cf = ClosedLoopDynamicSystem_vi.cost_function
        self.x      = np.nan
        self.x0_end = x0_end
        self.x1_end = x1_end
        self.traj = 0
        # Check Initial condition state-vector
        if self.x0.size != self.cds_vi.n:
            raise ValueError(
                "Number of elements in x0 must be equal to number of states"
            )
            
        self.traj_vi = self.compute_vi()
        self.traj_ttc = self.compute_ttc()
        
        self.traj_vi_J = np.zeros(np.shape(self.traj_vi.t)[0])
        self.traj_vi_dJ_t = np.zeros(np.shape(self.traj_vi.t)[0])
        self.traj_vi_dJ_confort = np.zeros(np.shape(self.traj_vi.t)[0])
        self.traj_vi_dJ_security = np.zeros(np.shape(self.traj_vi.t)[0])
        self.traj_vi_dJ_override = np.zeros(np.shape(self.traj_vi.t)[0])
        
        self.traj_ttc_J = np.zeros(np.shape(self.traj_ttc.t)[0])
        self.traj_ttc_dJ_t = np.zeros(np.shape(self.traj_ttc.t)[0])
        self.traj_ttc_dJ_confort = np.zeros(np.shape(self.traj_ttc.t)[0])
        self.traj_ttc_dJ_security = np.zeros(np.shape(self.traj_ttc.t)[0])
        self.traj_ttc_dJ_override = np.zeros(np.shape(self.traj_ttc.t)[0])
        
        self.compute_costs()

    ##############################
    def compute_vi(self, array = False):
        """ Integrate trought time """

        t  = np.linspace( self.t0 , self.tf , self.n )

        x_sol  = np.zeros((self.n,self.cds_vi.n))
        dx_sol = np.zeros((self.n,self.cds_vi.n))
        u_sol  = np.zeros((self.n,self.cds_vi.m))
        y_sol  = np.zeros((self.n,self.cds_vi.p))
        j      = np.zeros(self.n)
        
        i = 0
        self.x = x_sol[0,:] = self.x0
        dt = ( self.tf + 0.0 - self.t0 ) / ( self.n - 1 )
        while self.x[0] <= self.x0_end and self.x[1] >= self.x1_end and i<self.n:
            ti = t[i]
            self.x = x_sol[i,:]
            ui = self.cds_vi.controller.c(self.x, 0)
            
            j[i] = self.cds_vi.cost_function.g(self.x, ui, 0, 0)
            
            if i+1<self.n:
                dx_sol[i]    = self.cds_vi.f( self.x , ui , ti )
                x_sol[i+1,:] = dx_sol[i] * dt + self.x
            
            y_sol[i,:] = self.cds_vi.h( self.x , ui , ti )
            u_sol[i,:] = ui
            i = i + 1

        traj = simulation.Trajectory(
          x = x_sol[:i,:],
          u = u_sol[:i,:],
          t = t[:i],
          dx= dx_sol[:i,:],
          y = y_sol[:i,:]
        )
        #########################
        traj.tf = i*dt
        traj.steps = i
        self.traj = traj
        
        # Compute Cost function
        if self.cf is not None :
            traj = self.cf.trajectory_evaluation( traj )
        
        if array == False:  
          return traj 
        else:
          return np.sum(j)

    ##############################
    def compute_ttc(self, array = False):
        """ Integrate trought time """
    
        t  = np.linspace( self.t0 , self.tf , self.n )
    
        x_sol  = np.zeros((self.n,self.cds_ttc.n))
        dx_sol = np.zeros((self.n,self.cds_ttc.n))
        u_sol  = np.zeros((self.n,self.cds_ttc.m))
        y_sol  = np.zeros((self.n,self.cds_ttc.p))
        j      = np.zeros(self.n)
        
        i = 0
        self.x = x_sol[0,:] = self.x0
        dt = ( self.tf + 0.0 - self.t0 ) / ( self.n - 1 )
        while self.x[0] <= self.x0_end and self.x[1] >= self.x1_end and i<self.n:
            ti = t[i]
            self.x = x_sol[i,:]
            ui = self.cds_ttc.controller.c(self.x, 0)
            
            j[i] = self.cds_ttc.cost_function.g(self.x, ui, 0, 0)
            
            if i+1<self.n:
                dx_sol[i]    = self.cds_ttc.f( self.x , ui , ti )
                x_sol[i+1,:] = dx_sol[i] * dt + self.x
            
            y_sol[i,:] = self.cds_ttc.h( self.x , ui , ti )
            u_sol[i,:] = ui
            i = i + 1
    
        traj = simulation.Trajectory(
          x = x_sol[:i,:],
          u = u_sol[:i,:],
          t = t[:i],
          dx= dx_sol[:i,:],
          y = y_sol[:i,:]
        )
        #########################
        traj.tf = i*dt
        traj.steps = i
        self.traj = traj
        
        # Compute Cost function
        if self.cf is not None :
            traj = self.cf.trajectory_evaluation( traj )
        if array == False:  
          return traj 
        else:
          return np.sum(j)
    
    def compute_costs(self):
        #VI
        for i in range(np.shape(self.traj_vi.t)[0]):
            self.traj_vi_dJ_confort[i] = self.cf.g_confort(np.array([self.traj_vi.x[i,0],self.traj_vi.x[i,1]]), np.array([self.traj_vi.u[i,0],self.traj_vi.u[i,1]]),0,0)
            self.traj_vi_dJ_override[i] = self.cf.g_override(np.array([self.traj_vi.x[i,0],self.traj_vi.x[i,1]]), np.array([self.traj_vi.u[i,0],self.traj_vi.u[i,1]]),0,0)
            self.traj_vi_dJ_security[i] = self.cf.g_security(np.array([self.traj_vi.x[i,0],self.traj_vi.x[i,1]]), np.array([self.traj_vi.u[i,0],self.traj_vi.u[i,1]]),0,0)
            self.traj_vi_dJ_t[i] =  self.traj_vi_dJ_security[i]*self.dt + self.traj_vi_dJ_override[i]*self.dt + self.traj_vi_dJ_confort[i]*self.dt 
        self.traj_vi_J[0:i] = np.cumsum(self.traj_vi_dJ_t[0:i])   
        
        for i in range(np.shape(self.traj_ttc.t)[0]):
            self.traj_ttc_dJ_confort[i] = self.cf.g_confort(np.array([self.traj_ttc.x[i,0],self.traj_ttc.x[i,1]]), np.array([self.traj_ttc.u[i,0],self.traj_ttc.u[i,1]]),0,0)
            self.traj_ttc_dJ_override[i] = self.cf.g_override(np.array([self.traj_ttc.x[i,0],self.traj_ttc.x[i,1]]), np.array([self.traj_ttc.u[i,0],self.traj_ttc.u[i,1]]),0,0)
            self.traj_ttc_dJ_security[i] = self.cf.g_security(np.array([self.traj_ttc.x[i,0],self.traj_ttc.x[i,1]]), np.array([self.traj_ttc.u[i,0],self.traj_ttc.u[i,1]]),0,0)
            self.traj_ttc_dJ_t[i] =  self.traj_ttc_dJ_security[i]*self.dt + self.traj_ttc_dJ_override[i]*self.dt + self.traj_ttc_dJ_confort[i]*self.dt 
        self.traj_ttc_J[0:i] = np.cumsum(self.traj_ttc_dJ_t[0:i])   
        
    def plot_trajectories(self, name):
        fig, axs = plt.subplots(9, 2)
        axs[0][0].set_title('Value Iteration: ' + name)
        axs[0][0].plot(self.traj_vi.t, self.traj_vi.x[:,0], label='position')  
        axs[1][0].plot(self.traj_vi.t, self.traj_vi.x[:,1], label='vitesse')  
        axs[2][0].plot(self.traj_vi.t, self.traj_vi.dx[:,1], label='acceleration')            
        axs[3][0].plot(self.traj_vi.t, self.traj_vi.u[:,1], label='override')    
        axs[4][0].plot(self.traj_vi.t, self.traj_vi.u[:,0], label='slip')  
        axs[5][0].plot(self.traj_vi.t, self.traj_vi_J, label='J')  
        axs[6][0].plot(self.traj_vi.t, self.traj_vi_dJ_confort, label='confort')
        axs[7][0].plot(self.traj_vi.t, self.traj_vi_dJ_security, label='security')  
        axs[8][0].plot(self.traj_vi.t, self.traj_vi_dJ_override, label='override')  

        
        axs[0][1].set_title('TTC: ' + name)
        axs[0][1].plot(self.traj_ttc.t, self.traj_ttc.x[:,0], label='position')  
        axs[1][1].plot(self.traj_ttc.t, self.traj_ttc.x[:,1], label='vitesse')  
        axs[2][1].plot(self.traj_ttc.t, self.traj_ttc.dx[:,1], label='acceleration')     
        axs[3][1].plot(self.traj_ttc.t, self.traj_ttc.u[:,1], label='override')    
        axs[4][1].plot(self.traj_ttc.t, self.traj_ttc.u[:,0], label='slip')
        axs[5][1].plot(self.traj_ttc.t, self.traj_ttc_J, label='J')
        axs[6][1].plot(self.traj_ttc.t, self.traj_ttc_dJ_confort, label='confort') 
        axs[7][1].plot(self.traj_ttc.t, self.traj_ttc_dJ_security, label='security') 
        axs[8][1].plot(self.traj_ttc.t, self.traj_ttc_dJ_override, label='override') 

        i=0
        for p in axs:
            axs[i][0].legend()
            axs[i][1].legend()
            i=i+1

        