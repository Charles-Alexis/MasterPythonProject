# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:02:23 2022

@author: Charles-Alexis
"""
import numpy as np
import matplotlib.pyplot as plt
from pyro.analysis import costfunction

class DriverModelCostFunction(costfunction.CostFunction):
    ############################
    def __init__(self, n, m, p):
        
        super().__init__()

        self.n = n
        self.m = m
        self.p = p

        self.INF = 999999999999999999
        
        self.driver = None
        self.road = None
        
        self.state_lab = None
        self.state_uni = None
        
        self.x_ub = 0
        self.x_lb = 0
        self.u_ub = 0
        self.u_lb = 0
        
        self.override_coef = 10
        self.override_diff_coef = 100
        self.confort_coef = 0.01
        self.security_coef = 10000
        self.secrity_slope = 30
        self.security_distance = 0.3
        
        self.ry = None 
        self.rr = None 
        self.rf = None
        self.m    = None 
        self.gravity    = None
        self.rcda = None

        self.sys_human_array = None
        self.dmax = None
        self.slip2force = None
        self.f = None
        self.u_levels = None
        self.x_grid = None
        self.f_levels = None
        self.human_levels = None
        
        # Optionnal zone of zero cost if ||dx|| < EPS 
        self.ontarget_check = False
        self.sys_test = None
        self.dmin = None
        
        # BUGGED STATES
        self.pos_array = None
        self.vit_array = None
        self.bugged_states = None
    
    ############################
    @classmethod
    def from_sys(cls, sys):
        """ From ContinuousDynamicSystem instance """
        
        instance = cls( sys.n , sys.m , sys.p)

        instance.driver = sys.driver
        instance.road = sys.road
        
        instance.x_ub = sys.x_ub
        instance.x_lb = sys.x_lb
        instance.u_ub = sys.u_ub
        instance.u_lb = sys.u_lb
        
        instance.ry, instance.rr, instance.rf = sys.compute_ratios()
        instance.m = sys.mass
        instance.gravity = sys.gravity
        instance.rcda = sys.rho * sys.cdA
        
        instance.state_lab = sys.state_label
        instance.state_uni = sys.state_units
        instance.dmax = sys.dmax
        instance.sys_human_array = sys.human_model
        instance.slip2force = sys.slip2force
        instance.f = sys.f
        
        instance.x_grid = sys.x_grid
        instance.u_levels = sys.u_level
        
        instance.pos_array = sys.pos_array
        instance.vit_array = sys.vit_array
        
        return instance
    

    #############################
    def h(self, x , t = 0):
        return 0

    def g(self, x, u, t, e = 0):
        dJ = self.g_confort(x, u, t, e) + self.g_override(x, u, t, e) + self.g_security(x, u, t, e) 
        if u[0] < -0.3:
            dJ = 9999999999999999999
        return dJ   
    
    def g_security(self, x, u, t, e = 0):
        J = (((x[1])**2)) / (1+np.exp(-self.security_slope*(x[0]+self.security_distance))) 
        if J < 0.0001:
            J = 0
        J = J * self.security_coef
        return J 
        
    def g_confort(self, x, u, t, e = 0):
        pos = x[0]
        vit = x[1]
        slip = u[0]
        override = u[1]
        
        human_slip = self.sys_human_array(pos, vit, e)
        dx = self.f([pos,vit],[slip,override], e) 
        dec = np.clip((dx[1])**2, 0, (self.dmax[1]**2))
        
        if override == 0 and human_slip == 0:
            J = 0
        else:
            J = float(self.confort_coef) * (dec) #/ (np.abs(self.dmax[1]**2))
            
        # if override == 0 and human_slip == -0.3:
        #     J = 999999999

        if vit == 0:
            J = 0
        return J
    
    def g_override(self, x, u, t, e = 0):
        slip = u[0]
        override = u[1]
        pos = x[0]
        vit = x[1]
        
        scaling_factor = (1/(0.3**1)) * np.abs(self.dmax[1])
        human_slip = self.sys_human_array(pos, vit, e)
        u_diff = np.abs(human_slip-slip)
        J = self.override_coef * u_diff * override * scaling_factor 

        if vit == 0:
            J = 0
        return J
        
   
    def print_security(self, x , u, cfg, name):
         x_0 = x[0][100:150]
         j = np.zeros([len(x_0), len(x[1])])
         for v in range(len(x[1])):
              for p in range(len(x_0)):
                   x1 = x_0[p]
                   x2 = x[1][v]
                   j[p][v] = cfg(np.array([x1, x2]),u,0)  
         POS, VIT = np.meshgrid(x_0,x[1])
         fig = plt.figure()
         ax = plt.axes(projection='3d')
         ax.plot_surface(POS, VIT, j.T, rstride=1, cstride=1,cmap='plasma', edgecolor='none')
         ax.set_title('Coût pour le paramètre de Sécurité')
         ax.set_xlabel('Distance')
         ax.set_ylabel('Vitesse')
         ax.set_zlabel('Coût');
         return j
    
    def find_nearest_arg(self, value, array):
         array = np.asarray(array)
         idx = (np.abs(array - value)).argmin()
         return idx
    
    def compute_f(self):
        self.f_levels = np.zeros([len(self.u_levels), self.x_grid[1]])
        self.vit_array = np.arange(0, self.x_grid[1])/((self.x_grid[1]-1)/20)
        for u in range(len(self.u_levels)):
            for v in range(len(self.vit_array)):
                self.f_levels[u][v] = self.f([-100,self.vit_array[v]] , [self.u_levels[u], 1],0)[1]
                
    def compute_human(self):
        self.human_levels = np.zeros([self.x_grid[0], self.x_grid[1]])
        self.pos_array = -1*np.arange(0, self.x_grid[0])/((self.x_grid[0]-1)/75)
        self.vit_array = np.arange(0, self.x_grid[1])/((self.x_grid[1]-1)/20)
        for p in range(len(self.pos_array)):
            for v in range(len(self.vit_array)):
                self.human_levels[p][v] = self.sys_human_array(self.pos_array[p], self.vit_array[v], 0)

    def transform_ind(self, p, v):
        p_ind = self.find_nearest_arg(p, self.pos_array)
        v_ind = self.find_nearest_arg(v, self.vit_array)
        return [p_ind,v_ind]

                
class DriverModelCostFunction_forhumanmodel(costfunction.CostFunction):
    ############################
    def __init__(self, n, m, p):
        
        super().__init__()

        self.n = n
        self.m = m
        self.p = p

        self.INF = 999999999999999999
        
        self.driver = None
        self.road = None
        
        self.state_lab = None
        self.state_uni = None
        
        self.x_ub = 0
        self.x_lb = 0
        self.u_ub = 0
        self.u_lb = 0
        
        self.override_coef = 10
        self.override_diff_coef = 100
        self.confort_coef = 0.01
        self.security_coef = 10000
        self.secrity_slope = 30
        self.security_distance = 0.3
        
        self.ry = None 
        self.rr = None 
        self.rf = None
        self.m    = None 
        self.gravity    = None
        self.rcda = None

        self.sys_human_array = None
        self.dmax = None
        self.slip2force = None
        self.f = None
        self.u_levels = None
        self.x_grid = None
        self.f_levels = None
        self.human_levels = None
        
        # Optionnal zone of zero cost if ||dx|| < EPS 
        self.ontarget_check = False
        self.sys_test = None
        self.dmin = None
        
        # BUGGED STATES
        self.pos_array = None
        self.vit_array = None
        self.bugged_states = None
    
    ############################
    @classmethod
    def from_sys(cls, sys):
        """ From ContinuousDynamicSystem instance """
        
        instance = cls( sys.n , sys.m , sys.p)

        instance.driver = sys.driver
        instance.road = sys.road
        
        instance.x_ub = sys.x_ub
        instance.x_lb = sys.x_lb
        instance.u_ub = sys.u_ub
        instance.u_lb = sys.u_lb
        
        instance.ry, instance.rr, instance.rf = sys.compute_ratios()
        instance.m = sys.mass
        instance.gravity = sys.gravity
        instance.rcda = sys.rho * sys.cdA
        
        instance.state_lab = sys.state_label
        instance.state_uni = sys.state_units
        instance.dmax = sys.dmax
        instance.sys_human_array = sys.human_model
        instance.slip2force = sys.slip2force
        instance.f = sys.f
        
        instance.x_grid = sys.x_grid
        instance.u_levels = sys.u_level
        
        instance.pos_array = sys.pos_array
        instance.vit_array = sys.vit_array
        
        return instance
    

    #############################
    def h(self, x , t = 0):
        return 0

    def g(self, x, u, t, e = 0):
        dJ = self.g_confort(x, u, t, e) + self.g_override(x, u, t, e) + self.g_security(x, u, t, e) 
        return dJ   
    
    def g_security(self, x, u, t, e = 0):
        J = (((x[1])**2)) / (1+np.exp(-self.security_slope*(x[0]+self.security_distance)))     
        if J < 0.01:
            J = 0
        J = J * self.security_coef        
        return J 
        
    def g_confort(self, x, u, t, e = 0):
        pos = x[0]
        vit = x[1]
        if self.sys_human_array(pos, vit, e) == 0:
            J = 0
        else:
            dx = self.f([pos,vit],[0,0], e) 
            dec = np.clip((dx[1])**2, 0, (self.dmax[1]**2))
            J = self.confort_coef * dec  #/ (np.abs(self.dmax[1]**2))
        if vit == 0:
            J = 0
        return J
    
    def g_override(self, x, u, t, e = 0):
        return 0
        
   
    def print_security(self, x , u, cfg, name):     
         j = np.zeros([len(x[0]), len(x[1])])
         for v in range(len(x[1])):
              for p in range(len(x[0])):
                   x1 = x[0][p]
                   x2 = x[1][v]
                   j[p][v] = cfg(np.array([x1, x2]),u,0)  
         POS, VIT = np.meshgrid(x[0],x[1])
         fig = plt.figure()
         ax = plt.axes(projection='3d')
         ax.plot_surface(POS, VIT, j.T, rstride=1, cstride=1,cmap='plasma', edgecolor='none')
         ax.set_title('Performance: Sécurité')
         ax.set_xlabel('Distance')
         ax.set_ylabel('Vitesse')
         ax.set_zlabel('Coût');
         return j
    
    def find_nearest_arg(self, value, array):
         array = np.asarray(array)
         idx = (np.abs(array - value)).argmin()
         return idx
    
    def compute_f(self):
        self.f_levels = np.zeros([len(self.u_levels), self.x_grid[1]])
        self.vit_array = np.arange(0, self.x_grid[1])/((self.x_grid[1]-1)/20)
        for u in range(len(self.u_levels)):
            for v in range(len(self.vit_array)):
                self.f_levels[u][v] = self.f([-100,self.vit_array[v]] , [self.u_levels[u], 1],0)[1]
                
    def compute_human(self):
        self.human_levels = np.zeros([self.x_grid[0], self.x_grid[1]])
        self.pos_array = -1*np.arange(0, self.x_grid[0])/((self.x_grid[0]-1)/75)
        self.vit_array = np.arange(0, self.x_grid[1])/((self.x_grid[1]-1)/20)
        for p in range(len(self.pos_array)):
            for v in range(len(self.vit_array)):
                self.human_levels[p][v] = self.sys_human_array(self.pos_array[p], self.vit_array[v], 0)

    def transform_ind(self, p, v):
        p_ind = self.find_nearest_arg(p, self.pos_array)
        v_ind = self.find_nearest_arg(v, self.vit_array)
        return [p_ind,v_ind]
    
    