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
        
        costfunction.CostFunction.__init__(self)

        self.n = n
        self.m = m
        self.p = p

        self.INF = 1E3
        
        self.xbar = np.zeros(self.n)
        self.ubar = np.zeros(self.m)
        self.ybar = np.zeros(self.p)
        
        self.driver = None
        self.road = None
        
        self.state_lab = None
        self.state_uni = None

        # Quadratic cost weights
        self.Q = np.diag( np.ones(n)  )
        self.R = np.diag( np.ones(m)  )
        self.V = np.diag( np.zeros(p) )
        
        self.x_ub = 0
        self.x_lb = 0
        self.u_ub = 0
        self.u_lb = 0
        
        self.override_coef = 10
        self.confort_coef = 0.01
        self.security_coef = 10000
        
        self.ry = None 
        self.rr = None 
        self.rf = None
        self.m    = None 
        self.gravity    = None
        self.rcda = None
        
        # Optionnal zone of zero cost if ||dx|| < EPS 
        self.ontarget_check = False
        self.sys_test = None
    
    ############################
    @classmethod
    def from_sys(cls, sys):
        """ From ContinuousDynamicSystem instance """
        
        instance = cls( sys.n , sys.m , sys.p)
        
        instance.xbar = sys.xbar
        instance.ubar = sys.ubar
        instance.ybar = np.zeros( sys.p )
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
        
        return instance
    

    #############################
    def h(self, x , t = 0):
        h = 0
        if x[0] > 19.95:
            h = 10000
        return h

    def g(self, x, u, t):
        dJ = self.g_confort(x, u, t) + self.g_override(x, u, t) + self.g_security(x, u, t)   
        if (u[1] == 0 and u[0] != 0) or (u[1] == 1 and u[0] == 0):
            dJ = 2000000000
        return dJ   
    
    def g_security(self, x, u, t):
        return ((((float(self.security_coef))*x[1])**2) + (float(self.security_coef)))  / (1+np.exp(-10.0*(x[0]-(0.99*(self.x_ub[0])))))

    def g_confort(self, x, u, t):
        slip = u[0]
        v    = x[1] 
        p = x[0]
        
        if slip > 0:
            mu = ((self.road[0]*(1 - np.exp(-1*self.road[1]*slip)))-(self.road[2]*slip))*np.exp(-1*self.road[3]*slip*v)
        else:
            slip = np.abs(slip)
            mu = -1*((self.road[0]*(1 - np.exp(-1*self.road[1]*slip)))-(self.road[2]*slip))*np.exp(-1*self.road[3]*slip*v)
        
        # Drag froce
        fd = 0.5 * self.rcda * v * np.abs( v ) # drag froce with the right sign
        a  = (mu * self.m * self.gravity * self.rr - fd)/( self.m * (1 + mu * self.ry ))
        
        #return (float(self.confort_coef)*(a*a) * (((self.x_ub[0]-5)-p)*((self.x_ub[0]-5)-p)))
        return (float(self.confort_coef)*(a*a) * (((self.x_ub[0])-p)*((self.x_ub[0])-p)))
#        return float(self.confort_coef)*(a*a)
    
    def g_override(self, x, u, t):
        return (u[1]*float(self.override_coef))
   
    def print_security(self):
         pos = np.arange(self.x_lb[0], self.x_ub[0]+0.01, self.x_ub[0]/200)
         vit = np.arange(self.x_lb[1], self.x_ub[1]+0.01, self.x_ub[1]/200)
         j = np.zeros([np.shape(pos)[0], np.shape(vit)[0]])
         for v in range(np.shape(vit)[0]):
              for p in range(np.shape(pos)[0]):
                   j[p][v] = self.g_security(np.array([pos[p], vit[v]]),0,0)
                   
                   
         fig, axs = plt.subplots(1)
         plt.ion()
         xname = self.state_lab[0] + ' ' + self.state_uni[0]
         yname = self.state_lab[1] + ' ' + self.state_uni[1]  
         fig.suptitle('Security mapin CF')
         axs.set(xlabel=xname, ylabel=yname)
         i1 = axs.pcolormesh(pos, vit, j.T, shading='gouraud')
         axs.axis([self.x_lb[0], self.x_ub[0], self.x_lb[1], self.x_ub[1]])
         fig.colorbar(i1, ax=axs)
         axs.grid(True)  


# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:02:23 2022

@author: Charles-Alexis
"""
from pyro.analysis import costfunction


class DriverModelCostFunction_ROS(costfunction.CostFunction):
    ############################
    def __init__(self, n, m, p):

        costfunction.CostFunction.__init__(self)

        self.n = n
        self.m = m
        self.p = p

        self.INF = 1E3

        self.xbar = np.zeros(self.n)
        self.ubar = np.zeros(self.m)
        self.ybar = np.zeros(self.p)

        self.driver = None
        self.road = None

        # Quadratic cost weights
        self.Q = np.diag(np.ones(n))
        self.R = np.diag(np.ones(m))
        self.V = np.diag(np.zeros(p))

        self.ry = None
        self.rr = None
        self.rf = None
        self.m = None
        self.gravity = None
        self.rcda = None

        # Optionnal zone of zero cost if ||dx|| < EPS
        self.ontarget_check = False
        self.sys_test = None

    ############################
    @classmethod
    def from_sys(cls, sys):
        """ From ContinuousDynamicSystem instance """

        instance = cls(sys.n, sys.m, sys.p)

        instance.xbar = sys.xbar
        instance.ubar = sys.ubar
        instance.ybar = np.zeros(sys.p)
        instance.driver = sys.driver
        instance.road = sys.road

        instance.ry, instance.rr, instance.rf = sys.compute_ratios()
        instance.m = sys.mass
        instance.gravity = sys.gravity
        instance.rcda = sys.rho * sys.cdA

        return instance

    #############################
    def h(self, x, t=0):
        h = 0
        return h

    def g(self, x, u, dx, t):
        dJ = self.g_confort(x, u, dx, t) + self.g_override(x, u, dx, t) + self.g_security(x, u, dx, t)
        if (u[1] == 0 and u[0] != 0) or (u[1] == 1 and u[0] == 0):
            dJ = 200000
        return dJ

    def g_security(self, x, u, dx, t):
        p = x[0]
        v = x[1]
        security = 0
        if p > 6:
            security = 1
        if p > 8:
            security = 5
        if p >= 9:
            security = 10 * ((v) ** 2)
        if p >= 10:
            security = 10000 * ((v) ** 2)  # + (100 * ((v)**2) * p-99)

        security = (100 * v ** 2) / (1 + np.exp(-0.5 * (p - 95)))

        return security

    def g_confort(self, x, u, dx, t):
        slip = u[0]
        v = x[1]
        p = x[0]
        a = dx[1]
        return (0.01 * (a ** 2)) * ((100 - p) ** 2)

    def g_override(self, x, u, y, t):
        return (u[1] * 10)
