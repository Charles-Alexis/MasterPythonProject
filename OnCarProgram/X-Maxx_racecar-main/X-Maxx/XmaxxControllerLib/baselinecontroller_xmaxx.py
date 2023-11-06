#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:12:02 2022

@author: clearpath-robot
"""

###############################################################################
import numpy as np
from pyro.control  import controller
###############################################################################

class TTCController(controller.StaticController):
    """ 
    Feedback controller
    ---------------------------------------
    r  : reference signal vector  k x 1
    y  : sensor signal vector     m x 1
    u  : control inputs vector    p x 1
    t  : time                     1 x 1
    ---------------------------------------
    u = c( y , r , t ) 

    """
    ############################
    def __init__(self, sys, grid_sys, human_model, ttc_ref=1.5, position_obs=20, slip_cmd=-0.2):
        """ """
        
        # Dimensions
        self.sys = sys
        self.grid_sys = grid_sys
        self.slip_cmd = slip_cmd
        self.human_model = human_model
        self.acceleration = self.sys.f
        self.ttc_min = ttc_ref
        self.p_obs = position_obs
        
        controller.StaticController.__init__(self, self.sys.n, self.sys.m, self.sys.p)
        
        # Label
        self.name = 'BaseLine Controller'

    #############################
    def c( self , y , r , t = 0 ):
        """  State feedback (y=x) - no reference - time independent """
        x = y
        u = np.array([0.,0.])
        
        u[0] = self.human_model(x[1],x[0])
        u[1] = 1
        dx = self.acceleration(x, u)
        
        px = self.p_obs - x[0]
        vx = 0 - x[1] - 0.0000001
        ax = r
        #ttc = -(vx/ax) - (np.sqrt((vx**2)-(2*px*ax))/ax)
        ttc =  x[0]/(x[1]+0.00001)
        if ttc > self.ttc_min:
            u[0] = 0
            u[1] = 0
        elif ttc < 0:
            u[0] = 0
            u[1] = 0
        else:
            ttc_p = np.abs(((ttc-self.ttc_min)/((self.ttc_min*0.75)-self.ttc_min)))
            if ttc_p <= 1:
                u[0] = self.slip_cmd * ttc_p
            else:
                u[0] = self.slip_cmd 
	    u[0] = 0.2
            u[1] = 1
        return u
    
    def c_array(self):
        u = np.zeros(self.grid_sys.x_grid_dim)
        for p in range(len(self.grid_sys.x_level[0])):     
            for v in range(len(self.grid_sys.x_level[1])):
                arr = np.array([ self.grid_sys.x_level[0][p], self.grid_sys.x_level[1][v]])
                u[p][v] = self.c(arr, 0)[0]
        return u
    
class humanController(controller.StaticController):
    """ 
    Feedback controller
    ---------------------------------------
    r  : reference signal vector  k x 1
    y  : sensor signal vector     m x 1
    u  : control inputs vector    p x 1
    t  : time                     1 x 1
    ---------------------------------------
    u = c( y , r , t ) 

    """
    ############################
    def __init__(self, sys, grid_sys, human_model):
        """ """
        
        # Dimensions
        self.sys = sys
        self.grid_sys = grid_sys
        self.human_model = human_model
        self.acceleration = self.sys.f
        
        super().__init__(self.sys.n, self.sys.m, self.sys.p)
        
        # Label
        self.name = 'Human model controller'

    #############################
    def c( self , y , r , t = 0 ):
        slip = self.human_model(y[1],y[0])
        if slip != 0:
            override = 1
        else:
            override = 0
        
        return np.array([slip,override])


class TTCController_ROS(controller.StaticController):
    """
    Feedback controller
    ---------------------------------------
    r  : reference signal vector  k x 1
    y  : sensor signal vector     m x 1
    u  : control inputs vector    p x 1
    t  : time                     1 x 1
    ---------------------------------------
    u = c( y , r , t )

    """

    ############################
    def __init__(self, ttc_treshold):
        """ """

        # Dimensions
        self.k = 2
        self.m = 2
        self.p = 2
        self.ttc_treshold = ttc_treshold
        self.ttc=100
        self.flag_possible = True

        controller.StaticController.__init__(self, self.k, self.m, self.p)

        # Label
        self.name = 'BaseLine Controller'

    #############################
    def c(self, x, dx, t=0):
        """  State feedback (y=x) - no reference - time independent """
        u = np.array([0., 0.])
        px = x[0]
        vx = x[1]
        ax = dx[1]
        self.ttc = 100



        if np.abs(ax) < 0.05 and vx < 0:
            self.ttc  = -px / vx
        elif vx < 0 and np.abs(ax) > 0.05:
            self.ttc  = (-vx / ax) - (np.sqrt((vx ** 2) - (2 * px * ax)) / ax)
        elif vx >= 0 and ax < 0:
            self.ttc  = (-vx / ax) + (np.sqrt((vx ** 2) - (2 * px * ax)) / ax)
        else:
            self.ttc  = 50

        if self.ttc  < 0:
            self.ttc  = 0
        if ((vx ** 2) - (2 * px * ax)) < 0:
            self.ttc  = np.inf
            self.flag_possible = False

        if self.ttc >= self.ttc_treshold and self.flag_possible == True:
            u[0] = 0
            u[1] = 0
        elif self.ttc < self.ttc_treshold and self.ttc > 0 and self.flag_possible == True:
            u[0] = -0.2
            u[1] = 1
        else:
            self.flag_possible = True
            u[0] = 0
            u[1] = 0
        return u
    
class ViBaselineController(controller.StaticController):
    """ 
    Feedback controller
    ---------------------------------------
    r  : reference signal vector  k x 1
    y  : sensor signal vector     m x 1
    u  : control inputs vector    p x 1
    t  : time                     1 x 1
    ---------------------------------------
    u = c( y , r , t ) 

    """
    ############################
    def __init__(self, k, m, p, human_model, acceleration):
        """ """
        
        # Dimensions
        self.k = k   
        self.m = m   
        self.p = p
        self.human_model = human_model
        self.acceleration = acceleration
        self.ttc_min = 2
        
        super().__init__(self.k, self.m, self.p)
        
        # Label
        self.name = 'BaseLine Controller'

    #############################
    def c( self , y , r , t = 0 ):
        """  State feedback (y=x) - no reference - time independent """
        x = y
        u = np.array([0.,0.])
        
        u[0] = self.human_model(x[1],x[0])
        u[1] = 1
        dx = self.acceleration(x, u)
        
        px = 100 - x[0]
        vx = 0 - x[1] - 0.0000001
        ax = 0 - dx[1]- 0.0000001
        ttc = -(vx/ax) - (np.sqrt((vx**2)-(2*px*ax))/ax)
        if ttc > self.ttc_min:
            u[0] = 0
            u[1] = 0
        elif ttc < 0:
            u[0] = 0
            u[1] = 0
        else:
            u[0] = -0.2
            u[1] = 1
        if x[1] <= 0:
            u[0] = -0.0001
            u[1] = 1
        return u
