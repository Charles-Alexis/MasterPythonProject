#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:12:02 2022

@author: clearpath-robot
"""

###############################################################################
import numpy as np
from pyro.control  import controller
import scipy.interpolate as inter
import matplotlib.pyplot as plt
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
    def __init__(self, sys, grid_sys, security_distance = 3):
        """ """
        
        # Dimensions
        self.sys = sys
        self.worst_e = 0
        for di in sys.driver[0]:
            if di[0]> self.worst_e:
                self.worst_e = di[0]
        self.constant_dec_flag = False
        self.constant_dec = -1.0
        self.security_distance = security_distance
        self.grid_sys = grid_sys
        self.slip_max = self.sys.return_max_mu()[1]
        
        super().__init__(self.sys.n, self.sys.m, self.sys.p)
        
        # Label
        self.name = 'BaseLine Controller'

    #############################
    def c( self , x, r , t = 0 ):
        dx = self.sys.f([x[0],x[1]],[0,0], e = self.worst_e)
        if self.constant_dec_flag:
            constant_dec = np.clip(self.constant_dec, self.sys.dmax[1], 0)
            state = [x[0], x[1], constant_dec , self.sys.dmax[1]] #Systeme braking fix
        else:
            state = [x[0], x[1], dx[1], self.sys.dmax[1]]
            
        res_ttc = self.calc_ttc(state)
        res_treshhold = self.calc_ttc_treshhold(state)
        
        if x[0] >= -self.security_distance:
            return [self.slip_max, 1]
        if res_ttc <= res_treshhold:
            return [self.slip_max, 1]
        else:   
            return [0.0, 0]
         
    def calc_ttc(self, state):
        px = -self.security_distance - state[0]
        vx = 0 - state[1]
        ax = 0 - state[2] + 0.0000001
        if self.sys.human_model(state[0],state[1]) == 0 :
            if vx == 0:
                ttc = 1000
            else:
                # ttc = -px/(vx)
                # ttc = -(vx/(0-state[3]))
                ax = 0-state[3] + 0.00001
                if ((vx**2)-(2*px*ax)) <= 0.0:
                    ttc = 1000
                else:
                    ttc_minus = -(vx/ax) - (np.sqrt((vx**2)-(2*px*ax))/ax)
                    ttc_plus = -(vx/ax) + (np.sqrt((vx**2)-(2*px*ax))/ax)
                    if ttc_minus <= ttc_plus:
                        ttc = ttc_minus
                    else:
                         ttc = ttc_plus 
                
        else:
            if ((vx**2)-(2*px*ax)) <= 0.0:
                ttc = 1000
            else:
                ttc_minus = -(vx/ax) - (np.sqrt((vx**2)-(2*px*ax))/ax)
                ttc_plus = -(vx/ax) + (np.sqrt((vx**2)-(2*px*ax))/ax)
                if ttc_minus <= ttc_plus:
                    ttc = ttc_minus
                else:
                     ttc = ttc_plus                 
        return ttc

    def calc_ttc_treshhold(self, state):
        px = -self.security_distance-state[0]
        vx = 0-state[1]
        ax = 0-state[2]
        a = state[2]
        axmax = 0 - (-9.81)
        axb = 0-state[3]

        treshhold_res =  -(vx/axb) + (np.sqrt((vx**2)-(a*px*axmax))/(axmax))
        return treshhold_res

    def c_array(self):
        u = np.zeros(self.grid_sys.x_grid_dim)
        for p in range(len(self.grid_sys.x_level[0])):     
            for v in range(len(self.grid_sys.x_level[1])):
                arr = np.array([ self.grid_sys.x_level[0][p], self.grid_sys.x_level[1][v]])
                u[p][v] = self.c(arr, 0)[0]
        return u



    #############################
    def c_plot( self , x, r , t = 0 ):
        dx = self.sys.f([x[0],x[1]],[0,0], e = self.worst_e)
        if self.constant_dec_flag:
            constant_dec = np.clip(self.constant_dec, self.sys.dmax[1], 0)
            state = [x[0], x[1], constant_dec , self.sys.dmax[1]] #Systeme braking fix
        else:
            state = [x[0], x[1], dx[1], self.sys.dmax[1]]
            
        res_ttc = self.calc_ttc(state)
        res_treshhold = self.calc_ttc_treshhold(state)
        
        if x[0] >= -self.security_distance:
            return [self.slip_max, 1]
        if res_ttc <= res_treshhold:
            return [self.slip_max, 1]
        else:   
            return [0.0, 0]

    def c_array_plot(self, constant_dec_custom):
        u = np.zeros(self.grid_sys.x_grid_dim)
        for p in range(len(self.grid_sys.x_level[0])):     
            for v in range(len(self.grid_sys.x_level[1])):
                arr = np.array([ self.grid_sys.x_level[0][p], self.grid_sys.x_level[1][v]])
                x = arr
                dx = self.sys.f([x[0],x[1]],[0,0], e = 0)
                if self.constant_dec_flag:
                    constant_dec = np.clip(constant_dec_custom, self.sys.dmax[1], 0)
                    state = [x[0], x[1], constant_dec , self.sys.dmax[1]] #Systeme braking fix
                else:
                    state = [x[0], x[1], dx[1], self.sys.dmax[1]]
                    
                res_ttc = self.calc_ttc_plot(state)
                res_treshhold = self.calc_ttc_treshhold_plot(state)

                if res_ttc <= res_treshhold:
                    res_treshhold = self.slip_max
                else:   
                    res_treshhold = 0
                u[p][v] = res_treshhold
        return u
    
    def calc_ttc_treshhold_plot(self, state):
        px = -3-state[0]
        vx = 0-state[1]
        ax = 0-state[2]
        a = state[2]
        axmax = 0-state[3]
        axb = 0-state[3]
        if (vx**2)-(a*px*axmax) <= 0:
            treshhold_res = 0
        else:
            treshhold_res =  -(vx/axb) + (np.sqrt((vx**2)-(a*px*axmax))/(axmax))
        return treshhold_res
    
    def c_array_tresh(self, constant_dec_custom = -3):
        u = np.zeros(self.grid_sys.x_grid_dim)
        for p in range(len(self.grid_sys.x_level[0])):     
            for v in range(len(self.grid_sys.x_level[1])):
                arr = np.array([ self.grid_sys.x_level[0][p], self.grid_sys.x_level[1][v]])
                x = arr
                dx = self.sys.f([x[0],x[1]],[0,0], e = 0)
                if self.constant_dec_flag:
                    constant_dec = np.clip(constant_dec_custom, self.sys.dmax[1], 0)
                    state = [x[0], x[1], constant_dec , self.sys.dmax[1]] #Systeme braking fix
                else:
                    state = [x[0], x[1], dx[1], self.sys.dmax[1]]
                    
                res_treshhold = self.calc_ttc_treshhold_plot(state)
                if res_treshhold == np.nan:
                    res_treshhold = 0
                u[p][v] = res_treshhold
        return u

    def calc_ttc_plot(self, state):
        px = -3 - state[0]
        vx = 0 - state[1]
        ax = 0 - state[2] + 0.0000001

        if ((vx**2)-(2*px*ax)) <= 0.0:
            ttc = 10000
        else:
            ttc_minus = -(vx/ax) - (np.sqrt((vx**2)-(2*px*ax))/ax)
            ttc_plus = -(vx/ax) + (np.sqrt((vx**2)-(2*px*ax))/ax)
            if ttc_minus <= ttc_plus:
                ttc = ttc_minus
            else:
                 ttc = ttc_plus                 
        return ttc

    def c_array_ttc(self, constant_dec_custom = -3):
        u = np.zeros(self.grid_sys.x_grid_dim)
        for p in range(len(self.grid_sys.x_level[0])):     
            for v in range(len(self.grid_sys.x_level[1])):
                arr = np.array([ self.grid_sys.x_level[0][p], self.grid_sys.x_level[1][v]])
                x = arr
                dx = self.sys.f([x[0],x[1]],[0,0], e = 0)
                if self.constant_dec_flag:
                    constant_dec = np.clip(constant_dec_custom, self.sys.dmax[1], 0)
                    state = [x[0], x[1], constant_dec , self.sys.dmax[1]] #Systeme braking fix
                else:
                    state = [x[0], x[1], dx[1], self.sys.dmax[1]]
                    
                res_treshhold = self.calc_ttc_plot(state)
                if res_treshhold != 1000 and res_treshhold >= 0:
                    u[p][v] = res_treshhold
                else:
                    u[p][v] = np.nan
        return u

class MSDController(controller.StaticController):
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
    def __init__(self, sys, grid_sys, security_distance = 3):
        """ """
        
        # Dimensions
        self.sys = sys
        self.grid_sys = grid_sys
        self.security_distance = security_distance
        self.minimale_esp = 0
        for d in self.sys.driver[0]:
            if d[0]>self.minimale_esp:
                self.minimale_esp=d[0]
            
        self.slip_data = self.sys.return_max_mu()
        self.amax = self.sys.f([sys.x_lb[0],sys.x_ub[1]],[self.slip_data[1],1])[1]
        
        super().__init__(self.sys.n, self.sys.m, self.sys.p)
        
        # Label
        self.name = 'MSD Controller'

    #############################
    def c( self , x , r , t = 0, e = 0 ):
        px = -self.security_distance-x[0]
        vx = np.abs(0-x[1])
        axmax = 0-self.amax
        t1 = 0.75 
        t2 = 0.75
        t3 = 0.25
        
        Sa = vx*(t1+t2+t3) - ((axmax*(t3**2))/6) + ((vx**2)/(2*axmax)) - ((vx*t3)/2) + (axmax*(t3**2)/8)
    
        if Sa >= px:
            return [self.slip_data[1],1]
        else:
            return [0.,0.]   
    
    def c_array(self):
        u = np.zeros(self.grid_sys.x_grid_dim)
        for p in range(len(self.grid_sys.x_level[0])):     
            for v in range(len(self.grid_sys.x_level[1])):
                arr = np.array([ self.grid_sys.x_level[0][p], self.grid_sys.x_level[1][v]])
                u[p][v] = self.c(arr, 0)[0]
        return u

    def c_array_worst_e(self):
        u = np.zeros(self.grid_sys.x_grid_dim)
        for p in range(len(self.grid_sys.x_level[0])):     
            for v in range(len(self.grid_sys.x_level[1])):
                arr = np.array([ self.grid_sys.x_level[0][p], self.grid_sys.x_level[1][v]])
                u[p][v] = self.c(arr, 0,e = self.minimale_esp)[0]
        return u


class viController(controller.StaticController):
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
    def __init__(self, sys, grid_sys, dp):
        """ """
        # Dimensions
        self.sys = sys
        self.grid_sys = grid_sys
        self.dp = dp
        
        super().__init__(self.sys.n, self.sys.m, self.sys.p)
        
        self.levels = tuple(self.grid_sys.x_level[i] for i in range(self.grid_sys.sys.n))
        self.c_func = inter.RegularGridInterpolator(self.levels, self.dp.cleared_data, 'linear', True, fill_value=np.nan)
        
        # Label
        self.name = 'Human model controller'

    #############################
    def c( self , y , r , t = 0 ):
        slip = self.c_func([y[0] ,y[1]])[0]
        if slip != 0:
            override = 1
        else:
            override = 0
        
        return np.array([slip,override])
    
    def c_array(self):
        u = np.zeros(self.grid_sys.x_grid_dim)
        for p in range(len(self.grid_sys.x_level[0])):     
            for v in range(len(self.grid_sys.x_level[1])):
                arr = np.array([ self.grid_sys.x_level[0][p], self.grid_sys.x_level[1][v]])
                u[p][v] = self.c(arr, 0)[0]
        return u
    
    def plot_difference_with_cleared_data(self):
        u_cleared = self.c_array()
        u_not_clear = self.grid_sys.get_grid_from_array(self.grid_sys.get_input_from_policy(self.dp.pi, 0))
        
        fig, axs = plt.subplots(1, 2)
        plt.ion()
        i = axs[0].pcolormesh(self.grid_sys.x_level[0], self.grid_sys.x_level[1], u_cleared.T, shading='gouraud', cmap = 'plasma')
        i = axs[1].pcolormesh(self.grid_sys.x_level[0], self.grid_sys.x_level[1], u_not_clear.T, shading='gouraud', cmap = 'plasma')
        axs[0].axis([self.grid_sys.x_level[0][0], self.grid_sys.x_level[0][-1], self.grid_sys.x_level[1][0], self.grid_sys.x_level[1][-1]])
        axs[1].axis([self.grid_sys.x_level[0][0], self.grid_sys.x_level[0][-1], self.grid_sys.x_level[1][0], self.grid_sys.x_level[1][-1]])
        fig.colorbar(i, ax=axs[0])
        fig.colorbar(i, ax=axs[1])
        axs[0].grid()
        axs[0].set_ylabel('Vitesse')
        axs[0].set_xlabel('Position')
        axs[1].grid()
        axs[1].set_ylabel('Vitesse')
        axs[1].set_xlabel('Position')
    
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
    def __init__(self, sys, grid_sys):
        """ """
        
        # Dimensions
        self.sys = sys
        self.grid_sys = grid_sys
        
        super().__init__(self.sys.n, self.sys.m, self.sys.p)
        
        # Label
        self.name = 'Human model controller'

    #############################
    def c( self , y , r , t = 0 ):
        slip = self.sys.human_model(y[0],y[1])
        if slip != 0:
            override = 1
        else:
            override = 0
        
        return np.array([slip,override])
    
    def c_array(self):
        u = np.zeros(self.grid_sys.x_grid_dim)
        for p in range(len(self.grid_sys.x_level[0])):     
            for v in range(len(self.grid_sys.x_level[1])):
                arr = np.array([ self.grid_sys.x_level[0][p], self.grid_sys.x_level[1][v]])
                u[p][v] = self.c(arr, 0)[0]
        return u

