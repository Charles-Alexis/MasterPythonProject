# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:59:10 2022

@author: Charles-Alexis
"""

import numpy as np
import matplotlib.pyplot as plt
from pyro.dynamic  import longitudinal_vehicule
from pyro.dynamic import system


class LongitudinalFrontWheelDriveCarWithDriverModel( system.ContinuousDynamicSystem ):    
    ############################
    def __init__(self):
        """ """
    
        # Dimensions
        self.n = 2   
        self.m = 2   
        self.p = 2
        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, self.n, self.m, self.p)
        
        # Labels
        self.name = 'Front Wheel Drive Car'
        self.state_label = ['x','dx']
        self.input_label = ['slip','override']
        self.output_label = ['x','dx']
        
        # Units
        self.state_units = ['[m]','[m/sec]']
        self.input_units = ['[]','[]']
        self.output_units = ['[m]','[m/sec]']
        
        self.x_ub = np.array([+100, 20])
        self.x_lb = np.array([0, 0])
        self.u_ub = np.array([0.0, 1])
        self.u_lb = np.array([-0.3, 0])
        self.x_grid = np.array([100,100])
        
        # w
        self.temp_w = None 
        self.flag_w = False
        
        # Model param
        self.lenght  = 2          # distance between front wheel and back wheel [m]
        self.xc      = 1          # distance from back wheel to c.g.  [m]
        self.yc      = 0.5        # height from ground to c.g.  [m]
        self.mass    = 1500       # total car mass [kg]
        self.gravity = 9.81       # gravity constant [N/kg]
        self.rho     = 1.225      # air density [kg/m3]
        self.cdA     = 0.3 * 2    # drag coef time area [m2]

        
        # Ground traction curve parameters
        self.roads = {
            'AsphalteDry': [1.2801, 23.99, 0.52, 0.02, "AsphalteDry"],
            'CementDry': [1.1973, 25.168, 0.5373, 0.02, "CementDry"],
            'AsphalteWet': [0.857, 33.822, 0.347, 0.02, "AsphalteWet"],
            'CobblestoneWet': [0.400, 33.71, 0.12, 0.02, "CobblestoneWet"],
            'Snow': [0.1946, 94.129, 0.0646, 0.02, "Snow"],
            'Ice': [0.05, 306.39, 0, 0.02, "Ice"], 
            }      
        self.road = self.roads['AsphalteDry']
        
        # Driver models
        self.flag_human_model = False
        self.dmax = np.nan

        self.driver_jackal = {
            'Bad': [9.7, 10.5,'Bad'],
            'Ok': [9.5, 10.,'Ok'],
            'Good': [9.2, 10.,'Good'],
            }
        
        self.driver_xmaxx = {
            'Bad': [18.7, 19.5,'Bad'],
            'Ok': [18., 19.,'Ok'],
            'Good': [17.2, 18.5,'Good'],
            }
        
        self.driver_xmaxx_fort = {
            'Bad': [15.0, 18.,'Bad'],
            'Ok': [13.5, 16.5,'Ok'],
            'Good': [12.0, 15.0,'Good'],
            }
              
        self.drivers10 = {
            'Bad': [85,98,'Bad'],
            'Ok': [80,95,'Ok'],
            'Good': [75,90,'Good'],
            }
        
        self.drivers10_v2 = {
            'Bad': [60,85,'Bad'],
            'Ok': [55,80,'Ok'],
            'Good': [50,75,'Good'],
            }
        
        self.drivers15 = {
            'Bad': [30,45,'Bad'],
            'Ok': [20,35,'Ok'],
            'Good': [10,25,'Good'],
            }
        
        self.drivers20 = {
            'Bad': [25,35,'Bad'],
            'Ok': [15,25,'Ok'],
            'Good': [5,15,'Good'],
            }
        
        self.driver = self.drivers10['Good']
        
        # Graphic output parameters 
        self.dynamic_domain  = False
        self.dynamic_range   = self.lenght * 2
        
        # Animation output graphical parameters
        self.linestyle = '-'
        self.obs_dist =  self.x_ub[0] + self.lenght * 2 # using the upper bound on x range
        self.obs_size = 2
        
    def return_max_mu(self):
        slip_min = -0.3 
        slip_max = 0.3
        slips = np.arange(slip_min, slip_max, 0.001)
        force = np.zeros(len(slips))
        for slip in range(len(slips)):
            force[slip] = self.slip2force(slips[slip], self.x_ub[1])
        
        return [np.max(force), slips[np.argmax(force)]]

    #############################
    def compute_ratios(self):
        """ Shorcut function for comuting usefull length ratios """
        
        ry = self.yc / self.lenght
        rr = self.xc / self.lenght # ratio of space rear of the c.g.
        rf = 1 - rr                # ratio of space in front of c.g.
        
        return ry, rr, rf
    
    #############################
    def slip2force(self, slip, v, mu_max=None, mu_slope=None):
        """ Shorcut function for comuting usefull length ratios """
        if slip > 0:
            mu = ((self.road[0]*(1 - np.exp(-1*self.road[1]*slip)))-(self.road[2]*slip))*np.exp(-1*self.road[3]*slip*v)
        else:
            slip = np.abs(slip)
            mu = -1*((self.road[0]*(1 - np.exp(-1*self.road[1]*slip)))-(self.road[2]*slip))*np.exp(-1*self.road[3]*slip*v)
        
        return mu

    #############################
    def plot_slip2force(self):    
        slip_min = -0.3 
        slip_max = 0.3
        slips = np.arange( slip_min, slip_max, 0.001 )
        force = np.zeros(len(slips))
          
        fig = plt.figure(figsize=(4, 2), dpi=300, frameon=True)
        fig.canvas.manager.set_window_title('Ground traction curve')
        ax  = fig.add_subplot(1, 1, 1)
        
        temp = self.road
        for key in self.roads:
            self.road = self.roads[key]
            i = 0
            for s in slips:
                force[i] = self.slip2force(s, 20)
                i = i + 1
            print(np.max(force), slips[np.argmax(force)], self.road)
            ax.plot( slips , force, label = key)

        self.road = temp

        ax.set_ylabel('mu = |Fx/Fz|', fontsize=5)
        ax.set_xlabel('Slip ratio', fontsize=5 )
        ax.tick_params( labelsize = 5 )
        ax.grid(True)
        ax.legend(loc = 'upper left', fontsize=5)
        fig.tight_layout()
        fig.canvas.draw()
        plt.show()
        
    #############################
    def f(self, x , u , t = 0 ):
        """ 
        Continuous time foward dynamics evaluation
        
        dx = f(x,u,t)
        
        INPUTS
        x  : state vector             n x 1
        u  : control inputs vector    m x 1
        t  : time                     1 x 1
        
        OUPUTS
        dx : state derivative vectror n x 1
        
        """
        
        dx = np.zeros(self.n) # State derivative vector
        
        ###################
        
        slip = u[0]
        override = u[1]
        pos = x[0]
        v    = x[1]
        
        if override != 1:
            slip = self.human_model(v,pos)
        mu = self.slip2force(slip, v) 
        
        # constant params local vairables
        ry, rr, rf = self.compute_ratios() 
        m    = self.mass 
        g    = self.gravity
        rcda = self.rho * self.cdA
        
        # Drag froce
        fd = 0.5 * rcda * v * np.abs( v ) # drag froce with the right sign
        a  = (mu * m * g * rr - fd)/( m * (1 + mu * ry ))
      
        ###################
        dx[0]  = v # velocity
        dx[1]  = a # acc
        
        ###################
        # Normal force check
        fn_front = m * g * rr - m * a * ry
        fn_rear  = m * g * rf + m * a * ry
        if (fn_front<0) :
            print('Normal force on front wheel is negative: fn = ', fn_front)
        if (fn_rear<0) : 
            print('Normal force on rear wheel is negative: fn = ', fn_rear)
        ###################
        
        return dx

    def human_model(self,y, x):
      dist1 = self.driver[0]
      dist2 = self.driver[1]
      vmax = self.x_ub[1]
      m1 = -1
      m2 = m1
      m_p = -1/m1
      b1 = vmax - (m1*dist1)
      b2 = vmax - (m2*dist2)
      
      if self.flag_human_model is False:
        self.flag_human_model = True
        y_test = self.x_ub[1]/2
        x_test = m1*(y_test-b1)
        b_test = (y_test - x_test)/m_p
        x_test2 = (b_test - b2)/(m2-m_p)
        y_test2 = m_p*x_test2+b_test
        self.dmax = np.sqrt(((y_test2-y_test)**2)+((x_test2-x_test)**2))
        
      if (m1*x+b1-y) >= 0:
        slip = self.u_ub[0]
      elif (m2*x+b2-y) <= 0:
        slip = self.u_lb[0]
      else:
        b = y - m_p*x
        x_d = (b-b1)/(m1-m_p)
        y_d = x_d*m_p+b
        d = np.sqrt(((y-y_d)**2)+((x-x_d)**2))
        slip = self.u_lb[0]*(d/self.dmax)      
      return slip  

  
    def plot_human_model(self, plot=True):      
      pos = np.arange(0, (self.x_ub[0]+0.0001), (self.x_ub[0]/(self.x_grid[0]-1)))
      vit = np.arange(0, (self.x_ub[1]+0.00001), (self.x_ub[1]/(self.x_grid[1]-1)))
      i = 0
      j = 0
      grid= np.zeros([self.x_grid[0],self.x_grid[1]])
      for y in vit:
        for x in pos:
          grid[j][i] = self.human_model(x,y)
          j = j+1
        j=0
        i = i+1
        
      if plot is not True:
           return grid
      fig, axs = plt.subplots(1, 1)
      plt.ion()
      fig.suptitle('Driver: ' + self.driver[-1])
      xname = 'x'
      yname = 'dx'
      axs.set(xlabel=xname, ylabel=yname)
      ci = axs.pcolormesh(pos, vit, grid.T, shading='gouraud')
      axs.axis([self.x_lb[0], self.x_ub[0], self.x_lb[1], self.x_ub[1]])
      fig.colorbar(ci, ax=axs)
      axs.grid(True)
#      fig = plt.figure()
#      ax = fig.add_subplot(111, projection='3d')
#      ax.contourf(pos,vit,grid,150)
#      ax.set_ylabel('vit')
#      ax.set_xlabel('pos')
#      plt.show()
    
    ###########################################################################
    # For graphical output
    ###########################################################################
    
    
    #############################
    def xut2q( self, x , u , t ):
        """ compute config q """
        
        q   = np.append(  x , u[0] ) # steering angle is part of the config
        
        return q
    
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = self.dynamic_range
        
        x = q[0]
        y = 0
        z = 0
        
        if self.dynamic_domain:
        
            domain  = [ ( -l + x , l + x ) ,
                        ( -l + y , l + y ) ,
                        ( -l + z , l + z ) ]#  
        else:
            
            domain  = [ ( 0 , self.obs_dist + self.obs_size * 2 ) ,
                        ( 0 , 1 ) ,
                        ( 0 , self.obs_dist + self.obs_size * 2 ) ]#
            
                
        return domain
    
    
    ###########################################################################
    def forward_kinematic_lines(self, q ):
        """ 
        Compute points p = [x;y;z] positions given config q 
        ----------------------------------------------------
        - points of interest for ploting
        
        Outpus:
        lines_pts = [] : a list of array (n_pts x 3) for each lines
        
        """
        
        # Variables
        
        travel   = q[0]
        slipping = (np.abs( q[2] ) > 0.03 ) # bool
        
        
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        
        
        ###########################
        # bottom line
        ###########################
        
        pts = np.zeros((2,3))
        
        pts[0,:] = [ -1000  , 0 , 0 ]
        pts[1,:] = [  1000  , 0 , 0 ]
        
        lines_pts.append( pts )
        
        ###########################
        # obstacle
        ###########################
        
        pts = np.zeros((5,3))
        
        d = self.obs_size
        
        pts[0,:] = [ 0  , 0 , 0 ]
        pts[1,:] = [ d  , 0 , 0 ]
        pts[2,:] = [ d  , d , 0 ]
        pts[3,:] = [ 0  , d , 0 ]
        pts[4,:] = [ 0  , 0 , 0 ]
        
        pts[:,0] = pts[:,0] + self.obs_dist

        
        lines_pts.append( pts )
        
        ###########################
        # Vehicule
        ###########################
        
        pts = np.zeros((13,3))
        
        r = 0.3
        x1 = 1
        y1 = 1
        y2 = 1.5
        y3 = 1.4
        x2 = 1
        x3 = 1
        y3 = 0.6
        
        l = self.lenght
        
        pts[0,:]  = [ 0  , 0 , 0 ]
        pts[1,:]  = [ -x1  , 0 , 0 ]
        pts[2,:]  = [ -x1  , y1 , 0 ]
        pts[3,:]  = [ 0  , y2 , 0 ]
        pts[4,:]  = [ l , y2 , 0 ]
        pts[5,:]  = [ l - x2 , y2 , 0 ]
        pts[6,:]  = [ l - x2  , y1 , 0 ]
        pts[7,:]  = [ l  , y1 , 0 ]
        pts[8,:]  = [ l  , y2 , 0 ]
        pts[9,:]  = [ l  , y1 , 0 ]
        pts[10,:] = [ l+x3  , y3 , 0 ]
        pts[11,:] = [ l+x3  , 0 , 0 ]
        pts[12,:] = [ 0  , 0 , 0 ]


        pts[:,0] = pts[:,0] + travel  # translate horizontally the car postion
        pts[:,1] = pts[:,1] + r       # translate vertically the wheel radius
        
        lines_pts.append( pts )
        
        ###########################
        # Wheels
        ###########################
        
        if slipping:
            r = r*1.2
        
        angles = np.arange(0,6.4,0.1)
        n      = angles.size
        
        pts = np.zeros((n,3))
        
        for i in range(n):
            a = angles[i]
            pts[i,:] = [ r * np.cos(a) , r * np.sin(a) , 0 ]

        pts[:,0] = pts[:,0] + travel
        pts[:,1] = pts[:,1] + r
        
        lines_pts.append( pts )
        
        pts = pts.copy()
        pts[:,0] = pts[:,0] + l
        
        lines_pts.append( pts )
                
        return lines_pts
