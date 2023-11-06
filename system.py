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
        
        self.roads_array_name = ['Asphalte Sec','Ciment Sec','Asphalte Mouillée','Gravier Mouillé','Neige','Glace']
        # Labels
        self.name = 'Front Wheel Drive Car'
        self.state_label = ['x','dx']
        #self.input_label = ['slip','override']
        self.input_label = ['slip']
        self.output_label = ['x','dx']
        
        # Units
        self.state_units = ['[m]','[m/sec]']
        self.input_units = ['[]','[]']
        self.output_units = ['[m]','[m/sec]']
        
        self.x_ub = np.array([+100, 20])
        self.x_lb = np.array([0, 0])
        self.u_ub = np.array([0.0, 1])
        self.u_lb = np.array([-1, 0])
        self.x_grid = np.array([100,100])
        
        # Model param
        self.lenght  = 2          # distance between front wheel and back wheel [m]
        self.xc      = 1          # distance from back wheel to c.g.  [m]
        self.yc      = 0.5        # height from ground to c.g.  [m]
        self.mass    = 1500       # total car mass [kg]
        self.gravity = 9.81       # gravity constant [N/kg]
        self.rho     = 1.225      # air density [kg/m3]
        self.cdA     = 0.3 * 2    # drag coef time area [m2]
        self.u_dim = [80]
        self.u_level = (np.arange(0,(self.u_dim[0])) - (self.u_dim[0]-1)) /  ((self.u_dim[0]-1)/-self.u_lb[0])
        
        # Ground traction curve parameters
        self.roads = {
            'AsphalteDry': [1.2801, 23.99, 0.52, 0.02, "AsphalteDry"],
            'CementDry': [1.1973, 25.168, 0.5373, 0.02, "CementDry"],
            'AsphalteWet': [0.857, 33.822, 0.347, 0.02, "AsphalteWet"],
            'CobblestoneDry': [1.37, 6.46, 0.67, 0.02, "CobblestoneDry"],
            'CobblestoneWet': [0.400, 33.71, 0.12, 0.02, "CobblestoneWet"],
            'Snow': [0.1946, 94.129, 0.0646, 0.02, "Snow"],
            'Ice': [0.05, 306.39, 0, 0.02, "Ice"], 
            } 
                
        self.road = self.roads['AsphalteDry']
        
        self.roads_array = ['d\'asphalte sec',' de ciment sec','d\'asphalte mouillée','de gravier sec','de gravier mouillé','de neige','de glace']
        self.drivers_array = ['bon conducteur','conducteur Normale','mauvais conducteur','conducteur endormi','Pas d\'espérence']   
        self.roads_ind = 2
        self.drivers_ind = 2
        
        self.timing = 0.5
        self.timing_conservateur = -1*self.timing
        self.timing_normal = +0.0
        self.timing_aggressif = self.timing
        self.timing_sleep = +100.0
            
        # self.drivers = {
        #     'Bad': [[[self.timing_conservateur, 0.05], [self.timing_normal, 0.20], [self.timing_aggressif, 0.75]], "bad"],
        #     'Normal': [[[self.timing_conservateur, 0.25], [self.timing_normal, 0.50], [self.timing_aggressif, 0.25]], "normal"],
        #     'Good': [[[self.timing_conservateur, 0.85], [self.timing_normal, 0.10], [self.timing_aggressif, 0.05]], "good"],
        #     'Sleepy': [[[self.timing_normal, 0.01], [self.timing_sleep, 0.99]], "sleepy"],
        #     'Null': [[[+0.0, 1.0]], "null"]
        #     } 
        
        self.timing = 0.6
        
        self.timing_mm = -1*self.timing
        self.timing_m = -1*self.timing/2
        self.timing_0 = 0.0
        self.timing_p = self.timing/2
        self.timing_pp = self.timing
        self.timing_s = +100.0
        
        self.drivers = {
            'Bad': [ [[self.timing_mm, 0.05],
                      [self.timing_m, 0.10],
                      [self.timing_0, 0.20],
                      [self.timing_p, 0.40],
                      [self.timing_pp, 0.25]], "Mauvais"],
            
            'Normal': [ [[self.timing_mm, 0.05],
                      [self.timing_m, 0.20],
                      [self.timing_0, 0.65],
                      [self.timing_p, 0.08],
                      [self.timing_pp, 0.02]], "Normal"],
            
            'Good': [ [[self.timing_mm, 0.25],
                      [self.timing_m, 0.5],
                      [self.timing_0, 0.20],
                      [self.timing_p, 0.03],
                      [self.timing_pp, 0.02]], "Bon"],
            
            'Sleepy': [[[self.timing_normal, 0.01], [self.timing_sleep, 0.99]], "Endormi"],
            'Null': [[[0.0, 1.0]], "null"]
            } 

        # Driver models
        self.flag_human_model = False
        self.dmax = np.nan
        
        
        self.driver_type = None
        self.driver = None
        self.use_human_model = True
        # Graphic output parameters 
        self.dynamic_domain  = False
        self.dynamic_range   = self.lenght * 2
        
        self.tm = 3.5
        self.tf = 1.75
        self.tm_dot = 0.75
        self.tm_coef = 0.6
        self.mu_coef = 1
        
        # Animation output graphical parameters
        self.linestyle = '-'
        self.obs_dist =  self.x_ub[0] + self.lenght * 2 # using the upper bound on x range
        self.obs_size = 2
        self.best_slip = -0.2
        
        
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
        
        mu = -1*((self.road[0]*(1 - np.exp(-1*self.road[1]*slip)))-(self.road[2]*slip))*np.exp(-1*self.road[3]*slip*v)
        
        if slip < 0: 
            slip = np.abs(slip)
            mu = -1*((self.road[0]*(1 - np.exp(-1*self.road[1]*slip)))-(self.road[2]*slip))
        else:
            slip = np.abs(slip)
            mu = ((self.road[0]*(1 - np.exp(-1*self.road[1]*slip)))-(self.road[2]*slip))
        
        return mu

    def return_max_mu(self):
        slip_min = -0.3 
        slip_max = 0.
        slips = np.arange(slip_min, slip_max, 0.001)
        force = np.zeros(len(slips))
        for slip in range(len(slips)):
            force[slip] = self.slip2force(slips[slip], self.x_ub[1])

        slip_min = 0
        slip_max = 0.3
        slips = np.arange(slip_min, slip_max, 0.001)
        force = np.zeros(len(slips))
        for slip in range(len(slips)):
            force[slip] = self.slip2force(slips[slip], self.x_ub[1])
        
        return [-np.max(force), -slips[np.argmax(force)]]

    def return_max_slip_mu(self):
        slip = -0.5
        mu = self.slip2force(slip, self.x_ub[1])
        return mu  


    def return_slip_wanted(self, force_wanted):
        def find_nearest_arg(value, array):
             array = np.asarray(array)
             idx = (np.abs(array - value)).argmin()
             return idx
        
        slip_min = -0.2 
        slip_max = 0.0
        slips = np.arange(slip_min, slip_max, 0.001)
        force = np.zeros(len(slips))
        for slip in range(len(slips)):
            force[slip] = self.f([-10,4.5] , [slips[slip], 1])[1]
        
        f_arg = find_nearest_arg(force_wanted, force)
        return [force_wanted, slips[f_arg]]

    #############################
    def plot_slip2force(self, plot_slip = True):    
        slip_min = -1
        slip_max = 1
        slips = np.arange( slip_min, slip_max, 0.001 )
        force = np.zeros(len(slips))
        if plot_slip is True:
            fig = plt.figure(figsize=(4, 4), dpi=300, frameon=True)
            fig.canvas.manager.set_window_title('Ground traction curve')
            ax  = fig.add_subplot(1, 1, 1)
        
        temp = self.road
        colors = ['dimgray','goldenrod','black','tomato','firebrick','royalblue','lightskyblue']
        index_color = 0
        i_road = 0
        for key in self.roads:
            self.road = self.roads[key]
            i = 0
            for s in slips:
                force[i] = self.slip2force(s, 0)
                i = i + 1
            if plot_slip is True:    
                print(np.max(force), slips[np.argmax(force)], np.min(force), slips[np.argmin(force)])
                ploting_line_max = np.arange(0,np.max(force),0.01)
                ploting_line_min = np.arange(np.min(force),0,0.01)
                
                label_name = self.roads_array_name[i_road] 
                # ax.fill_between()
                if key != 'CobblestoneDry':
                    ax.plot(np.zeros(len(ploting_line_max))+slips[np.argmax(force)], ploting_line_max, colors[index_color],linestyle='--', linewidth = 1)
                    ax.plot(np.zeros(len(ploting_line_min))+slips[np.argmin(force)], ploting_line_min, colors[index_color],linestyle='--', linewidth = 1)
                    
                    ax.plot( slips , force, colors[index_color], label = label_name)
                    i_road = i_road+1
                index_color = index_color+1
                
        self.road = temp


        if plot_slip is True:
            ax.set_ylabel('mu = |Fx/Fz|', fontsize=5)
            ax.set_xlabel('Slip ratio', fontsize=5 )
            ax.tick_params( labelsize = 5 )
            ax.grid(True)
            ax.legend(loc = 'upper left', fontsize=5)
            fig.tight_layout()
            fig.canvas.draw()
            plt.show()

        return [np.max(force), slips[np.argmax(force)]]
    
    #############################
    def best_slip2force(self, plot_slip = True):    
        slip_min = -0.3 
        slip_max = 0.3
        slips = np.arange( slip_min, slip_max, 0.001 )
        force = np.zeros(len(slips))
        
        temp = self.road
        for key in self.roads:
            self.road = self.roads[key]
            i = 0
            for s in slips:
                force[i] = self.slip2force(s, 4.5)
                i = i + 1
        self.road = temp

        return [np.max(force), slips[np.argmax(force)]]
        
    #############################
    def f(self, x , u, t = 0, e = 0):
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
        if self.use_human_model == True:
            if override != 1.:
                slip = self.human_model(pos, v, e)
        mu = self.slip2force(slip, v) 

        # constant params local vairables
        ry, rr, rf = self.compute_ratios() 
        m    = self.mass 
        g    = self.gravity
        rcda = self.rho * self.cdA
        
        # Drag froce
        fd = 0.5 * rcda * v * np.abs( v ) # drag froce with the right sign
        a  = (mu * m * g * rr - (fd*((mu*ry)+1)))/( m * (1 + mu * ry ))
        
        # Enlevé une décélération negative
        if (v + a*0.1) < 0:
            a = (-v)/0.1
        if v == 0:
            a = 0
            
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
        
        #ACCIDENT
        if pos + dx[0]*0.1 >=0:
            dx[0]  = -(pos/0.1)
            #dx[1]  = -(v/0.1) # acc
        if pos == 0:
            dx[1]  = -(v/0.1) # acc
        return dx

    def human_model_2(self, pos, vit, e = 0):
        vx = 0-vit
        px = -1-pos
        ax_max = 0-(self.dmax[1]*self.tm_coef)
        if (vx**2)-(2*px*ax_max) > 0:
            self.tm =((-(vx)/ax_max) - (np.sqrt((vx**2)-(2*px*ax_max))/ax_max)) + 2
        else:
            self.tm = -vx/(ax_max)
        
        ttc = -(px)/(0-vit-0.00000001)
        if pos > -1.0:
             slip = -1
             return slip
        elif ttc < (self.tm-(self.tf+e)):
             
             a_desired = -1 * (self.tm_dot+1) * (vit**2) / np.abs(px)
    
             a = np.clip(a_desired, (-1*ax_max), 0.) 
             slip = a/ax_max
             return slip  
        else:
             return 0

    def human_model(self, pos, vit, e = 0):
        max_slip = -self.u_lb[0]
        
        security_distance = 0.5
        vx = 0-vit
        px = -security_distance-pos
        a_max = (self.dmax[1]*self.tm_coef)
        #a_max = (-8.56*self.tm_coef)
        ax_max = 0-a_max
        
        #DECELERATION VOULUE
        if ax_max == 0:
            tp = 1    
        else:
            tp = (-px/(vx+0.0000000000001)) / (-(vx+0.0000000000001)/ax_max)
        D = (2*(1+self.tm_dot)*(1+(2*self.tm_dot*tp))) 
        
        if D<0:
            D=0
        D = D**(-(2+(1/self.tm_dot)))
        a_desired = -D*(ax_max)
        a = np.clip(a_desired, a_max, 0.) 
        
        slip = -max_slip*a/a_max
        slip = self.find_nearest(self.u_level, slip)
        ax = 0-a
        
        if ax == 0:
            tm = (-px/(vx+0.0000000000001))+self.tf
        else:
            tm =-(vx/(ax_max))+self.tf
            
           
        tm =-(vx/(ax_max))+self.tf
        ttc = -(px)/(vx+0.0000000000001)
        
        if pos > -security_distance:
             return -max_slip
        elif ttc < (tm-(self.tf+e)):
             return slip 
        else:
             return 0

    
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx] 
    
    def find_buuged_states(self, plotting=False):
        self.bugged_states_pos = np.zeros(self.x_grid)
        self.bugged_states_vit = np.zeros(self.x_grid)
        self.bugged_states = np.zeros(self.x_grid)
        
        pos_array = (np.arange(-self.x_grid[0],0)+1)/((self.x_grid[0]-1)) * -self.x_lb[0]
        vit_array = np.arange(0, self.x_grid[1])/((self.x_grid[1]-1)/self.x_ub[1])
        self.pos_array = pos_array
        self.vit_array = vit_array
        for p in range(len(pos_array)):
            for v in range(len(vit_array)):
                position = pos_array[p]
                vitesse = vit_array[v]
                dx = self.f([position,vitesse], [0,0], 0)
                
                next_state_pos = position + dx[0] * 0.1
                next_state_vit = vitesse + dx[1] * 0.1
                pos_near = self.find_nearest(pos_array, next_state_pos)
                vit_near = self.find_nearest(vit_array, next_state_vit)
                if pos_near == position:
                    self.bugged_states_pos[p][v] = 1
                
                if vit_near == vitesse:
                    self.bugged_states_vit[p][v] = 1

                if vit_near == vitesse and pos_near == position:
                    self.bugged_states[p][v] = 1
        if plotting:        
            fig, axs = plt.subplots(1,3)
            plt.ion()
            i = axs[0].pcolormesh(pos_array, vit_array, self.bugged_states_pos.T, shading='gouraud', cmap = 'plasma')
            fig.colorbar(i, ax=axs[0])
    
            i = axs[1].pcolormesh(pos_array, vit_array, self.bugged_states_vit.T, shading='gouraud', cmap = 'plasma')
            fig.colorbar(i, ax=axs[1])
            
            i = axs[2].pcolormesh(pos_array, vit_array, self.bugged_states.T, shading='gouraud', cmap = 'plasma')
            fig.colorbar(i, ax=axs[2])
    
    def plot_human_model(self, pos, vit,name='Modèle Humain', plot=True, e = 0):      
      grid= np.zeros([len(pos),len(vit), len(self.driver[0])])  
      for esp_id in range(len(self.driver[0])):
          for y in range(len(vit)):
            for x in range(len(pos)):
                u = self.human_model(pos[x],vit[y], self.driver[0][esp_id][0])
                grid[x][y][esp_id] = self.f([pos[x],vit[y]], [u,1])[1]
                grid[x][y][esp_id] = u
        
      if plot is not True:
           return grid
       
      if  len(self.driver[0])>1:
        fig, axs = plt.subplots(1, len(self.driver[0]))
        fig.suptitle('Modèle humain pour un ' + str(self.drivers_array[self.drivers_ind]) + ' sur une route ' + self.roads_array[self.roads_ind])
        plt.ion()
        xname = 'Position (m)'
        yname = 'Vitesse (m/s)'
        
        for esp_id in range(len(self.driver[0])): 
            name = str(self.driver[0][esp_id][0]) + 's à ' + '{:.2f}'.format(self.driver[0][esp_id][1]*100)+ '%'
            axs[esp_id].set_title(name)
            ci = axs[esp_id].pcolormesh(pos, vit, grid[:,:,esp_id].T, shading='gouraud',cmap = 'plasma')
            axs[esp_id].axis([self.x_lb[0], self.x_ub[0], self.x_lb[1], self.x_ub[1]])
            axs[esp_id].grid(True)
            
        axs[0].set(xlabel=xname, ylabel=yname)
      else:
        fig, axs = plt.subplots(1, 1)
        plt.ion()
        fig.suptitle(name)
        xname = 'Position (m)'
        yname = 'Vitesse (m/s)'
        axs.set(xlabel=xname, ylabel=yname)
        ci = axs.pcolormesh(pos, vit, grid[:,:,0].T, shading='gouraud',cmap = 'plasma')
        axs.axis([self.x_lb[0], self.x_ub[0], self.x_lb[1], self.x_ub[1]])
        fig.colorbar(ci, ax=axs)
        axs.grid(True)
                    
    def plot_ttc_no_controler(self, grid_sys, use_human = True, worst_e_flag = False, plot_flag = True,dec = None):
        pos = grid_sys.x_level[0]
        vit = grid_sys.x_level[1]
         
        worst_e = 0
        if worst_e_flag:
            for di in self.driver[0]:
                if di[0]> worst_e:
                    worst_e = di[0]
        
        ttc_res = np.zeros([len(pos),len(vit)])
        
        for p in range(len(pos)):
            for v in range(len(vit)):  
                px = -0 - pos[p]
                vx = 0 - vit[v]
                if use_human:
                    dx = self.f([pos[p],vit[v]],[0,0], e = worst_e)
                else:
                    dx = self.f([pos[p],vit[v]],[0,1], e = worst_e)
                    
                if dec == None:  
                    ax = 0-dx[1]
                else:
                    ax = 0-(dec)
                    
                if ax == 0:
                    if vx == 0:
                        ttc = np.nan
                    else:
                        ttc = -px/vx
                else:
                    if ((vx**2)-(2*px*ax)) <= 0.0:
                        ttc = np.nan
                    else:
                         ttc_minus = -(vx/ax) - (np.sqrt((vx**2)-(2*px*ax))/ax)
                         ttc_plus = -(vx/ax) + (np.sqrt((vx**2)-(2*px*ax))/ax)
                         
                         if ttc_minus <= ttc_plus:
                              ttc = ttc_minus
                         else:
                              ttc = ttc_plus
                ttc_res[p,v] = ttc
                
        ttc_res = np.clip(ttc_res,0,5)
        if plot_flag:
            fig, axs = plt.subplots(1,1)
            axs.set_title('Temps de collision pour une décélération fixe: ' + str(-ax))
            plt.ion()
            xname = 'Position (m)'
            yname = 'Vitesse (m/s)'
            axs.set(xlabel=xname, ylabel=yname)
            i = axs.pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], ttc_res.T, shading='gouraud', cmap = 'plasma')
            axs.axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
            fig.colorbar(i, ax=axs)
            axs.grid(True)
                
        return ttc_res
    
    def plot_treshhold_no_controller(self, grid_sys, use_human = True, worst_e_flag = False, plot_flag = True,dec = None):
        pos = grid_sys.x_level[0]
        vit = grid_sys.x_level[1]
         
        worst_e = 0
        if worst_e_flag:
            for di in self.driver[0]:
                if di[0]> worst_e:
                    worst_e = di[0]
        
        treshhold_res = np.zeros([len(pos),len(vit)])
        
        for p in range(len(pos)):
            for v in range(len(vit)):  
                px = 0 - pos[p]
                vx = 0 - vit[v]
                
                if use_human:
                    dx = self.f([pos[p],vit[v]],[0,0], e = worst_e)
                else:
                    dx = self.f([pos[p],vit[v]],[0,1], e = worst_e)
                
                if dec == None:  
                    ax = 0-dx[1]
                else:
                    ax = 0-(dec)
                

                ax_max = 0 - self.dmax[1]
                ax_max = 0 - (-8.5)
                axb = 0 - self.dmax[1]
                
                treshhold_res[p,v] =  -(vx/axb) + (np.sqrt((vx**2)-((-ax)*px*ax_max))/(ax_max))
                treshhold_res[p,v] =  -(vx/axb) + (np.sqrt((vx**2)-((-ax)*px*axb))/(axb))
                
        if plot_flag:
            fig, axs = plt.subplots(1,1)
            axs.set_title('Temps de collision pour un conducteur: ' + self.driver[1])
            plt.ion()
            xname = 'Position (m)'
            yname = 'Vitesse (m/s)'
            axs.set(xlabel=xname, ylabel=yname)
            i = axs.pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], treshhold_res.T, shading='gouraud', cmap = 'plasma')
            axs.axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
            fig.colorbar(i, ax=axs)
            axs.grid(True)
                
        return treshhold_res
    
    def plot_ttc_response(self, grid_sys, use_human = True, worst_e_flag = False, plot_flag = True, dec = None):
        pos = grid_sys.x_level[0]
        vit = grid_sys.x_level[1]
         
        worst_e = 0
        if worst_e_flag:
            for di in self.driver[0]:
                if di[0]> worst_e:
                    worst_e = di[0]
        
        ttc_res = np.zeros([len(pos),len(vit)])
        treshhold_res = np.zeros([len(pos),len(vit)])
        u_res = np.zeros([len(pos),len(vit)])
        
        ttc_res =  self.plot_ttc_no_controler(grid_sys, use_human = use_human, worst_e_flag=worst_e_flag, plot_flag = False, dec = dec)
        treshhold_res =  self.plot_treshhold_no_controller(grid_sys, use_human = use_human, worst_e_flag=worst_e_flag, plot_flag = False, dec = dec)
        
        for p in range(len(pos)):
            for v in range(len(vit)):  
                if ttc_res[p,v] < treshhold_res[p,v]:
                    u_res[p,v] = self.best_slip
                else:
                    u_res[p,v] = 0
                

                
        if plot_flag:
            fig, axs = plt.subplots(1,3)
            axs[0].set_title('Temps de collision pour un conducteur: ' + self.driver[1])
            axs[1].set_title('Seuil de temps à respecter')
            axs[2].set_title('Commande du Contrôleur TTC')
            plt.ion()
            xname = 'Position (m)'
            yname = 'Vitesse (m/s)'
            axs[0].set(xlabel=xname, ylabel=yname)
            axs[1].set(xlabel=xname, ylabel=yname)
            axs[2].set(xlabel=xname, ylabel=yname)
            
            i0 = axs[0].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], ttc_res.T, shading='gouraud', cmap = 'plasma')
            i1 = axs[1].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], treshhold_res.T, shading='gouraud', cmap = 'plasma')
            i2 = axs[2].pcolormesh(grid_sys.x_level[0], grid_sys.x_level[1], u_res.T, shading='gouraud', cmap = 'plasma')
            
            axs[0].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
            axs[1].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
            axs[2].axis([grid_sys.x_level[0][0], grid_sys.x_level[0][-1], grid_sys.x_level[1][0], grid_sys.x_level[1][-1]])
            
            fig.colorbar(i0, ax=axs[0])
            fig.colorbar(i1, ax=axs[1])
            fig.colorbar(i2, ax=axs[2])
            
            axs[0].grid(True)
            axs[1].grid(True)
            axs[2].grid(True)
                
        return u_res    

    def plot_model_prob(self):
        fig, ax = plt.subplots()
        fruits = ['-0.6', '-0.3', '0', '+0.3', '+0.6']
        counts = [0,0,0,0,0]
        
        for esp in self.driver[0]:
            if esp[0] == -0.6:
                counts[0] = esp[1]
            if esp[0] == -0.3:
                counts[1] = esp[1]
            if esp[0] == 0.0:
                counts[2] = esp[1]
            if esp[0] == +0.3:
                counts[3] = esp[1]
            if esp[0] == +0.6:
                counts[4] = esp[1]
        print(counts)
                
        bar_labels = ['-0.6', '-0.3', '0', '+0.3', '+0.6']
        ax.bar(fruits, counts, label=bar_labels)
        ax.set_ylabel('Probabilité')
        ax.set_xlabel('Différence du temps de réaction')
        ax.set_title('Espérence du conducteur')
        
        plt.show()

class LongitudinalFrontWheelDriveCarWithDriverModel_withjerk( system.ContinuousDynamicSystem ):    
    ############################
    def __init__(self):
        """ """
    
        # Dimensions
        self.n = 3 
        self.m = 2   
        self.p = 3
        # initialize standard params
        system.ContinuousDynamicSystem.__init__(self, self.n, self.m, self.p)
        
        # Labels
        self.name = 'Front Wheel Drive Car JERK'
        self.state_label = ['x','dx','ddx']
        self.input_label = ['slip','override']
        self.output_label = ['x','dx', 'ddx']
        
        # Units
        self.state_units = ['[m]','[m/sec]', '[m/sec2]']
        self.input_units = ['[]','[]']
        self.output_units = ['[m]','[m/sec]', '[m/sec2]']

        self.x_ub = np.array([+100, 20, 0])
        self.x_lb = np.array([0, 0, -10.9])
        self.u_ub = np.array([0.0, 1])
        self.u_lb = np.array([-0.3, 0])
        self.x_grid = np.array([200,200,30])
        
        # Model param
        self.lenght  = 3.35          # distance between front wheel and back wheel [m]
        self.xc      = self.lenght/2          # distance from back wheel to c.g.  [m]
        self.yc      = 1.74/2        # height from ground to c.g.  [m]
        self.mass    = 760       # total car mass [kg]
        self.gravity = 9.81       # gravity constant [N/kg]
        self.rho     = 1.225      # air density [kg/m3]
        self.cdA     = (1.84*1.74)    # drag coef time area [m2]
        self.u_dim = [20]
        
        # Ground traction curve parameters
        self.roads = {
            'AsphalteDry': [1.2801, 23.99, 0.52, 0.02, "AsphalteDry"],
            'CementDry': [1.1973, 25.168, 0.5373, 0.02, "CementDry"],
            'AsphalteWet': [0.857, 33.822, 0.347, 0.02, "AsphalteWet"],
            'CobblestoneDry': [1.37, 6.46, 0.67, 0.02, "CobblestoneDry"],
            'CobblestoneWet': [0.400, 33.71, 0.12, 0.02, "CobblestoneWet"],
            'Snow': [0.1946, 94.129, 0.0646, 0.02, "Snow"],
            'Ice': [0.05, 306.39, 0, 0.02, "Ice"], 
            } 
                
        self.road = self.roads['AsphalteWet']
        
        
        self.timing = 0.5
        self.timing_conservateur = -1*self.timing
        self.timing_normal = +0.0
        self.timing_aggressif = self.timing
        self.timing_sleep = +100.0
        
        self.drivers = {
            'Bad': [[[self.timing_conservateur, 0.05], [self.timing_normal, 0.20], [self.timing_aggressif, 0.75]], "bad"],
            'Normal': [[[self.timing_conservateur, 0.25], [self.timing_normal, 0.50], [self.timing_aggressif, 0.25]], "normal"],
            'Good': [[[self.timing_conservateur, 0.85], [self.timing_normal, 0.10], [self.timing_aggressif, 0.05]], "good"],
            'Sleepy': [[[self.timing_normal, 0.01], [self.timing_sleep, 0.99]], "sleepy"],
            'Null': [[[+0.0, 1.0]], "null"]
            } 

        
        # Driver models
        self.flag_human_model = False
        self.dmax = np.nan
        
        
        self.driver_type = None
        self.driver = self.drivers['Null']
        self.use_human_model = True
        # Graphic output parameters 
        self.dynamic_domain  = False
        self.dynamic_range   = self.lenght * 2
        
        self.tm = 3.5
        self.tf = 1.75
        self.tm_dot = 0.75
        self.tm_coef = 0.6
        self.mu_coef = 1
        
        # Animation output graphical parameters
        self.linestyle = '-'
        self.obs_dist =  self.x_ub[0] + self.lenght * 2 # using the upper bound on x range
        self.obs_size = 2
        self.best_slip = -0.2
        
        
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
        
        mu = -1*((self.road[0]*(1 - np.exp(-1*self.road[1]*slip)))-(self.road[2]*slip))*np.exp(-1*self.road[3]*slip*v)
        
        if slip < 0: 
            slip = np.abs(slip)
            mu = -1*((self.road[0]*(1 - np.exp(-1*self.road[1]*slip)))-(self.road[2]*slip))
        else:
            slip = np.abs(slip)
            mu = ((self.road[0]*(1 - np.exp(-1*self.road[1]*slip)))-(self.road[2]*slip))
        
        return mu
    
    def return_max_mu(self):
        slip_min = -0.3 
        slip_max = 0.
        slips = np.arange(slip_min, slip_max, 0.001)
        force = np.zeros(len(slips))
        for slip in range(len(slips)):
            force[slip] = self.slip2force(slips[slip], self.x_ub[1])
        return [np.min(force), slips[np.argmin(force)]]

    #############################
    def f(self, x , u, t = 0, e = 0):
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
        
        ### INPUT
        slip = u[0]
        override = u[1]
        
        ### STATE
        p_0 = x[0]
        v_0 = x[1]
        a_0 = x[2]
        
        if self.use_human_model == True:
            if override != 1.:
                slip = self.human_model(p_0, v_0)
        mu = self.slip2force(slip, v_0) 

        # constant params local vairables
        ry, rr, rf = self.compute_ratios() 
        m    = self.mass 
        g    = self.gravity
        rcda = self.rho * self.cdA
        
        # Drag froce
        fd = 0.5 * rcda * v_0 * np.abs( v_0 ) # drag froce with the right sign
        a  = (mu * m * g * rr - (fd*((mu*ry)+1)))/( m * (1 + mu * ry ))
      
        ###################
        dx[0]  = v_0 # velocity
        dx[1]  = a # acc
        dx[2]  = (a - a_0)/0.2 # jerk
        
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

    def human_model(self, pos, vit, e = 0):
        security_distance = 1
        vx = 0-vit
        px = -security_distance-pos
        a_max = (self.dmax[1]*self.tm_coef)
        ax_max = 0-a_max
        
        #DECELERATION VOULUE
        if ax_max == 0:
            tp = 1    
        else:
            tp = (-px/vx) / (-vx/ax_max)
        D = (2*(1+self.tm_dot)*(1+(2*self.tm_dot*tp))) 
        if D<0:
            D=0
        D = D**(-(2+(1/self.tm_dot)))
        a_desired = -D*ax_max
        a = np.clip(a_desired, a_max, 0.) 
        slip = -1*a/a_max
        ax = 0-a
        
        if ax == 0:
            tm = (-px/vx)+self.tf
        else:
            tm =-(vx/(ax_max))+self.tf
            
           
        tm =-(vx/(ax_max))+self.tf
        ttc = -(px)/(vx+0.0000000000001)
        
        if pos > -security_distance:
             return -1
        elif ttc < (tm-(self.tf+e)):
             return slip  
        else:
             return 0
    
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx] 
    

    
     