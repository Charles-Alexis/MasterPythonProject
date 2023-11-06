#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 20:48:32 2022

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

from scipy.interpolate import RectBivariateSpline as interpol2D
from scipy.interpolate import RegularGridInterpolator as rgi



from pyro.control  import controller


###############################################################################
### DP controllers
###############################################################################

class LookUpTableController( controller.StaticController ):

    ############################
    def __init__(self, grid_sys , pi ):
        """
        Pyro controller based on a discretized lookpup table of control inputs

        Parameters
        ----------
        grid_sys : pyro GridDynamicSystem class
            A discretized dynamic system
        pi : numpy array, dim =  self.grid_sys.nodes_n , dtype = int
            A list of action index for each node id
        """
        
        if grid_sys.nodes_n != pi.size:
            raise ValueError("Grid size does not match optimal action table size")
        
        k = 2                   # Ref signal dim
        m = grid_sys.sys.m      # control input signal dim
        p = grid_sys.sys.n      # output signal dim (state feedback)
        
        super().__init__(k, m, p)
        
        # Grid sys
        self.grid_sys = grid_sys
        
        # Table of actions
        self.pi = pi
        
        # Label
        self.name = 'Tabular Controller'
        
        # Interpolation Options
        self.interpol_method = []
        
        for k in range(self.m):
            
            # options can be changed for each control input axis
            self.interpol_method.append('linear') # "linear”, “nearest”, “slinear”, “cubic”, and “quintic”
            
        self.compute_interpol_functions()
        
    
    #############################
    def compute_interpol_functions( self  ):
        """  """
        
        self.u_interpol = [] 
        
        for k in range(self.m):
            
            u_k      = self.grid_sys.get_input_from_policy( self.pi, k)
            self.u_interpol.append( self.grid_sys.compute_interpolation_function( u_k , 
                                                                                 self.interpol_method[k],
                                                                                 bounds_error = False   , 
                                                                                 fill_value = 0  ) )

        
    
    #############################
    def lookup_table_selection( self , x ):
        """  select the optimal u given actual cost map """
        
        u = np.zeros( self.m )
        
        for k in range(self.m):
            
            u[k] = self.u_interpol[k]( x )
            
        return u
    

    #############################
    def c( self , y , r , t = 0 ):
        """  State feedback (y=x) - no reference - time independent """
        
        x = y
        
        u = self.lookup_table_selection( x )
        
        return u
    
    

###############################################################################
### DP Algo
###############################################################################

class DynamicProgramming:
    """ Dynamic programming on a grid sys """
    
    ############################
    def __init__(self, grid_sys , cost_function , final_time = 0, esperance = [[0,1]], cf_flag = False):
        
        # Dynamic system
        self.grid_sys  = grid_sys         # Discretized Dynamic system class
        self.sys       = grid_sys.sys     # Base Dynamic system class
        self.nbr_iteration_to_do = 10000
        
        # Cost function
        self.cf  = cost_function
        self.tf  = final_time
        # Use cf params
        self.cf_flag = cf_flag
        # Options
        self.alpha                = 1.0 # facteur d'oubli exponentiel
        self.interpol_method      ='linear' # "linear”, “nearest”, “slinear”, “cubic”, and “quintic”
        self.save_time_history    = False
        
        # Memory Variables
        self.t = self.tf   # time of the computed step (useful for time-varying system)
        self.k = 0         # Number of computed steps
        
        # Esperance
        self.esperance = esperance
        
        # Start time (needed to plot elapsed computation time)
        self.start_time = time.time()
        
        # Final cost
        self.evaluate_terminal_cost()
        
        self.last_dmax = -1
        self.t_nbr = 0
        
        # Esperance
        self.E = [0]
        self.cleared_data = None
        
        # 
        if self.save_time_history:
            self.t_list  = []
            self.J_list  = []
            if self.cf_flag:
                self.J_confort_list  = []
                self.J_override_list  = []
                self.J_security_list  = []
            self.pi_list = []
            
            # Value at t = t_f
            self.J_list.append(  self.J  )
            if self.cf_flag:
                self.J_confort_list.append(  self.J_confort )
                self.J_override_list.append(  self.J_override  )
                self.J_security_list.append(  self.J_security  )
            self.t_list.append(  self.tf )
            self.pi_list.append( self.pi )
        
    def clear_bad_data(self):
        debug_array = self.grid_sys.get_grid_from_array(self.grid_sys.get_input_from_policy(self.pi, 0))
        brake_starting = np.zeros(len(debug_array[:,0]))
        for posInd in range(len(brake_starting)):
            index = 0
            flag = False
            for x in range(len(debug_array[posInd,:])):
                flipped = np.flip(debug_array[posInd,:])
                if flipped[x] == 0 and flag == False:
                    flag = True
                    index = (self.grid_sys.x_grid_dim[1]-1)-x
            brake_starting[posInd] = index
        for bInd in range(len(brake_starting)):
            for xInd in range(len(debug_array[bInd,:])):
                if xInd <= brake_starting[bInd]:
                    debug_array[bInd,xInd] =0
        self.cleared_data = debug_array
        
        
    ##############################
    def evaluate_terminal_cost(self):
        """ initialize cost-to-go and policy """

        self.J  = np.zeros( self.grid_sys.nodes_n , dtype = float )
        if self.cf_flag:
            self.J_confort  = np.zeros( self.grid_sys.nodes_n , dtype = float )
            self.J_override  = np.zeros( self.grid_sys.nodes_n , dtype = float )
            self.J_security  = np.zeros( self.grid_sys.nodes_n , dtype = float )
        self.pi = np.zeros( self.grid_sys.nodes_n , dtype = int   )

        # Initial cost-to-go evaluation       
        for s in range( self.grid_sys.nodes_n ):  
            
                xf = self.grid_sys.state_from_node_id[ s , : ]
                
                # Final Cost of all states
                self.J[ s ] = self.cf.h( xf , self.tf )
                if self.cf_flag:
                    self.J_confort[ s ] = self.cf.h( xf , self.tf )
                    self.J_override[ s ] = self.cf.h( xf , self.tf )
                    self.J_security[ s ] = self.cf.h( xf , self.tf )
                
    
    ###############################
    def initialize_backward_step(self):
        """ One step of value iteration """
        
        # Update values
        self.k      = self.k + 1                  # index backward in time
        self.t      = self.t - self.grid_sys.dt   # time
        self.J_next = self.J
        if self.cf_flag:
            self.J_next_confort = self.J_confort
            self.J_next_override = self.J_override
            self.J_next_security = self.J_security
        
        # New Cost-to-go and policy array to be computed
        self.J  = np.zeros( self.grid_sys.nodes_n , dtype = float )
        if self.cf_flag:
            self.J_confort  = np.zeros( self.grid_sys.nodes_n , dtype = float )
            self.J_override  = np.zeros( self.grid_sys.nodes_n , dtype = float )
            self.J_security  = np.zeros( self.grid_sys.nodes_n , dtype = float )
        self.pi = np.zeros( self.grid_sys.nodes_n , dtype = int   )
        
        # Create interpol function
        self.J_interpol = self.grid_sys.compute_interpolation_function(self.J_next, self.interpol_method, bounds_error = False, fill_value = 0)
        if self.cf_flag:
            self.J_interpol_confort = self.grid_sys.compute_interpolation_function(self.J_next_confort, self.interpol_method, bounds_error = False, fill_value = 0)
            self.J_interpol_override = self.grid_sys.compute_interpolation_function(self.J_next_override, self.interpol_method, bounds_error = False, fill_value = 0)
            self.J_interpol_security = self.grid_sys.compute_interpolation_function(self.J_next_security, self.interpol_method, bounds_error = False, fill_value = 0)
                        
                
    ###############################
    def compute_backward_step(self):
        """ One step of value iteration """
        # For all state nodes
        print('HELLO PAS CERTAIN')
        for s in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.state_from_node_id[ s , : ]
                
                Q = np.zeros( self.grid_sys.actions_n  ) 
                
                # For all control actions
                for a in range( self.grid_sys.actions_n ):
                    
                    u = self.grid_sys.input_from_action_id[ a , : ]               
                        
                    # If action is in allowable set
                    if self.sys.isavalidinput( x , u ):
                        # Forward dynamics
                        for e in self.E:
                            x_next = self.sys.f( x , u , self.t, e) * self.grid_sys.dt + x
                            # if the next state is not out-of-bound
                            if self.sys.isavalidstate(x_next):
    
                                # Estimated (interpolation) cost to go of arrival x_next state
                                J_next = self.J_interpol( x_next )
                                
                                # Cost-to-go of a given action
                                J = self.cf.g( x , u , self.t ) * self.grid_sys.dt + self.alpha * J_next
                                
                            else:
                                # Out of bound terminal cost
                                J_next = self.J_interpol( x_next )
                                J = self.cf.g( x , u , self.t ) * self.grid_sys.dt + self.alpha * J_next
                            Q[ a ] = Q[ a ] + J
                        Q[ a ] = Q[ a ] / len(self.E)
                            
                    else:
                        
                        # Invalide control input at this state
                        Q[ a ] = self.cf.INF
                        
                self.J[ s ]  = Q.min()
                self.pi[ s ] = Q.argmin()
                    
    
    ###############################
    def finalize_backward_step(self):
        """ One step of value iteration """
        
        # Computation time
        elapsed_time = time.time() - self.start_time
        
        # Convergence check        
        delta     = self.J - self.J_next
        j_max     = self.J.max()
        delta_max = delta.max()
        delta_min = delta.min()
        
        #if self.k%20 == 0 or self.k==1:
        print('%d t:%.2f Elasped:%.4f max: %.4f dmax:%.4f dmin:%.4f' % (self.k,self.t,elapsed_time,j_max,delta_max,delta_min) )
        
        # List in memory
        if self.save_time_history:
            self.J_list.append( self.J  )
            if self.cf_flag:
                self.J_confort_list.append(  self.J_confort )
                self.J_override_list.append(  self.J_override  )
                self.J_security_list.append(  self.J_security  )
            
            self.t_list.append( self.t )
            self.pi_list.append( self.pi )
            
        # return largest J change for usage as stoping criteria
        return abs(np.array([delta_max,delta_min])).max() 

    ###############################
    def finalize_backward_step_pourcent(self):
        """ One step of value iteration """
        
        # Computation time
        elapsed_time = time.time() - self.start_time
        computing_elapsed_time = time.time() - self.computing_start_time
        # Convergence check        
        delta     = self.J - self.J_next
        j_max     = self.J.max()
        delta_max = delta.max()
        delta_min = delta.min()
        #delta_min_none_0 = np.min(delta[np.nonzero(delta)])
        delta_min_none_0 = 0
        delta_mean = np.sum(delta)/np.size(delta)
        
        if self.k%100 == 0 or self.k==1:
            size = self.nbr_iteration_to_do
            approx_time = ((computing_elapsed_time * size) /self.k) - computing_elapsed_time
            print('%d t:%.4f Elasped:%.4f max: %.2f dmax:%.4f dmin:%.4f' % (self.k,self.t,computing_elapsed_time,j_max,delta_max,delta_min) )
            print('none 0 min: %.4f Mean: %.4f and approx time to end:%.4f'%(delta_min_none_0,delta_mean,approx_time))
        
        # List in memory
        if self.save_time_history:
            self.J_list.append( self.J  )
            if self.cf_flag:
                self.J_confort_list.append(  self.J_confort )
                self.J_override_list.append(  self.J_override  )
                self.J_security_list.append(  self.J_security  )
            self.t_list.append( self.t )
            self.pi_list.append( self.pi )
            
        # return largest J change for usage as stoping criteria
        
        return [delta_max, delta_min, delta_min_none_0, delta_mean]
        return abs(np.array([delta_max,delta_min])).max()#/j_max
    
    ################################
    def compute_steps(self, n = 50 , animate_cost2go = False , animate_policy = False , k = 0, treshhold = 0.01, jmax = 5):
        """ compute number of step """
        ## PLOTTING
        self.nbr_iteration_to_do= n
        if self.cf_flag:
            if animate_cost2go: self.plot_cost2gos(jmax=5)
        else:
            if animate_cost2go: self.plot_cost2go(jmax=5)
        if animate_policy: self.plot_policy( k )
        self.computing_start_time = time.time()
        
        
        flag = False
        for i in range(n):
            if flag is not True:
                ## THIS CLASS
                self.initialize_backward_step()
                
                ## OTHER CLASS
                self.compute_backward_step()
                
                ## THIS CLASS
                delta = self.finalize_backward_step_pourcent()
                dmax = delta[0]
                ## STOPPING CODE
                if dmax < treshhold: #and dmax < 1.5:
                    self.t_nbr = self.t_nbr + 1
                    if self.t_nbr >= 50:    
                        flag = True
                else:
                    self.t_nbr = 0      
                self.last_dmax = dmax
                
                ## PLOTTING
                if self.cf_flag:
                    if animate_cost2go: self.update_cost2gos_plot()
                else:
                    if animate_cost2go: self.update_cost2go_plot()
                if animate_policy: self.update_policy_plot( k )
            
    
    ################################
    def solve_bellman_equation(self, tol = 0.1 , animate_cost2go = False , animate_policy = False , k = 0 ):
        """ 
        Value iteration algorithm
        --------------------------
        
        Do Dp backward iterations until changes to J are under the tolerance 
        
        Note: self.alpha should be smaller then 1 to garantee convergence
        
        
        """
        
        print('\nComputing backward DP iterations until dJ<%2.2f:'%tol)
        print('---------------------------------------------------------')
        
        if animate_cost2go: self.plot_cost2go()
        if animate_policy: self.plot_policy( k )
        
        delta = self.cf.INF
        
        while (delta>tol):
            self.initialize_backward_step()
            self.compute_backward_step()
            delta = self.finalize_backward_step()
            if animate_cost2go: self.update_cost2go_plot()
            if animate_policy: self.update_policy_plot( k )
            
        print('Bellman equation solved!' )
        
        
    ################################
    ### Data tools
    ################################
    
    ################################
    def clean_infeasible_set(self , tol = 1):
        """
        Set default policy and cost2go to cf.INF for state for  which it is unavoidable
        that they will reach unallowable regions

        """
        
        default_action = self.grid_sys.get_nearest_action_id_from_input( self.sys.ubar )
        
        infeasible_node_IDs = self.J > ( self.cf.INF - tol )
        
        self.J[  infeasible_node_IDs ] = self.cf.INF
        self.pi[ infeasible_node_IDs ] = default_action
        
        
    ################################
    ### Print quick shorcuts
    ################################
            
            
    ################################
    def plot_cost2go(self , jmax = None , i = 0 , j = 1 , show = True ):
        
        if jmax == None: jmax = np.max(self.J)
               
        fig, ax, pcm = self.grid_sys.plot_grid_value( self.J , 'Cost-to-go' , i , j , jmax , 0 )
        
        text = ax.text(0.05, 0.05, '', transform=ax.transAxes, fontsize = 8 )
        
        self.cost2go_fig = [fig, ax, pcm, text]
        
        if show: plt.pause( 0.001 )
        #plt.ion()

    ################################
    def plot_cost2gos(self , jmax = None , i = 0 , j = 1 , show = True ):
        
        if jmax != None: 
            J = np.clip(self.J,0,jmax)
            J_confort = np.clip(self.J_confort,0,jmax)
            J_override = np.clip(self.J_override,0,jmax)
            J_security = np.clip(self.J_security,0,jmax)
        else:
            J = self.J
            J_confort = self.J_confort
            J_override = self.J_override
            J_security = self.J_security          
               
        fig, ax, pcm, pcm_confort, pcm_override, pcm_security = self.grid_sys.plot_grid_values( J , J_confort , J_override , J_security ,'Cost-to-go' , i , j , jmax , 0 )
        fig.colorbar(pcm, ax=ax[0])
        fig.colorbar(pcm_confort, ax=ax[1])
        fig.colorbar(pcm_override, ax=ax[2])
        fig.colorbar(pcm_security, ax=ax[3])
        
        text = ax[0].text(0.05, 0.05, '', transform=ax[0].transAxes, fontsize = 8 )
        
        self.cost2go_fig = [fig, ax, pcm, pcm_confort, pcm_override, pcm_security, text]
        
        if show: plt.pause( 0.001 )
        #plt.ion()
        
    ################################
    def plot_everything(self , jmax = None , i = 0 , j = 1 , show = True ):
        
        if jmax == None: jmax = np.max(self.J)
               
        fig, ax, picm, pcm, pcm_confort, pcm_override, pcm_security = self.grid_sys.plot_every_values( self.pi, self.J , self.J_confort,
                                                                                                      self.J_override , self.J_security,
                                                                                                      'Cost-to-go' , i , j , jmax , 0 )
        fig.colorbar(picm, ax=ax[0])
        fig.colorbar(pcm, ax=ax[1])
        fig.colorbar(pcm_confort, ax=ax[2])
        fig.colorbar(pcm_override, ax=ax[3])
        fig.colorbar(pcm_security, ax=ax[4])
        
        ax[0].set_title('Policy')
        ax[1].set_title('Total')
        ax[2].set_title('Confort: '+str(self.cf.confort_coef))
        ax[3].set_title('Override: '+str(self.cf.override_coef))
        ax[4].set_title('Security: '+str(self.cf.security_coef))
        
        text = ax[0].text(0.05, 0.05, '', transform=ax[0].transAxes, fontsize = 8 )
        
        self.cost2go_fig = [fig, ax, picm, pcm, pcm_confort, pcm_override, pcm_security, text]
        
        if show: plt.pause( 0.001 )
        #plt.ion()
             
    ################################
    def update_cost2go_plot(self, i = 0 , j = 1 , show = True ):
        
        J_grid = self.grid_sys.get_grid_from_array( self.J )
        
        J_2d = self.grid_sys.get_2D_slice_of_grid( J_grid , i , j )
               
        self.cost2go_fig[2].set_array( np.ravel( J_2d.T ) )
        self.cost2go_fig[3].set_text('Optimal cost2go at time = %4.2f' % ( self.t ))
        
        if show: plt.pause( 0.001 )
             
    ################################
    def update_cost2gos_plot(self, i = 0 , j = 1 , show = True ):
        
        J_grid = self.grid_sys.get_grid_from_array( self.J )
        J_grid_confort = self.grid_sys.get_grid_from_array( self.J_confort )
        J_grid_override = self.grid_sys.get_grid_from_array( self.J_override )
        J_grid_security = self.grid_sys.get_grid_from_array( self.J_security )
        
        J_2d = self.grid_sys.get_2D_slice_of_grid( J_grid , i , j )
        J_2d_confort = self.grid_sys.get_2D_slice_of_grid( J_grid_confort , i , j )
        J_2d_override = self.grid_sys.get_2D_slice_of_grid( J_grid_override , i , j )
        J_2d_security = self.grid_sys.get_2D_slice_of_grid( J_grid_security , i , j )

        self.cost2go_fig[2].set_array( np.ravel( J_2d.T ) )
        self.cost2go_fig[3].set_array( np.ravel( J_2d_confort.T ) )
        self.cost2go_fig[4].set_array( np.ravel( J_2d_override.T ) )
        self.cost2go_fig[5].set_array( np.ravel( J_2d_security.T ) )
        self.cost2go_fig[6].set_text('Optimal cost2go at time = %4.2f' % ( self.t ))
        
        if show: plt.pause( 0.001 )
        
    ################################
    def update_every_plot(self, i = 0 , j = 1 , show = True ):
        
        pi_grid = self.grid_sys.get_grid_from_array(self.grid_sys.get_input_from_policy( self.pi, k=0 ))
        J_grid = self.grid_sys.get_grid_from_array( self.J )
        J_grid_confort = self.grid_sys.get_grid_from_array( self.J_confort )
        J_grid_override = self.grid_sys.get_grid_from_array( self.J_override )
        J_grid_security = self.grid_sys.get_grid_from_array( self.J_security )
        
        pi_2d = self.grid_sys.get_2D_slice_of_grid( pi_grid , i , j )
        J_2d = self.grid_sys.get_2D_slice_of_grid( J_grid , i , j )
        J_2d_confort = self.grid_sys.get_2D_slice_of_grid( J_grid_confort , i , j )
        J_2d_override = self.grid_sys.get_2D_slice_of_grid( J_grid_override , i , j )
        J_2d_security = self.grid_sys.get_2D_slice_of_grid( J_grid_security , i , j )
        
        self.cost2go_fig[2].set_clim(-0.3, 0)
        self.cost2go_fig[3].set_clim(0, np.max(J_2d))
        self.cost2go_fig[4].set_clim(0, np.max(J_2d_confort))
        self.cost2go_fig[5].set_clim(0, np.max(J_2d_override))
        self.cost2go_fig[6].set_clim(0, np.max(J_2d_security))
        
        self.cost2go_fig[2].set_array( np.ravel( pi_2d.T ) )
        self.cost2go_fig[3].set_array( np.ravel( J_2d.T ) )
        self.cost2go_fig[4].set_array( np.ravel( J_2d_confort.T ) )
        self.cost2go_fig[5].set_array( np.ravel( J_2d_override.T ) )
        self.cost2go_fig[6].set_array( np.ravel( J_2d_security.T ) )

        self.cost2go_fig[7].set_text('Optimal cost2go at time = %4.2f' % ( self.t ))
        
        if show: plt.pause( 0.001 )
                
    
    ################################
    def plot_policy(self , k = 0 , i = 0 , j = 1 , show = True ):
               
        fig, ax, pcm = self.grid_sys.plot_control_input_from_policy( self.pi , k)
        
        text = ax.text(0.05, 0.05, '', transform=ax.transAxes, fontsize = 8 )
        
        self.policy_fig = [fig, ax, pcm, text]
        
        if show: plt.pause( 0.001 )
        #plt.ion()
        
        
    ################################
    def update_policy_plot(self, k , i = 0 , j = 1 , show = True  ):
        
        uk    = self.grid_sys.get_input_from_policy( self.pi, k)
        uk_nd = self.grid_sys.get_grid_from_array( uk ) 
        uk_2d = self.grid_sys.get_2D_slice_of_grid( uk_nd , i , j )
               
        self.policy_fig[2].set_array( np.ravel( uk_2d.T ) )
        self.policy_fig[3].set_text('Optimal policy at time = %4.2f' % ( self.t ))
        
        if show: plt.pause( 0.001 )
        
        
    ################################
    def animate_cost2go(self , show = True , save = False , file_name = 'cost2go_animation'):
        
        self.J  = self.J_list[0]
        self.pi = self.pi_list[0]
        self.t  = self.t_list[0]
        self.clean_infeasible_set()
        self.plot_cost2go( show = False  )

        self.ani = animation.FuncAnimation( self.cost2go_fig[0], self.__animate_cost2go, 
                                                len(self.J_list), interval = 20 )
        
        if save: self.ani.save( file_name + '.gif', writer='imagemagick', fps=30)
        
        if show: self.cost2go_fig[0].show()

    ################################
    def animate_cost2gos(self , show = True , save = False , file_name = 'cost2go_animation'):
        
        self.J  = self.J_list[1]
        self.J_confort  = self.J_confort_list[1]
        self.J_override  = self.J_override_list[1]
        self.J_security  = self.J_security_list[1]
        
        self.pi = self.pi_list[0]
        self.t  = self.t_list[0]
        self.clean_infeasible_set()
        self.plot_cost2gos( show = False  )
        self.ani = animation.FuncAnimation( self.cost2go_fig[0], self.__animate_cost2gos, 
                                                len(self.J_list), interval = 20 )
        
        if save: self.ani.save( file_name + '.gif', writer='imagemagick', fps=30)
        
        if show: self.cost2go_fig[0].show()        

    ################################
    def animate_everything(self , show = True , save = False , file_name = 'cost2go_animation'):
        
        self.pi = self.pi_list[1]
        self.J  = self.J_list[1]
        self.J_confort  = self.J_confort_list[1]
        self.J_override  = self.J_override_list[1]
        self.J_security  = self.J_security_list[1]
        
        self.t  = self.t_list[0]
        self.clean_infeasible_set()
        self.plot_everything( show = False  )
        self.ani = animation.FuncAnimation( self.cost2go_fig[0], self.__animate_everything, 
                                                len(self.J_list), interval = 10 )
        
        if save: self.ani.save( file_name + '.gif', writer='imagemagick', fps=30)
        
        if show: self.cost2go_fig[0].show()  
    
    ################################
    def __animate_cost2go(self , i ):
        
        self.J  = self.J_list[i]
        self.pi = self.pi_list[i]
        self.t  = self.t_list[i]
        self.clean_infeasible_set()
        self.update_cost2go_plot( show = False )
        
    ################################
    def __animate_cost2gos(self , i ):
        
        self.J  = self.J_list[i]
        self.J_confort  = self.J_confort_list[i]
        self.J_override  = self.J_override_list[i]
        self.J_security  = self.J_security_list[i]
        self.pi = self.pi_list[i]
        self.t  = self.t_list[i]
        self.clean_infeasible_set()
        self.update_cost2gos_plot( show = False )
        
    ################################
    def __animate_everything(self , i ):
        self.pi = self.pi_list[i]
        self.J  = self.J_list[i]
        self.J_confort  = self.J_confort_list[i]
        self.J_override  = self.J_override_list[i]
        self.J_security  = self.J_security_list[i]
        self.t  = self.t_list[i]
        self.clean_infeasible_set()
        self.update_every_plot( show = False )     
        
    ################################
    def animate_policy(self , show = True , save = False , file_name = 'policy_animation'):
        
        self.J  = self.J_list[1]
        self.pi = self.pi_list[1]
        self.t  = self.t_list[1]
        self.clean_infeasible_set()
        self.plot_policy( k = 0 , show = False )

        self.ani = animation.FuncAnimation( self.policy_fig[0], self.__animate_policy, 
                                                len(self.pi_list)-1, interval = 20 )
        
        if save: self.ani.save( file_name + '.gif', writer='imagemagick', fps=30)
        
        if show: self.policy_fig[0].show()
        
    
    ################################
    def __animate_policy(self , i ):
        
        self.J  = self.J_list[i+1]
        self.pi = self.pi_list[i+1]
        self.t  = self.t_list[i+1]
        self.clean_infeasible_set()
        self.update_policy_plot( k = 0 , show = False )
        
    
    ################################
    ### Quick utility shorcuts
    ################################
    
    ################################
    def get_lookup_table_controller(self):
        """ Create a pyro controller object based on the latest policy """
        
        ctl = LookUpTableController( self.grid_sys, self.pi )
        
        return ctl
        
        
    ################################
    def save_latest(self, name = 'test_data'):
        """ save cost2go and policy of the latest iteration (further back in time) """
        
        np.save(name + '_J_inf', self.J_next)
        np.save(name + '_pi_inf', self.pi.astype(int) )
        
    
    ################################
    def load_J_next(self, name = 'test_data'):
        """ Load J_next from file """
        
        try:

            self.J_next = np.load( name + '_J_inf'   + '.npy' )
            self.pi     = np.load( name + '_pi_inf'  + '.npy' ).astype(int)
            
        except:
            
            print('Failed to load J_next ' )
            

                    
###############################################################################
    
class DynamicProgrammingWithLookUpTable( DynamicProgramming ):
    """ Dynamic programming on a grid sys """
    
    ############################
    def __init__(self, grid_sys , cost_function , final_time = 0, compute_cost = True, esperance = [[0,1]], cf_flag = False):
        print(cf_flag)
        
        DynamicProgramming.__init__(self, grid_sys, cost_function, final_time, esperance, cf_flag)
        if compute_cost:
             self.compute_cost_lookuptable()
    
    
    ###############################
    def compute_cost_lookuptable(self):
        """ One step of value iteration """
        
        start_time = time.time()
        print('Computing g(x,u,t) look-up table..  ')
        
        self.G = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
        
        if self.cf_flag:
            self.G_confort = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
            self.G_override = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
            self.G_security = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )

        # For all state nodes   
        ind = 0
        for s in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.state_from_node_id[ s , : ]
                
                # For all control actions
                for a in range( self.grid_sys.actions_n ):
                    u = self.grid_sys.input_from_action_id[ a , : ] 

                    # If action is in allowable set
                    if self.grid_sys.action_isok[s,a]:
                        # if the next state is not out-of-bound
                        if len(self.esperance) > 1:
                            for esp_id in range(len(self.esperance)):
                                ind = ind + 1
                                if self.cf_flag is False:
                                    self.G[ s , a ] = self.G[ s , a ] + ((self.cf.g( x , u , self.t, e = self.esperance[esp_id][0] ) * self.grid_sys.dt) * self.esperance[esp_id][1])
                                    
                                if (ind % 100000) == 0:
                                    computation_time = time.time() - start_time
                                    size = self.grid_sys.actions_n * self.grid_sys.nodes_n * len(self.grid_sys.esperance)
                                    approx_time = ((computation_time * size)/ind) - computation_time
                                    print('computing glut ' + str(ind) + '/' + str(size) + ' in ' + str(computation_time)
                                          + ' approx time to end = ' + str(approx_time))
            
                        else:
                            ind = ind + 1
                            if self.cf_flag is False:
                                self.G[ s , a ] = self.cf.g( x , u , self.t) * self.grid_sys.dt
                            
                            if (ind % 100000) == 0:
                                computation_time = time.time() - start_time
                                size = self.grid_sys.actions_n * self.grid_sys.nodes_n * len(self.grid_sys.esperance)
                                approx_time = ((computation_time * size)/ind) - computation_time
                                print('computing glut ' + str(ind) + '/' + str(size) + ' in ' + str(computation_time)
                                      + ' approx time to end = ' + str(approx_time))
                    else:
                        # Not allowable input at this state
                        self.G[ s , a ] = self.cf.g( x , u , self.t ) * self.grid_sys.dt
                        print('Input unavailable for state: '+str(x))
                
        # Print update
        computation_time = time.time() - start_time
        print('completed in %4.2f sec'%computation_time)
    
                
    ###############################
    def compute_backward_step_cf_flag(self):
        """ One step of value iteration """
        
        
        if self.cf_flag:
            self.Q       = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
            self.Q_confort       = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
            self.Q_override       = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
            self.Q_security       = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
        else:
            self.Q       = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
        
        if self.cf_flag:
            self.Q_temp  = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n, len(self.esperance) ) , dtype = float )
            self.Q_temp_confort  = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n, len(self.esperance) ) , dtype = float )
            self.Q_temp_override  = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n, len(self.esperance) ) , dtype = float )
            self.Q_temp_security  = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n, len(self.esperance) ) , dtype = float )
        else:
            self.Q_temp  = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n, len(self.esperance) ) , dtype = float )
        
        if self.cf_flag:
            self.Jx_next = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n, len(self.esperance) ) , dtype = float )
            self.Jx_next_confort = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n, len(self.esperance) ) , dtype = float )
            self.Jx_next_override = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n, len(self.esperance) ) , dtype = float )
            self.Jx_next_security = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n, len(self.esperance) ) , dtype = float )
        else: 
            self.Jx_next = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n, len(self.esperance) ) , dtype = float )
        
        for esp_id in range(len(self.esperance)):
            # Computing the J_next of all x_next in the look-up table
            if self.cf_flag:
                self.Jx_next_confort[:,:,esp_id] = self.J_interpol_confort( self.grid_sys.x_next_table[:,:,esp_id,:] )
                self.Jx_next_override[:,:,esp_id] = self.J_interpol_override( self.grid_sys.x_next_table[:,:,esp_id,:] )
                self.Jx_next_security[:,:,esp_id] = self.J_interpol_security( self.grid_sys.x_next_table[:,:,esp_id,:] )
                self.Jx_next[:,:,esp_id] = self.Jx_next_confort[:,:,esp_id]+self.Jx_next_override[:,:,esp_id]+self.Jx_next_security[:,:,esp_id]
            else:
                self.Jx_next[:,:,esp_id] = self.J_interpol( self.grid_sys.x_next_table[:,:,esp_id,:])
            
            if self.cf_flag:
                self.Q_temp_confort[:,:,esp_id] = (self.G_confort + self.alpha * self.Jx_next_confort[:,:,esp_id]) * self.esperance[esp_id][1]
                self.Q_temp_override[:,:,esp_id] = (self.G_override + self.alpha * self.Jx_next_override[:,:,esp_id]) * self.esperance[esp_id][1]
                self.Q_temp_security[:,:,esp_id] = (self.G_security + self.alpha * self.Jx_next_security[:,:,esp_id]) * self.esperance[esp_id][1]
                self.Q_temp[:,:,esp_id] = self.Q_temp_confort[:,:,esp_id]+self.Q_temp_override[:,:,esp_id]+self.Q_temp_security[:,:,esp_id] 
            else:
                self.Q_temp[:,:,esp_id] = (self.G + self.alpha * self.Jx_next[:,:,esp_id]) * self.esperance[esp_id][1]
            
            
            if self.cf_flag:
                self.Q_confort = self.Q_confort + self.Q_temp_confort[:,:,esp_id]
                self.Q_override = self.Q_override + self.Q_temp_override[:,:,esp_id]
                self.Q_security = self.Q_security + self.Q_temp_security[:,:,esp_id]
                self.Q = self.Q_confort + self.Q_override + self.Q_security
            else:
                self.Q = self.Q + self.Q_temp[:,:,esp_id]
            
        
        self.J  = self.Q.min( axis = 1 )
        if self.cf_flag:
            self.J_confort  = np.zeros(np.shape(self.J))
            self.J_override  = np.zeros(np.shape(self.J))
            self.J_security  = np.zeros(np.shape(self.J))
            
        self.pi = self.Q.argmin( axis = 1 )
        if self.cf_flag:
            for minQ in range(len(self.pi)):
                self.J_confort[minQ] = self.Q_confort[minQ][self.pi[minQ]]
                self.J_override[minQ] = self.Q_override[minQ][self.pi[minQ]]
                self.J_security[minQ] = self.Q_security[minQ][self.pi[minQ]]
    
                
    ###############################
    def compute_backward_step(self):
        """ One step of value iteration """
        self.Q       = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
        self.Q_temp  = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n, len(self.esperance) ) , dtype = float )
        self.Jx_next = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n, len(self.esperance) ) , dtype = float )
        
        for esp_id in range(len(self.esperance)):
            # Computing the J_next of all x_next in the look-up table
            self.Jx_next[:,:,esp_id] = self.J_interpol( self.grid_sys.x_next_table[:,:,esp_id,:])
            self.Q_temp[:,:,esp_id] = (self.G + self.alpha * self.Jx_next[:,:,esp_id]) * self.esperance[esp_id][1]
            self.Q = self.Q + self.Q_temp[:,:,esp_id]
                    
        self.J  = self.Q.min( axis = 1 )            
        self.pi = self.Q.argmin( axis = 1 )

                    
                    

###############################################################################

class DynamicProgramming2DRectBivariateSpline( DynamicProgrammingWithLookUpTable ):
    """ Dynamic programming on a grid sys """
    
    ###############################
    def initialize_backward_step(self):
        """ One step of value iteration """
        
        # Update values
        self.k      = self.k + 1                  # index backward in time
        self.t      = self.t - self.grid_sys.dt   # time
        self.J_next = self.J
        
        # New Cost-to-go and policy array to be computed
        self.J  = np.zeros( self.grid_sys.nodes_n , dtype = float )
        self.pi = np.zeros( self.grid_sys.nodes_n , dtype = int   )
        
        # Create interpol function
        self.J_interpol = self.grid_sys.compute_bivariatespline_2D_interpolation_function( self.J_next , kx=3, ky=3)
    
                
    ###############################
    def compute_backward_step(self):
        """ One step of value iteration """
        
        self.Q       = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
        self.Jx_next = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
        
        # Computing the J_next of all x_next in the look-up table
        X            = self.grid_sys.x_next_table
        Jx_next_flat = self.J_interpol( X[:,:,0].flatten() , X[:,:,1].flatten() , grid = False )
        Jx_next      = Jx_next_flat.reshape( (self.grid_sys.nodes_n , self.grid_sys.actions_n ) )
        
        # Matrix version of computing all Q values
        self.Q       = self.G + self.alpha * Jx_next
                        
        self.J  = self.Q.min( axis = 1 )
        self.pi = self.Q.argmin( axis = 1 )
        
        

        
###############################################################################
### Policy Evaluation
###############################################################################

class PolicyEvaluator( DynamicProgramming ):
    """ Evaluate the cost2o of a given control law """

    ############################
    def __init__(self, ctl , grid_sys , cost_function , final_time = 0 ):


        DynamicProgramming.__init__(self, grid_sys, cost_function, final_time )

        self.ctl = ctl

        # Evaluate policy (control law on the grid)


    ###############################
    def compute_backward_step(self):
        """ One step of value iteration """

        # For all state nodes        
        for s in range( self.grid_sys.nodes_n ):  

                x = self.grid_sys.state_from_node_id[ s , : ]

                ######################################
                # Action taken by the controller
                r = self.ctl.rbar
                u = self.ctl.c( x , r , self.t )   
                ######################################

                # If action is in allowable set
                if self.sys.isavalidinput( x , u ):

                    # Forward dynamics 
                    x_next = self.sys.f( x , u , self.t ) * self.grid_sys.dt + x

                    # if the next state is not out-of-bound
                    if self.sys.isavalidstate(x_next):

                        # Estimated (interpolation) cost to go of arrival x_next state
                        J_next = self.J_interpol( x_next )

                        # Cost-to-go of a given action
                        Q = self.cf.g( x , u , self.t ) * self.grid_sys.dt + self.alpha * J_next

                    else:
                        # Out of bound terminal cost
                        Q = self.cf.INF # TODO add option to customize this

                else:
                    # Invalide control input at this state
                    Q = self.cf.INF

                self.J[ s ]  = Q



###############################################################################

class PolicyEvaluatorWithLookUpTable( PolicyEvaluator ):
    """ Evaluate the cost2o of a given control law """

    ############################
    def __init__(self, ctl , grid_sys , cost_function , final_time = 0 ):

        PolicyEvaluator.__init__(self, ctl , grid_sys, cost_function, final_time)

        self.compute_lookuptable()


    ###############################
    def compute_lookuptable(self):
        """ One step of value iteration """

        start_time = time.time()
        print('Computing g(x,u,t) and X-next look-up table..  ', end = '')

        self.x_next_table = np.zeros( ( self.grid_sys.nodes_n , self.sys.n ) , dtype = float ) # lookup table for dynamic
        self.G            = np.zeros(   self.grid_sys.nodes_n                , dtype = float ) # lookup table for cost

        # For all state nodes        
        for s in range( self.grid_sys.nodes_n ):  


                x = self.grid_sys.state_from_node_id[ s , : ]

                ######################################
                # Action taken by the controller
                r = self.ctl.rbar
                u = self.ctl.c( x , r , self.t )   
                ######################################

                # Forward dynamics 
                x_next = self.sys.f( x , u , self.t ) * self.grid_sys.dt + x

                # Save to llokup table
                self.x_next_table[s,:] = x_next

                # If action is in allowable set
                if self.sys.isavalidinput( x , u ):

                    # if the next state is not out-of-bound
                    if self.sys.isavalidstate(x_next):

                        self.G[ s ] = self.cf.g( x , u , self.t ) * self.grid_sys.dt

                    else:
                        #print('Invalid_next state at: %4.2f and %4.2f'% (x[0], x[1]))
                        # Out of bound cost (J_interpol return 0 in this case)
                        self.G[ s ] = self.cf.g( x , u , self.t ) * self.grid_sys.dt

                else:
                    #print('Invalid_input')
                    # Not allowable input at this state
                    self.G[ s ] = self.cf.g( x , u , self.t ) * self.grid_sys.dt

        # Print update
        computation_time = time.time() - start_time
        print('completed in %4.2f sec'%computation_time)


    ###############################
    def compute_backward_step(self):
        """ One step of value iteration """

        self.J       = np.zeros(  self.grid_sys.nodes_n , dtype = float )
        self.Jx_next = np.zeros(  self.grid_sys.nodes_n , dtype = float )

        # Computing the J_next of all x_next in the look-up table
        self.Jx_next = self.J_interpol( self.x_next_table )

        # Matrix version of computing all Q values
        self.J       = self.G + self.alpha * self.Jx_next


class PolicyEvaluatorWithLookUpTable_multiple_coef( PolicyEvaluator ):
    """ Evaluate the cost2o of a given control law """

    ############################
    def __init__(self, ctl , grid_sys , cost_function , final_time = 0 ):

        PolicyEvaluator.__init__(self, ctl , grid_sys, cost_function, final_time)

        self.compute_lookuptable()


    ###############################
    def compute_lookuptable(self):
        """ One step of value iteration """

        start_time = time.time()
        print('Computing g(x,u,t) and X-next look-up table..  ', end = '')

        self.x_next_table = np.zeros( ( self.grid_sys.nodes_n , self.sys.n ) , dtype = float ) # lookup table for dynamic
        self.G            = np.zeros(   self.grid_sys.nodes_n                , dtype = float ) # lookup table for cost
        self.G_security            = np.zeros(   self.grid_sys.nodes_n                , dtype = float ) # lookup table for cost
        self.G_confort            = np.zeros(   self.grid_sys.nodes_n                , dtype = float ) # lookup table for cost
        self.G_override            = np.zeros(   self.grid_sys.nodes_n                , dtype = float ) # lookup table for cost

        # For all state nodes        
        for s in range( self.grid_sys.nodes_n ):  


                x = self.grid_sys.state_from_node_id[ s , : ]

                ######################################
                # Action taken by the controller
                r = self.ctl.rbar
                u = self.ctl.c( x , r , self.t )   
                ######################################

                # Forward dynamics 
                x_next = self.sys.f( x , u , self.t ) * self.grid_sys.dt + x

                # Save to llokup table
                self.x_next_table[s,:] = x_next

                # If action is in allowable set
                if self.sys.isavalidinput( x , u ):

                    # if the next state is not out-of-bound
                    if self.sys.isavalidstate(x_next):

                        self.G[ s ] = self.cf.g( x , u , self.t ) * self.grid_sys.dt
                        self.G_security[ s ] = self.cf.g_security( x , u , self.t ) * self.grid_sys.dt
                        self.G_confort[ s ] = self.cf.g_confort( x , u , self.t ) * self.grid_sys.dt
                        self.G_override[ s ] = self.cf.g_override( x , u , self.t ) * self.grid_sys.dt

                    else:
                        #print('Invalid_next state at: %4.2f and %4.2f'% (x[0], x[1]))
                        # Out of bound cost (J_interpol return 0 in this case)
                        self.G[ s ] = self.cf.g( x , u , self.t ) * self.grid_sys.dt
                        self.G_security[ s ] = self.cf.g_security( x , u , self.t ) * self.grid_sys.dt
                        self.G_confort[ s ] = self.cf.g_confort( x , u , self.t ) * self.grid_sys.dt
                        self.G_override[ s ] = self.cf.g_override( x , u , self.t ) * self.grid_sys.dt

                else:
                    #print('Invalid_input')
                    # Not allowable input at this state
                    self.G[ s ] = self.cf.g( x , u , self.t ) * self.grid_sys.dt
                    self.G_security[ s ] = self.cf.g_security( x , u , self.t ) * self.grid_sys.dt
                    self.G_confort[ s ] = self.cf.g_confort( x , u , self.t ) * self.grid_sys.dt
                    self.G_override[ s ] = self.cf.g_override( x , u , self.t ) * self.grid_sys.dt

        # Print update
        computation_time = time.time() - start_time
        print('completed in %4.2f sec'%computation_time)

    def compute_interpole(self):
        self.J_next = self.J
        self.J_next_confort = self.J_confort
        self.J_next_override = self.J_override
        self.J_next_security = self.J_security
        
        self.J_interpol_coef = self.grid_sys.compute_interpolation_function(self.J_next, self.interpol_method, bounds_error = False, fill_value = 0)
        self.J_interpol_coef_security = self.grid_sys.compute_interpolation_function(self.J_next_security, self.interpol_method, bounds_error = False, fill_value = 0)
        self.J_interpol_coef_confort = self.grid_sys.compute_interpolation_function(self.J_next_confort, self.interpol_method, bounds_error = False, fill_value = 0)
        self.J_interpol_coef_override = self.grid_sys.compute_interpolation_function(self.J_next_override, self.interpol_method, bounds_error = False, fill_value = 0)
        
    ###############################
    def compute_backward_step(self):
        """ One step of value iteration """
        
        self.compute_interpole()
        
        self.J       = np.zeros(  self.grid_sys.nodes_n , dtype = float )
        self.Jx_next = np.zeros(  self.grid_sys.nodes_n , dtype = float )
        self.J_security       = np.zeros(  self.grid_sys.nodes_n , dtype = float )
        self.Jx_next_security = np.zeros(  self.grid_sys.nodes_n , dtype = float )
        self.J_confort       = np.zeros(  self.grid_sys.nodes_n , dtype = float )
        self.Jx_next_confort = np.zeros(  self.grid_sys.nodes_n , dtype = float )
        self.J_override       = np.zeros(  self.grid_sys.nodes_n , dtype = float )
        self.Jx_next_overrride = np.zeros(  self.grid_sys.nodes_n , dtype = float )

        # Matrix version of computing all Q values
        self.J       = self.G + self.alpha * self.Jx_next
        self.J_security       = self.G_security + self.alpha * self.Jx_next_security
        self.J_confort       = self.G_confort + self.alpha * self.Jx_next_confort
        self.J_override       = self.G_override + self.alpha * self.Jx_next_override
    

            
            
            


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    import numpy as np

    from pyro.dynamic  import pendulum
    import discretizer
    import costfunction

    sys  = pendulum.SinglePendulum()

    # Discrete world 
    grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [3] )

    # Cost Function
    qcf = costfunction.QuadraticCostFunction.from_sys(sys)

    qcf.xbar = np.array([ -3.14 , 0 ]) # target
    qcf.INF  = 10000

    # DP algo
    dp = DynamicProgramming( grid_sys, qcf )

    