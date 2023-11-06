#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
###############################################################################
import numpy as np
import time
import matplotlib.pyplot as plt
###############################################################################
import discretizer2
import system 
import costfunction
import BaselineController 
import controller
import simulationv2 as s
import dynamic_programming as dprog
import cost2go2
###############################################################################
from datetime import date
import os
###############################################################################
temps_debut = time.time()

roads = ['AsphalteDry','CementDry','AsphalteWet','CobblestoneDry','CobblestoneWet','Snow','Ice']
drivers = ['Good','Normal','Bad','Sleepy','Null']
comparaison = ['NoDiff', 'Normal','Bool','NearObstacle','HighSpeed']

x_dims = [[600,80],[600,85],[600,100],[600,120],[600,150],[1000,200],[1000,200]]
#x_dims = [[500,80],[500,85],[500,100],[500,120],[500,150],[750,200],[750,200]]
x_dims = [[500,200],[600,85],[500,400],[600,120],[600,150],[1000,200],[1000,200]]
x_lbs = [[-100., 0],[-100., 0],[-100., 0],[-100., 0],[-100., 0],[-150., 0],[-150., 0]]
x_lbs = [[-100., 0],[-100., 0],[-100., 0],[-100., 0],[-100., 0],[-150., 0],[-150., 0]]
tm_arr = [3.40, 3.60, 4.2, 3.8, 8.0, 12,0, 20.0]
tm_arr = [3.40, 3.60, 4.2, 3.8, 7.0, 12,0, 20.0]
controler = ['Vi','Ttc','Msd', 'Human']

## SCÉNARIO
roads_to_test = [2]  
drivers_to_test = [4]
controlers_to_test = [0]
compare_mode = [0]

print('---------------------------------------')
print('Scénario')
print('Roads: ', roads_to_test)
print('Driver: ', drivers_to_test)
print('Controlers: ', controlers_to_test)
print('---------------------------------------')

## CREATING LIST FOR DATA
name_list = list()
grid_sys_list = list()
grid_sys_controller_list = list()
human_list = list()

vi_list = list()
ttc_list = list()
msd_list = list()
human_list = list()

vi_cl_list = list()
ttc_cl_list = list()
msd_cl_list = list()
human_cl_list = list()

vi_c2g_list = list()
ttc_c2g_list = list()
msd_c2g_list = list()
human_c2g_list = list()


#%%
compute_c2g_flag = False
plot_c2g_flag = False
plot_ttc_flag = False
sim_flag = False

if plot_ttc_flag:
    fig, axs = plt.subplots(2, 2)
    plt.ion()

for r in range(len(roads_to_test)):
    for d in range(len(drivers_to_test)): 
        print('---------------------------------------')
        print('Iteration: ' + str(1 + r*len(drivers_to_test) + d))
        print('Roads: ', roads[roads_to_test[r]])
        print('Driver: ', drivers[drivers_to_test[d]])

        u_dim = [75,2]
        ### SYSTEM CONFIGURATION
        sys = system.LongitudinalFrontWheelDriveCarWithDriverModel()
        sys.mass = 760
        sys.lenght = 3.35
        sys.xc = sys.lenght/2
        sys.yc = 1.74/2 
        sys.cdA = 0.3 * (1.84*1.74)
        sys.x_ub = np.array([0 , 20.0])
        sys.x_lb = np.array(x_lbs[roads_to_test[r]])
        sys.u_ub = np.array([0.0, 1])
        sys.u_lb = np.array([-0.3, 0])
        sys.u_dim = u_dim
        sys.m = len(sys.u_dim)
        sys.obs_dist = sys.x_ub[0]
        sys.x_grid = x_dims[roads_to_test[r]]
        sys.road = sys.roads[roads[roads_to_test[r]]]
        sys.timing = 0.75
        sys.driver = sys.drivers[drivers[drivers_to_test[d]]]
        
        
        slip_data = sys.return_max_mu()
        sys.dmax = sys.f([0,sys.x_ub[1]],[slip_data[1],1])
        sys.best_slip = sys.return_max_mu()[1]
        print(sys.road[-1] + ': ' + str(sys.dmax[1]))
        print(sys.road[-1] + ': ' + str(sys.dmax[1]/9.8))
        
        ### DRIVER MODEL CONFIGURATION
        sys.tm = tm_arr[roads_to_test[r]]
        sys.tm_dot = -0.40
        sys.tf = 1.75
        sys.tm_coef = 0.8
        
        ### COST FUNCTION CONFIGURATION
        cf = costfunction.DriverModelCostFunction.from_sys(sys) 
        cf.confort_coef = 0.01
        cf.override_coef = 1
        cf.security_coef = 1
        cf.security_slope = 10
        cf.security_distance = sys.lenght
        sys.cost_function = cf
        cf_list = list((sys.cost_function.g, sys.cost_function.g_confort, sys.cost_function.g_security, sys.cost_function.g_override))
        
        print('dx max: ', sys.dmax)
        print('x dim: ', sys.x_grid)
        print('x0 boundaries: ' + str(sys.x_lb[0]) + ' to ' + str(str(sys.x_ub[0])))
        print('x1 boundaries: ' + str(sys.x_lb[1]) + ' to ' + str(str(sys.x_ub[1])))
        print('Esperance: ', sys.driver[0])
        
        ### GRID SYSTEM CONFIGURATION
        dt = 0.1
        
        controlers_dim = [500,500]
        controlers_dim = x_dims[roads_to_test[r]]
        
        print('Number of states: ',sys.x_grid[0] * sys.x_grid[1])
        print('Number of actions: ', u_dim[0])
        print('Number of actions-states: ', sys.x_grid[0] * sys.x_grid[1] * u_dim[0])
        print('----------------------------------------------------------')
        
        temps_debut_grid_sys = time.time()
        for c2t in controlers_to_test:
            if c2t == 0:
                    grid_sys = discretizer2.GridDynamicSystem(sys, sys.x_grid, u_dim, dt, esperance = sys.driver[0], print_data=True)
                    grid_sys_list.append(grid_sys)
                    
        grid_sys_controller = discretizer2.GridDynamicSystem(sys, controlers_dim, u_dim, dt, esperance = sys.driver[0], print_data=False, lookup = False)
        grid_sys_controller_list.append(grid_sys_controller)
        temps_fin_grid_sys = time.time()
        print('Grid Sys Computing Time: '+ str(temps_fin_grid_sys - temps_debut_grid_sys) + ' secondes' )

        ## SAVING METADATA
        name_list.append('Road: ' + roads[roads_to_test[r]] + '     Driver: ' + drivers[drivers_to_test[d]])
        
        
        ### CONTROLER CONFIGURATION
        for c in range(len(controlers_to_test)): 
            ## VALUE ITERATION
            if controler[controlers_to_test[c]] == 'Vi':
                dp = dprog.DynamicProgrammingWithLookUpTable(grid_sys, cf, esperance = sys.driver[0])
                dp.compute_steps(5000,  treshhold=0.001, animate_policy=False, animate_cost2go=False, jmax = 1000)
                vi_controller = dprog.LookUpTableController( grid_sys , dp.pi )
                vi_controller.k = 2
                cl_sys_vi = controller.ClosedLoopSystem( sys , vi_controller ) 
                cl_sys_vi.cost_function = sys.cost_function
                
                
                if compute_c2g_flag: 
                     c2g_vi = cost2go2.cost2go_list(grid_sys, sys, cl_sys_vi, cf_list)
                     c2g_vi.compute_steps(print_iteration=False)
                     vi_c2g_list.append(c2g_vi)
                
                ## SAVING VI
                vi_list.append(vi_controller)
                vi_cl_list.append(cl_sys_vi)
                
            ## TIME TO COLLISION
            if controler[controlers_to_test[c]] == 'Ttc':
                ttc_controller = BaselineController.TTCController(sys, grid_sys_controller)
                ttc_controller.constant_dec_flag = False
                ttc_controller.constant_dec= -0
                cl_sys_ttc = controller.ClosedLoopSystem( sys , ttc_controller )  
                cl_sys_ttc.cost_function = sys.cost_function
                if compute_c2g_flag: 
                     c2g_ttc = cost2go2.cost2go_list(grid_sys_controller, sys, cl_sys_ttc, cf_list)
                     c2g_ttc.compute_steps(print_iteration=False)
                     ttc_c2g_list.append(c2g_ttc)
                
                ## SAVING TTC
                ttc_list.append(ttc_controller)
                ttc_cl_list.append(cl_sys_ttc)
            
            ## MINIMAL STOPPING DISTANCE    
            if controler[controlers_to_test[c]] == 'Msd':
                msd_controller = BaselineController.MSDController(sys, grid_sys_controller)
                cl_sys_msd = controller.ClosedLoopSystem( sys , msd_controller )  
                cl_sys_msd.cost_function = sys.cost_function
                if compute_c2g_flag: 
                     c2g_msd = cost2go2.cost2go_list(grid_sys_controller, sys, cl_sys_msd, cf_list)
                     c2g_msd.compute_steps(print_iteration=False)
                     msd_c2g_list.append(c2g_msd)
                
                ## SAVING MSD
                msd_list.append(msd_controller)
                msd_cl_list.append(cl_sys_msd)
                
            if controler[controlers_to_test[c]] == 'Human':
                human_controller = BaselineController.humanController(sys, grid_sys_controller)
                cl_sys_human = controller.ClosedLoopSystem( sys , human_controller ) 
                cl_sys_human.cost_function = sys.cost_function
                if compute_c2g_flag: 
                     c2g_human = cost2go2.cost2go_list(grid_sys_controller, sys, cl_sys_human, cf_list)
                     c2g_human.compute_steps(print_iteration=False)
                     human_c2g_list.append(c2g_human)
                
                human_list.append(human_controller)
                human_cl_list.append(cl_sys_human)
                
        if plot_ttc_flag and len(roads_to_test) > 1:
            iteration = r*len(drivers_to_test) + d
            col = int(iteration/2)
            ran = int(iteration%2)
            
            ttc_map = sys.plot_ttc_response(grid_sys_controller, worst_e_flag=True,plot_flag=False)
            i = axs[col][ran].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttc_map.T, shading='gouraud', cmap = 'plasma')
            axs[col][ran].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
            axs[col][ran].set_title('Temps de collision pour une route: ' + sys.road[-1])
            fig.colorbar(i, ax=axs[col][ran])
            axs[col][ran].grid(True)

if plot_ttc_flag:                     
    fig, axs = plt.subplots(1, 3)
    plt.ion()
    
    fig.suptitle(('Contrôleur TTC pour de l\'Asphalte Mouillé (0.7\u03BC)'))
    
    ttc_map = sys.plot_ttc_no_controler(grid_sys_controller, worst_e_flag=True,plot_flag=False)
    ttc_treshhold = sys.plot_treshhold_no_controller(grid_sys_controller, worst_e_flag=True,plot_flag=False)
    ttc_response = sys.plot_ttc_response(grid_sys_controller, worst_e_flag=True,plot_flag=False)
     
    i = axs[0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttc_map.T, shading='gouraud', cmap = 'plasma')
    axs[0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    axs[0].set_title('Temps de Collision')
    fig.colorbar(i, ax=axs[0])
    axs[0].grid(True)
    
    i = axs[1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttc_treshhold.T, shading='gouraud', cmap = 'plasma')
    axs[1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    axs[1].set_title('Temps de Seuil')
    fig.colorbar(i, ax=axs[1])
    axs[1].grid(True)
    
    i = axs[2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttc_response.T, shading='gouraud', cmap = 'plasma')
    axs[2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
    axs[2].set_title('Commande du Contrôleur')
    fig.colorbar(i, ax=axs[2])
    axs[2].grid(True)

#%%
#ttc_1 = sys.plot_ttc_no_controler(grid_sys_controller,use_human=False, dec = -0.0, plot_flag=False) 
#ttc_3 = sys.plot_ttc_no_controler(grid_sys_controller,use_human=False, dec = -3.0, plot_flag=False) 
#ttc_6 = sys.plot_ttc_no_controler(grid_sys_controller,use_human=False, dec = -6.0, plot_flag=False)   
#
#
#fig, axs = plt.subplots(1, 3)
#plt.ion()
#fig.suptitle('Temps de collision pour une route d\'asphalte sec (~1\u03BC)')
#xname = sys.state_label[0] + ' ' + sys.state_units[0]
#yname = sys.state_label[1] + ' ' + sys.state_units[1] 
#
#i = axs[0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttc_1.T, shading='gouraud', cmap = 'plasma')
#axs[0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs[0])
#axs[0].set_ylabel('VITESSE') 
#axs[0].set_xlabel('POSITION') 
#axs[0].set_title('-0m/s^2')
#
#i = axs[1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttc_3.T, shading='gouraud', cmap = 'plasma')
#axs[1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs[1])
#axs[1].set_ylabel('VITESSE') 
#axs[1].set_xlabel('POSITION') 
#axs[1].set_title('-3m/s^2')
#
#i = axs[2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttc_6.T, shading='gouraud', cmap = 'plasma')
#axs[2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs[2])
#axs[2].set_ylabel('VITESSE') 
#axs[2].set_xlabel('POSITION') 
#axs[2].set_title('-6m/s^2')
#
#ttch_1 = sys.plot_treshhold_no_controller(grid_sys_controller,use_human=False, dec = -0.0, plot_flag=False) 
#ttch_3 = sys.plot_treshhold_no_controller(grid_sys_controller,use_human=False, dec = -3.0, plot_flag=False) 
#ttch_6 = sys.plot_treshhold_no_controller(grid_sys_controller,use_human=False, dec = -6.0, plot_flag=False)   
#
#fig, axs = plt.subplots(1, 3)
#plt.ion()
#fig.suptitle('Seuil de temps pour des décélérations fixes pour une route d\'asphalte sec (~1\u03BC)')
#xname = sys.state_label[0] + ' ' + sys.state_units[0]
#yname = sys.state_label[1] + ' ' + sys.state_units[1] 
#
#i = axs[0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttch_1.T, shading='gouraud', cmap = 'plasma')
#axs[0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs[0])
#axs[0].set_ylabel('VITESSE') 
#axs[0].set_xlabel('POSITION') 
#axs[0].set_title('-0m/s^2')
#
#i = axs[1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttch_3.T, shading='gouraud', cmap = 'plasma')
#axs[1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs[1])
#axs[1].set_ylabel('VITESSE') 
#axs[1].set_xlabel('POSITION') 
#axs[1].set_title('-3m/s^2')
#
#i = axs[2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttch_6.T, shading='gouraud', cmap = 'plasma')
#axs[2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs[2])
#axs[2].set_ylabel('VITESSE') 
#axs[2].set_xlabel('POSITION') 
#axs[2].set_title('-6m/s^2')
#
#ttcr_1 = sys.plot_ttc_response(grid_sys_controller,use_human=False, dec = -0.0, plot_flag=False) 
#ttcr_3 = sys.plot_ttc_response(grid_sys_controller,use_human=False, dec = -3.0, plot_flag=False) 
#ttcr_6 = sys.plot_ttc_response(grid_sys_controller,use_human=False, dec = -6.0, plot_flag=False)   
#
#fig, axs = plt.subplots(1, 3)
#plt.ion()
#fig.suptitle('Application des freins pour la loi de commande TTC pour une route d\'asphalte sec (~1\u03BC)')
#xname = sys.state_label[0] + ' ' + sys.state_units[0]
#yname = sys.state_label[1] + ' ' + sys.state_units[1] 
#
#i = axs[0].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttcr_1.T, shading='gouraud', cmap = 'plasma')
#axs[0].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs[0])
#axs[0].set_ylabel('VITESSE') 
#axs[0].set_xlabel('POSITION') 
#axs[0].set_title('-0m/s^2')
#
#i = axs[1].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttcr_3.T, shading='gouraud', cmap = 'plasma')
#axs[1].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs[1])
#axs[1].set_ylabel('VITESSE') 
#axs[1].set_xlabel('POSITION') 
#axs[1].set_title('-3m/s^2')
#
#i = axs[2].pcolormesh(grid_sys_controller.x_level[0], grid_sys_controller.x_level[1], ttcr_6.T, shading='gouraud', cmap = 'plasma')
#axs[2].axis([grid_sys_controller.x_level[0][0], grid_sys_controller.x_level[0][-1], grid_sys_controller.x_level[1][0], grid_sys_controller.x_level[1][-1]])
#fig.colorbar(i, ax=axs[2])
#axs[2].set_ylabel('VITESSE') 
#axs[2].set_xlabel('POSITION') 
#axs[2].set_title('-6m/s^2')

#%% PLOT EVERYTHING
compare_controllers = True

if len(controlers_to_test) > 1:
    for n in range(len(name_list)):
        
        x0 = np.array([-100.0,10.0])
        x1 = np.array([-100.0,15.0])
        x2 = np.array([-100.0,20.0])
        
        x3 = np.array([-80.0,15.0])
        x4 = np.array([-60.0,15.0])
        x5 = np.array([-20.0,15.0])
    
        x0_test = [x0,x1,x2,x3,x4,x5]
        
        sim_flag = True
               
        controlers = list()
        cost_to_go = list()
        
        sim_list_vi = list()
        sim_list_ttc = list()
        sim_list_msd = list()
        sim_list_human = list()
        
        for con in controlers_to_test:
           if controler[con] == 'Vi':  
               controlers.append(grid_sys_list[n].get_grid_from_array(grid_sys_list[n].get_input_from_policy(vi_list[n].pi, 0)).T)
               cost_to_go.append(vi_c2g_list[n])
               
           if controler[con] == 'Ttc':  
               controlers.append(ttc_list[n].c_array().T)
               cost_to_go.append(ttc_c2g_list[n])
               
           if controler[con] == 'Msd':  
               controlers.append(msd_list[n].c_array().T)
               cost_to_go.append(msd_c2g_list[n])
               
           if controler[con] == 'Human':  
               controlers.append(human_list[n].c_array().T)
               cost_to_go.append(human_c2g_list[n])     
           if sim_flag:
               for x0_to_test in x0_test:
                    if controler[con] == 'Vi':  
                        vi_cl_list[n].x0 = x0_to_test
                        sim_vi = s.SimulatorV2(vi_cl_list[n], x0_end=sys.x_ub[0])
                        sim_list_vi.append(sim_vi.traj_to_args(sim_vi.traj))
                        
                    if controler[con] == 'Ttc':  
                        ttc_cl_list[n].x0 = x0_to_test
                        sim_ttc = s.SimulatorV2(ttc_cl_list[n], x0_end=sys.x_ub[0])
                        sim_list_ttc.append(sim_ttc.traj_to_args(sim_ttc.traj))
                        
                    if controler[con] == 'Msd':  
                        msd_cl_list[n].x0 = x0_to_test
                        sim_msd = s.SimulatorV2(msd_cl_list[n], x0_end=sys.x_ub[0])
                        sim_list_msd.append(sim_msd.traj_to_args(sim_msd.traj))
                        
                    if controler[con] == 'Human':  
                        human_cl_list[n].x0 = x0_to_test
                        sim_human = s.SimulatorV2(human_cl_list[n], x0_end=sys.x_ub[0])
                        sim_list_human.append(sim_human.traj_to_args(sim_human.traj))
                 
        for c2c in range(len(compare_mode)):
             fig, axs = plt.subplots(len(controlers_to_test), 5)
             plt.ion()
             fig.suptitle(str(name_list[n])+'     Comparaison: '+comparaison[compare_mode[c2c]] + ' Vi - controler (- = vi meilleur)')
             xname = sys.state_label[0] + ' ' + sys.state_units[0]
             yname = sys.state_label[1] + ' ' + sys.state_units[1] 
             if len(controlers_to_test) > 1:
                 axs[0][0].set_title('Controler')
                 axs[0][1].set_title('Cost to Go Total')
                 axs[0][2].set_title('Cost to Go Confort: ' + str(cf.confort_coef))
                 axs[0][3].set_title('Cost to Go Security: '+ str(cf.security_coef))
                 axs[0][4].set_title('Cost to Go Override: '+ str(cf.override_coef))
             else:
                 axs[0].set_title('Controler')
                 axs[1].set_title('Cost to Go Total')
                 axs[2].set_title('Cost to Go Confort: ' + str(cf.confort_coef))
                 axs[3].set_title('Cost to Go Security: '+ str(cf.security_coef))
                 axs[4].set_title('Cost to Go Override: '+ str(cf.override_coef))
     
             ### PLOTTING 
             if len(controlers_to_test) > 1:
                 for c in range(len(controlers_to_test)):
                    if comparaison[compare_mode[c2c]] == 'NoDiff':
                          controler_map = controlers[c]
                          # controler_map[controler_map==0] = 1
                          # controler_map[controler_map<0] = 0
                          cost_to_go_total = cost_to_go[c].cost2go_map_list[0].T
                          cost_to_go_confort = cost_to_go[c].cost2go_map_list[1].T
                          cost_to_go_security = cost_to_go[c].cost2go_map_list[2].T
                          cost_to_go_override = cost_to_go[c].cost2go_map_list[3].T
                       
                    if comparaison[compare_mode[c2c]] == 'Normal':            
                         # controler_map = controlers[c]
                         if c == 0:
                              controler_map = controlers[c]
                             
                              
                              
                              cost_to_go_total = cost_to_go[c].cost2go_map_list[0].T
                              cost_to_go_confort = cost_to_go[c].cost2go_map_list[1].T
                              cost_to_go_security = cost_to_go[c].cost2go_map_list[2].T
                              cost_to_go_override = cost_to_go[c].cost2go_map_list[3].T
                         else:
                              controler_map = controlers[0] - controlers[c]
                              cost_to_go_total = cost_to_go[0].cost2go_map_list[0].T - cost_to_go[c].cost2go_map_list[0].T
                              cost_to_go_confort = cost_to_go[0].cost2go_map_list[1].T - cost_to_go[c].cost2go_map_list[1].T
                              cost_to_go_security = cost_to_go[0].cost2go_map_list[2].T - cost_to_go[c].cost2go_map_list[2].T
                              cost_to_go_override = cost_to_go[0].cost2go_map_list[3].T - cost_to_go[c].cost2go_map_list[3].T
    
                    if comparaison[compare_mode[c2c]] == 'Bool':
                         if c == 0:            
                              controler_map = controlers[c]
                              cost_to_go_total = cost_to_go[c].cost2go_map_list[0].T
                              cost_to_go_confort = cost_to_go[c].cost2go_map_list[1].T
                              cost_to_go_security = cost_to_go[c].cost2go_map_list[2].T
                              cost_to_go_override = cost_to_go[c].cost2go_map_list[3].T
                         else:
                              controler_map = controlers[0] - controlers[c]
                              controler_map[controler_map==0] = 0
                              controler_map[controler_map<0] = -1
                              controler_map[controler_map>0] = 1
                              cost_to_go_total = cost_to_go[0].cost2go_map_list[0].T - cost_to_go[c].cost2go_map_list[0].T
                              cost_to_go_confort = cost_to_go[0].cost2go_map_list[1].T - cost_to_go[c].cost2go_map_list[1].T
                              cost_to_go_security = cost_to_go[0].cost2go_map_list[2].T - cost_to_go[c].cost2go_map_list[2].T
                              cost_to_go_override = cost_to_go[0].cost2go_map_list[3].T - cost_to_go[c].cost2go_map_list[3].T
                              
                              controler_map[controler_map==0] = 0
                              controler_map[controler_map<0] = -1
                              controler_map[controler_map>0] = 1
                              
                              cost_to_go_total[cost_to_go_total==0] = 0
                              cost_to_go_total[cost_to_go_total<0] = -1
                              cost_to_go_total[cost_to_go_total>0] = 1
                              
                              cost_to_go_confort[cost_to_go_confort==0] = 0
                              cost_to_go_confort[cost_to_go_confort<0] = -1
                              cost_to_go_confort[cost_to_go_confort>0] = 1
                              
                              cost_to_go_security[cost_to_go_security==0] = 0
                              cost_to_go_security[cost_to_go_security<0] = -1
                              cost_to_go_security[cost_to_go_security>0] = 1
                              
                              cost_to_go_override[cost_to_go_override==0] = 0
                              cost_to_go_override[cost_to_go_override<0] = -1
                              cost_to_go_override[cost_to_go_override>0] = 1
                      
                    
                    plotting_colors = 'plasma'
                    
                    if controler[controlers_to_test[c]] == 'Vi':
                        grid_sys_plot = grid_sys_list[n]
                    else:
                        grid_sys_plot = grid_sys_controller_list[n]
                            
                        
                    
                    axs[c][0].set_ylabel(controler[controlers_to_test[c]]) 
                    i = axs[c][0].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], controler_map, shading='gouraud', cmap = plotting_colors)
                    axs[c][0].axis([grid_sys_plot.x_level[0][0], grid_sys_plot.x_level[0][-1], grid_sys_plot.x_level[1][0], grid_sys_plot.x_level[1][-1]])
                    fig.colorbar(i, ax=axs[c][0])
                    axs[c][0].grid(True)
                     
                    i = axs[c][1].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], cost_to_go_total, shading='gouraud', cmap = plotting_colors)
                    axs[c][1].axis([grid_sys_plot.x_level[0][0], grid_sys_plot.x_level[0][-1], grid_sys_plot.x_level[1][0], grid_sys_plot.x_level[1][-1]])
                    fig.colorbar(i, ax=axs[c][1])
                    axs[c][1].grid(True)
                    axs[c][1].set_yticklabels([])
                    
                    i = axs[c][2].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], cost_to_go_confort, shading='gouraud', cmap = plotting_colors)
                    axs[c][2].axis([grid_sys_plot.x_level[0][0], grid_sys_plot.x_level[0][-1], grid_sys_plot.x_level[1][0], grid_sys_plot.x_level[1][-1]])
                    fig.colorbar(i, ax=axs[c][2])
                    axs[c][2].grid(True)
                    axs[c][2].set_yticklabels([])
                     
                    i = axs[c][3].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], cost_to_go_security, shading='gouraud', cmap = plotting_colors)
                    axs[c][3].axis([grid_sys_plot.x_level[0][0], grid_sys_plot.x_level[0][-1], grid_sys_plot.x_level[1][0], grid_sys_plot.x_level[1][-1]])
                    fig.colorbar(i, ax=axs[c][3])
                    axs[c][3].grid(True)
                    axs[c][3].set_yticklabels([])
                     
                    i = axs[c][4].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], cost_to_go_override, shading='gouraud', cmap = plotting_colors)
                    axs[c][4].axis([grid_sys_plot.x_level[0][0], grid_sys_plot.x_level[0][-1], grid_sys_plot.x_level[1][0], grid_sys_plot.x_level[1][-1]])
                    fig.colorbar(i, ax=axs[c][4])
                    axs[c][4].grid(True)
                    axs[c][4].set_yticklabels([])
                    if sim_flag:
                        if controler[controlers_to_test[c]] == 'Vi':
                            for s_vi in sim_list_vi: 
                                 for ax_t in range(len(axs[1])):
                                     axs[c][ax_t].plot(s_vi[:,2],s_vi[:,3])
                                     axs[c][ax_t].plot(s_vi[:,0],s_vi[:,1])
                                     
                        if controler[controlers_to_test[c]] == 'Ttc':
                            for s_ttc in sim_list_ttc: 
                                 for ax_t in range(len(axs[1])):
                                     axs[c][ax_t].plot(s_ttc[:,2],s_ttc[:,3])
                                     axs[c][ax_t].plot(s_ttc[:,0],s_ttc[:,1])
                                     
                        if controler[controlers_to_test[c]] == 'Msd':
                            for s_msd in sim_list_msd: 
                                 for ax_t in range(len(axs[1])):
                                     axs[c][ax_t].plot(s_msd[:,2],s_msd[:,3])
                                     axs[c][ax_t].plot(s_msd[:,0],s_msd[:,1]) 
                                     
                        if controler[controlers_to_test[c]] == 'Human':
                            for s_human in sim_list_human: 
                                 for ax_t in range(len(axs[1])):
                                     axs[c][ax_t].plot(s_human[:,2],s_human[:,3])
                                     axs[c][ax_t].plot(s_human[:,0],s_human[:,1])                



if len(controlers_to_test) == 1:
    for n in range(len(name_list)):
        
        x0 = np.array([-100.0,10.0])
        x1 = np.array([-100.0,15.0])
        x2 = np.array([-100.0,20.0])
        
        x3 = np.array([-80.0,15.0])
        x4 = np.array([-60.0,15.0])
        x5 = np.array([-20.0,15.0])
    
        x0_test = [x0,x1,x2,x3,x4,x5]
               
        controlers = list()
        cost_to_go = list()
        
        sim_list_vi = list()
        sim_list_ttc = list()
        sim_list_msd = list()
        sim_list_human = list()
        
        for con in controlers_to_test:
           if controler[con] == 'Vi':  
               controlers.append(grid_sys_list[n].get_grid_from_array(grid_sys_list[n].get_input_from_policy(vi_list[n].pi, 0)).T)
               if compute_c2g_flag: 
                    cost_to_go.append(vi_c2g_list[n])
               
           if controler[con] == 'Ttc':  
               controlers.append(ttc_list[n].c_array().T)
               if compute_c2g_flag: 
                    cost_to_go.append(ttc_c2g_list[n])
               
           if controler[con] == 'Msd':  
               controlers.append(msd_list[n].c_array().T)
               if compute_c2g_flag: 
                    cost_to_go.append(msd_c2g_list[n])
               
           if controler[con] == 'Human':  
               controlers.append(human_list[n].c_array().T)
               if compute_c2g_flag: 
                    cost_to_go.append(human_c2g_list[n])     
           if sim_flag:
               for x0_to_test in x0_test:
                    if controler[con] == 'Vi':  
                        vi_cl_list[n].x0 = x0_to_test
                        sim_vi = s.SimulatorV2(vi_cl_list[n], x0_end=sys.x_ub[0])
                        sim_list_vi.append(sim_vi.traj_to_args(sim_vi.traj))
                        
                    if controler[con] == 'Ttc':  
                        ttc_cl_list[n].x0 = x0_to_test
                        sim_ttc = s.SimulatorV2(ttc_cl_list[n], x0_end=sys.x_ub[0])
                        sim_list_ttc.append(sim_ttc.traj_to_args(sim_ttc.traj))
                        
                    if controler[con] == 'Msd':  
                        msd_cl_list[n].x0 = x0_to_test
                        sim_msd = s.SimulatorV2(msd_cl_list[n], x0_end=sys.x_ub[0])
                        sim_list_msd.append(sim_msd.traj_to_args(sim_msd.traj))
                        
                    if controler[con] == 'Human':  
                        human_cl_list[n].x0 = x0_to_test
                        sim_human = s.SimulatorV2(human_cl_list[n], x0_end=sys.x_ub[0])
                        sim_list_human.append(sim_human.traj_to_args(sim_human.traj))
                 
        for c2c in range(len(compare_mode)):
             if plot_c2g_flag:
                  fig, axs = plt.subplots(len(controlers_to_test), 5)
                  plt.ion()
                  fig.suptitle(str(name_list[n])+'     Comparaison: '+comparaison[compare_mode[c2c]] + ' Vi - controler (- = vi meilleur)')
                  xname = sys.state_label[0] + ' ' + sys.state_units[0]
                  yname = sys.state_label[1] + ' ' + sys.state_units[1] 
                  
                  axs[0].set_title('Controler')
                  axs[1].set_title('Cost to Go Total')
                  axs[2].set_title('Cost to Go Confort: ' + str(cf.confort_coef))
                  axs[3].set_title('Cost to Go Security: '+ str(cf.security_coef))
                  axs[4].set_title('Cost to Go Override: '+ str(cf.override_coef))
          
                  ### PLOTTING 
                  if len(controlers_to_test) == 1:
                      for c in range(len(controlers_to_test)):
                         if comparaison[compare_mode[c2c]] == 'NoDiff':
                               controler_map = controlers[c]
                               controler_map[controler_map==0] = 0
                               cost_to_go_total = cost_to_go[c].cost2go_map_list[0].T
                               cost_to_go_confort = cost_to_go[c].cost2go_map_list[1].T
                               cost_to_go_security = cost_to_go[c].cost2go_map_list[2].T
                               cost_to_go_override = cost_to_go[c].cost2go_map_list[3].T
     
                           
                         
                         plotting_colors = 'plasma'
                         
                         if controler[controlers_to_test[c]] == 'Vi':
                             grid_sys_plot = grid_sys_list[n]
                         else:
                             grid_sys_plot = grid_sys_controller_list[n]
                                 
                             
                         
                         axs[0].set_ylabel(controler[controlers_to_test[c]]) 
                         i = axs[0].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], controler_map, shading='gouraud', cmap = plotting_colors)
                         axs[0].axis([grid_sys_plot.x_level[0][0], grid_sys_plot.x_level[0][-1], grid_sys_plot.x_level[1][0], grid_sys_plot.x_level[1][-1]])
                         fig.colorbar(i, ax=axs[0])
                         axs[0].grid(True)
                          
                         i = axs[1].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], cost_to_go_total, shading='gouraud', cmap = plotting_colors)
                         axs[1].axis([grid_sys_plot.x_level[0][0], grid_sys_plot.x_level[0][-1], grid_sys_plot.x_level[1][0], grid_sys_plot.x_level[1][-1]])
                         fig.colorbar(i, ax=axs[1])
                         axs[1].grid(True)
                         axs[1].set_yticklabels([])
                         
                         i = axs[2].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], cost_to_go_confort, shading='gouraud', cmap = plotting_colors)
                         axs[2].axis([grid_sys_plot.x_level[0][0], grid_sys_plot.x_level[0][-1], grid_sys_plot.x_level[1][0], grid_sys_plot.x_level[1][-1]])
                         fig.colorbar(i, ax=axs[2])
                         axs[2].grid(True)
                         axs[2].set_yticklabels([])
                          
                         i = axs[3].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], cost_to_go_security, shading='gouraud', cmap = plotting_colors)
                         axs[3].axis([grid_sys_plot.x_level[0][0], grid_sys_plot.x_level[0][-1], grid_sys_plot.x_level[1][0], grid_sys_plot.x_level[1][-1]])
                         fig.colorbar(i, ax=axs[3])
                         axs[3].grid(True)
                         axs[3].set_yticklabels([])
                          
                         i = axs[4].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], cost_to_go_override, shading='gouraud', cmap = plotting_colors)
                         axs[4].axis([grid_sys_plot.x_level[0][0], grid_sys_plot.x_level[0][-1], grid_sys_plot.x_level[1][0], grid_sys_plot.x_level[1][-1]])
                         fig.colorbar(i, ax=axs[4])
                         axs[4].grid(True)
                         axs[4].set_yticklabels([])
                         if sim_flag:
                             if controler[controlers_to_test[c]] == 'Vi':
                                 for s_vi in sim_list_vi: 
                                      for ax_t in range(len(axs)):
                                          axs[ax_t].plot(s_vi[:,2],s_vi[:,3])
                                          axs[ax_t].plot(s_vi[:,0],s_vi[:,1])
                                          
                             if controler[controlers_to_test[c]] == 'Ttc':
                                 for s_ttc in sim_list_ttc: 
                                      for ax_t in range(len(axs)):
                                          axs[ax_t].plot(s_ttc[:,2],s_ttc[:,3])
                                          axs[ax_t].plot(s_ttc[:,0],s_ttc[:,1])
                                          
                             if controler[controlers_to_test[c]] == 'Msd':
                                 for s_msd in sim_list_msd: 
                                      for ax_t in range(len(axs)):
                                          axs[ax_t].plot(s_msd[:,2],s_msd[:,3])
                                          axs[ax_t].plot(s_msd[:,0],s_msd[:,1]) 
                                          
                             if controler[controlers_to_test[c]] == 'Human':
                                 for s_human in sim_list_human: 
                                      for ax_t in range(len(axs)):
                                          axs[ax_t].plot(s_human[:,2],s_human[:,3])
                                          axs[ax_t].plot(s_human[:,0],s_human[:,1])  
             else:          
                  fig, axs = plt.subplots(len(controlers_to_test), 1)
                  plt.ion()
                  #fig.suptitle(('MSD pour de l\'Asphalte Mouillé (0.7\u03BC) avec t1 = 0.75, t2 = 0.75 et t3 = 0.25'))
                  fig.suptitle(('TTC pour un route: ' + sys.road[-1]))
                  xname = sys.state_label[0] + ' ' + sys.state_units[0]
                  yname = sys.state_label[1] + ' ' + sys.state_units[1] 
                  
#                  axs.set_title('Asphalte Mouillé ( 0.7\u03BC) avec un conducteur de base tf = 1.75, tm = 4.2 et t\'m = 0.75')
          
                  ### PLOTTING 
                  if len(controlers_to_test) == 1:
                      for c in range(len(controlers_to_test)):
                         if comparaison[compare_mode[c2c]] == 'NoDiff':
                               controler_map = controlers[c]
                           
                         
                         plotting_colors = 'plasma'
                         
                         if controler[controlers_to_test[c]] == 'Vi':
                             grid_sys_plot = grid_sys_list[n]
                         else:
                             grid_sys_plot = grid_sys_controller_list[n]
                                 
                             
                         
                         axs.set_ylabel('VITESSE') 
                         axs.set_xlabel('POSITION') 
                         i = axs.pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], controler_map, shading='gouraud', cmap = plotting_colors)
                         axs.axis([grid_sys_plot.x_level[0][0], grid_sys_plot.x_level[0][-1], grid_sys_plot.x_level[1][0], grid_sys_plot.x_level[1][-1]])
                         fig.colorbar(i, ax=axs)
                         axs.grid(True)

                         if sim_flag:
                             if controler[controlers_to_test[c]] == 'Vi':
                                 for s_vi in sim_list_vi: 
                                      axs.plot(s_vi[:,2],s_vi[:,3])
                                      axs.plot(s_vi[:,0],s_vi[:,1])
                                          
                             if controler[controlers_to_test[c]] == 'Ttc':
                                 for s_ttc in sim_list_ttc: 
                                      axs.plot(s_ttc[:,2],s_ttc[:,3])
                                      axs.plot(s_ttc[:,0],s_ttc[:,1])
                                          
                             if controler[controlers_to_test[c]] == 'Msd':
                                 for s_msd in sim_list_msd: 
                                      axs.plot(s_msd[:,2],s_msd[:,3])
                                      axs.plot(s_msd[:,0],s_msd[:,1]) 
                                          
                             if controler[controlers_to_test[c]] == 'Human':
                                 for s_human in sim_list_human: 
                                      axs.plot(s_human[:,2],s_human[:,3])
                                      axs.plot(s_human[:,0],s_human[:,1])  
#%%
for con in controlers_to_test:
   if controler[con] == 'Vi':  
       controlers.append(grid_sys_list[n].get_grid_from_array(grid_sys_list[n].get_input_from_policy(vi_list[n].pi, 0)).T)
       if compute_c2g_flag:
           cost_to_go.append(vi_c2g_list[n])
       
   if controler[con] == 'Ttc':  
       controlers.append(ttc_list[n].c_array().T)
       if compute_c2g_flag:
           cost_to_go.append(ttc_c2g_list[n])
       
   if controler[con] == 'Msd':  
       controlers.append(msd_list[n].c_array().T)
       if compute_c2g_flag:
           cost_to_go.append(msd_c2g_list[n])
       
   if controler[con] == 'Human':  
       controlers.append(human_list[n].c_array().T)
       if compute_c2g_flag:
           cost_to_go.append(human_c2g_list[n])     

if compare_controllers:
    controlers_vi = list()
    controlers_ttc = list()
    controlers_msd = list()
    controlers_human = list()
    
    
    if len(drivers_to_test)>1:
        fig, axs = plt.subplots(1, len(drivers_to_test))
        plt.ion()
        fig.suptitle(str(name_list[n])+'     Comparaison: '+comparaison[compare_mode[c2c]] + ' Vi - controler (- = vi meilleur)')
        xname = sys.state_label[0] + ' ' + sys.state_units[0]
        yname = sys.state_label[1] + ' ' + sys.state_units[1] 
        for c in range(len(controlers_to_test)):
            controler_map = controlers[c]
            axs[c].set_ylabel(controler[controlers_to_test[c]]) 
            i = axs[c].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], controler_map, shading='gouraud', cmap = plotting_colors)
            axs[c].axis([grid_sys_plot.x_level[0][0], grid_sys_plot.x_level[0][-1], grid_sys_plot.x_level[1][0], grid_sys_plot.x_level[1][-1]])
            fig.colorbar(i, ax=axs[c])
            axs[c].grid(True)
            
            
    if len(roads_to_test)>1:
        if len(controlers_to_test)==1:
             for c in range(len(controlers_to_test)):
                 fig, axs = plt.subplots(1, len(roads_to_test))
                 plt.ion()
                 fig.suptitle(controler[controlers_to_test[c]])
                 xname = sys.state_label[0] + ' ' + sys.state_units[0]
                 yname = sys.state_label[1] + ' ' + sys.state_units[1] 
                 
                 for r in range(len(roads_to_test)):
                     if controler[controlers_to_test[c]] == 'Vi':
                         controler_map = grid_sys_list[r].get_grid_from_array(grid_sys_list[r].get_input_from_policy(vi_list[r].pi, 0)).T
                     if controler[controlers_to_test[c]] == 'Ttc':
                         controler_map = ttc_list[r].c_array().T
                     if controler[controlers_to_test[c]] == 'Msd':
                         controler_map = msd_list[r].c_array().T
                     if controler[controlers_to_test[c]] == 'Human':
                         controler_map = human_list[r].c_array().T
                     
                     if controler[controlers_to_test[c]] == 'Vi':
                         i = axs[r].pcolormesh(grid_sys_list[r].x_level[0], grid_sys_list[r].x_level[1], controler_map, shading='gouraud', cmap = plotting_colors)
                         axs[r].axis([grid_sys_list[r].x_level[0][0], grid_sys_list[r].x_level[0][-1], grid_sys_list[r].x_level[1][0], grid_sys_list[r].x_level[1][-1]])
                         fig.colorbar(i, ax=axs[r])
                         axs[r].grid(True)
                     else:
                         i = axs[r].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], controler_map, shading='gouraud', cmap = plotting_colors)
                         axs[r].axis([grid_sys_plot.x_level[0][0], grid_sys_plot.x_level[0][-1], grid_sys_plot.x_level[1][0], grid_sys_plot.x_level[1][-1]])
                         fig.colorbar(i, ax=axs[r])
                         axs[r].grid(True) 
        
    if len(roads_to_test)>1:
        if len(controlers_to_test)>1:
             for c in range(len(controlers_to_test)):
                 fig, axs = plt.subplots(len(controlers_to_test), len(roads_to_test))
                 plt.ion()
                 fig.suptitle(controler[controlers_to_test[c]])
                 xname = sys.state_label[0] + ' ' + sys.state_units[0]
                 yname = sys.state_label[1] + ' ' + sys.state_units[1] 
                 
                 for r in range(len(roads_to_test)):
                     if controler[controlers_to_test[c]] == 'Vi':
                         controler_map = grid_sys_list[r].get_grid_from_array(grid_sys_list[r].get_input_from_policy(vi_list[r].pi, 0)).T
                     if controler[controlers_to_test[c]] == 'Ttc':
                         controler_map = ttc_list[r].c_array().T
                     if controler[controlers_to_test[c]] == 'Msd':
                         controler_map = msd_list[r].c_array().T
                     if controler[controlers_to_test[c]] == 'Human':
                         controler_map = human_list[r].c_array().T
                     
                     if controler[controlers_to_test[c]] == 'Vi':
                         i = axs[r].pcolormesh(grid_sys_list[r].x_level[0], grid_sys_list[r].x_level[1], controler_map, shading='gouraud', cmap = plotting_colors)
                         axs[r].axis([grid_sys_list[r].x_level[0][0], grid_sys_list[r].x_level[0][-1], grid_sys_list[r].x_level[1][0], grid_sys_list[r].x_level[1][-1]])
                         fig.colorbar(i, ax=axs[r])
                         axs[r].grid(True)
                     else:
                         i = axs[r].pcolormesh(grid_sys_plot.x_level[0], grid_sys_plot.x_level[1], controler_map, shading='gouraud', cmap = plotting_colors)
                         axs[r].axis([grid_sys_plot.x_level[0][0], grid_sys_plot.x_level[0][-1], grid_sys_plot.x_level[1][0], grid_sys_plot.x_level[1][-1]])
                         fig.colorbar(i, ax=axs[r])
                         axs[r].grid(True)        
             
        
        
        
        