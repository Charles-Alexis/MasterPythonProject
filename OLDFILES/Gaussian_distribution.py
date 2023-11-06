#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 12:07:50 2022

@author: clearpath-robot
"""
#%%
import numpy as np
import valueiteration
import discretizer
import system
import costfunction
import copy

sys = system.LongitudinalFrontWheelDriveCarWithDriverModel()
sys.lenght = 0.6
sys.xc = 0.3
sys.yc = 0.175
sys.mass = 25
sys.cdA = 0.3 * 0.105
sys.x_ub = np.array([+20.0, 6.0])
sys.x_lb = np.array([0, 0])
sys.u_ub = np.array([0.0, 1])
sys.u_lb = np.array([-0.3, 0])
#GRIDSYS
grid_sys = discretizer.GridDynamicSystem(sys, (81, 81), (16, 2), 0.1)

sys.driver = sys.driver_xmaxx['Ok']
sys.road = sys.roads['AsphalteDry']
sys.obs_dist = sys.x_ub[0]

cf = costfunction.DriverModelCostFunction.from_sys(sys)
cf.confort_coef = 1
cf.override_coef = 1
cf.security_coef = 5
cf.xbar = np.array( [0, 0] ) # target
sys.cost_function = cf

vi = valueiteration.ValueIteration_ND(grid_sys, cf)
vi.threshold = 0.5
vi.uselookuptable = False
vi.initialize()
vi_u11 = copy.deepcopy(vi)
vi_u1 = copy.deepcopy(vi)
vi_u075 = copy.deepcopy(vi)
vi_u04 = copy.deepcopy(vi)
vi_u015 = copy.deepcopy(vi)
vi_u005 = copy.deepcopy(vi)

vi_u1.load_data('XMAXX_MAP/XMAXX_u1_ok')
vi_u1.assign_interpol_controller()
vi_u1.ctl.vi_law = vi_u1.vi_law

vi_u11.load_data('XMAXX_MAP/XMAXX_u11_ok')
vi_u11.assign_interpol_controller()
vi_u11.ctl.vi_law = vi_u11.vi_law

vi_u075.load_data('XMAXX_MAP/XMAXX_u075_ok')
vi_u075.assign_interpol_controller()
vi_u075.ctl.vi_law = vi_u075.vi_law

vi_u04.load_data('XMAXX_MAP/XMAXX_u040_ok')
vi_u04.assign_interpol_controller()
vi_u04.ctl.vi_law = vi_u04.vi_law

vi_u015.load_data('XMAXX_MAP/XMAXX_u015_ok')
vi_u015.assign_interpol_controller()
vi_u015.ctl.vi_law = vi_u015.vi_law

vi_u005.load_data('XMAXX_MAP/XMAXX_u005_ok')
vi_u005.assign_interpol_controller()
vi_u005.ctl.vi_law = vi_u005.vi_law 


#%%
import numpy as np
import matplotlib.pyplot as plt

mu_ref = 0.5
sigma_ref = 0.2
x = (np.arange(1001)/500)

def gaussian_func(x_d, mu_d, sigma_d):
  return np.exp(-1*(((x_d-mu_d)**2)/(2*(sigma_d**2))))/(sigma_d*np.sqrt(2*np.pi))

x_t = [x[25], x[75], x[200], x[375], x[500], x[550]]
mu = [0.05, 0.15, 0.4, 0.75, 1.0, 1,1]
res = np.zeros(len(mu)-1)
res_p = np.zeros(len(mu)-1)

i = 0
plt.figure()
plt.plot(x,gaussian_func(x,mu_ref,sigma_ref))
for x_test in x_t:
  res[i] = gaussian_func(x_test,mu_ref,sigma_ref)
  plt.plot(x_test, res[i], 'x',)
  i=i+1
plt.show()

res_tot = np.sum(res)
res_p = res / res_tot

vis = [vi_u005, vi_u015, vi_u04, vi_u075, vi_u1, vi_u11]

v_test = 5
d_test = 12
i=0
cmd_r = np.zeros([6,4])

for vi_c in vis:
  #vi_c.plot_policy()
  cmd = vi_c.ctl.vi_law(np.array([d_test, v_test])) 
  cmd_r[i,0] = cmd[0]
  cmd_r[i,1] = cmd[1]
  cmd_r[i,2] = cmd[0]* res_p[i]
  cmd_r[i,3] = cmd[1]* res_p[i]
  i = i+1  

print(cmd_r)
print(np.sum(cmd_r[:,2]))
print(np.sum(cmd_r[:,3]))
  
  

