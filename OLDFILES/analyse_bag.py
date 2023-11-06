#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:42:16 2022

@author: clearpath-robot
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import costfunction
from scipy.interpolate import interp1d

path = '/home/clearpath-robot/maitrise/PythonProject/bag/'
#path = '/home/nvidia/X-Maxx_racecar/BagReader/BagFiles_XMAXX'
folder = 'g_test/'
file_path = path+folder

user_vel = genfromtxt(file_path+'bluetooth_teleop-cmd_vel.csv', delimiter=',')
con_vel = genfromtxt(file_path+'con_vel.csv', delimiter=',')
front_scan = genfromtxt(file_path+'front_scan.csv', delimiter=',')
imu = genfromtxt(file_path+'imu-data.csv', delimiter=',')
velocity = genfromtxt(file_path+'jackal_velocity_controller-odom_no_cov.csv', delimiter=',')

def g(x, u,):
  dJ = g_confort(x, u) + g_override(x, u) + g_security(x, u)   
  if (u[1] == 0 and u[0] != 0) or (u[1] == 1 and u[0] == 0):
      dJ = 200000
  
  return dJ   
    
def g_security( x, u):
  p = x[0]
  v = x[1]
  security = 0
  if p > 95:
    security = 1
  if p > 97:
    security = 5
  if p >= 99:
    security = 10 * ((v)**2)
  if p >= 100:
    security = 10000 * ((v)**2) #+ (100 * ((v)**2) * p-99)
  
  security = (100 * v**2) / (1+np.exp(-0.5*(p-95)))
  
  return security

def g_confort( x, u): 
    p = x[0]
    a = x[2]
    
    return (0.01*(a**2)) * ((100-p)**2)

def g_override(x, u):
    return (u[1]*10)
  
def down_sample(array, pts):
  interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
  x = interpolated(np.linspace(0, len(array), pts))
  return x

x = front_scan[:,1]
xp = velocity[:,13]
xpp = imu[:,30]
user = user_vel[:,1]
con_vel = con_vel[:,1]

t = imu[1:-2,0]
t = t-t[0]
x = down_sample(x,680)
xp = down_sample(xp,680)
xpp = down_sample(xpp,680)
user = down_sample(user,680)
con_vel = down_sample(con_vel,680)

x1 = np.array([x, xp, xpp])

u = np.zeros([680,2])
for i in range(680):
  if con_vel[i] >= 1.95:
    u[i][0] = 0
    u[i][1] = 0
  else:
    u[i][0] = -0.2
    u[i][1] = 1
    
dJ = np.zeros(680)
for i in range(680):
  dJ[i] = g(x1[:,i], u[i])
  
dJ[0] = 0
J = np.cumsum(dJ)*0.1

#plt.plot(x)
#plt.plot(xp)
#plt.plot(xpp)
#plt.plot(con_vel)
#plt.plot(user)

fig, axs = plt.subplots(3, 1)
axs[0].set_title('États du véhicule')
axs[0].plot(t, x, label='position')
axs[0].plot(t, xp, label='vitesse')
axs[0].plot(t, xpp, label='acceleration')
axs[0].legend()

axs[1].set_title('Commandes user et controller')
axs[1].plot(t, user, label='User')
axs[1].plot(t, con_vel, label='controller')
axs[1].legend()

axs[2].set_title('Coût')
axs[2].plot(t, J, label='J')
axs[2].plot(t, dJ, label='dJ')
axs[2].legend()



