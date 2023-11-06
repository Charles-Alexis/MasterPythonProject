# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 13:35:15 2022

@author: Charles-Alexis
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

#%%

from numpy import genfromtxt
#path = '/home/nvidia/X-Maxx_racecar/BagReader/BagFiles_XMAXX/'
#path = '/home/nvidia/X-Maxx_racecar/BagReader/BagFiles_XMAXX'
path = '/home/nvidia/bags_Xmaxx'


folder = 'bag24aout/'
file = 'Acceleration_24aout_cl_pid_smc_dehors/'

file_path = path+folder+file

#imu_acc = genfromtxt(file_path+'imu-acceleration.csv', delimiter=',')
prop_sensor = genfromtxt(file_path+'prop_sensors.csv', delimiter=',')
prop_cmd_costum = genfromtxt(file_path+'prop_cmd_costum.csv', delimiter=',')
joy = genfromtxt(file_path+'joy.csv', delimiter=',')
#mu_est = genfromtxt(file_path+'mu_est.csv', delimiter=',')

#prop_sensor[:,12] = np.nan_to_num(prop_sensor[:,12])
'''
imu_acc_file = open(file_path+'joy.csv')
csvreader = csv.reader(imu_acc_file)
imu_acc_header = []
header = next(csvreader)
imu_acc_rows = []
for row in csvreader:
        imu_acc_rows.append(row)
imu_acc_file.close()'''

slip_time = np.zeros(5*np.shape(prop_sensor[:,13])[0])
slip_droit5 = np.zeros(5*np.shape(prop_sensor[:,13])[0])
slip_gauche5 = np.zeros(5*np.shape(prop_sensor[:,13])[0])

for i in range(np.shape(prop_sensor[:,13])[0]):
        if i < np.shape(prop_sensor[:,13])[0]-1:
            dt_slip = (prop_sensor[(i+1),0] - prop_sensor[(i),0])/5
        
        slip_time[(i*5)+0] = prop_sensor[i,0] + (dt_slip*0)
        slip_time[(i*5)+1] = prop_sensor[i,0] + (dt_slip*1)
        slip_time[(i*5)+2] = prop_sensor[i,0] + (dt_slip*2)
        slip_time[(i*5)+3] = prop_sensor[i,0] + (dt_slip*3)
        slip_time[(i*5)+4] = prop_sensor[i,0] + (dt_slip*4)
        
        slip_droit5[(i*5)+0] = prop_sensor[i,18]
        slip_droit5[(i*5)+1] = prop_sensor[i,19]
        slip_droit5[(i*5)+2] = prop_sensor[i,20]
        slip_droit5[(i*5)+3] = prop_sensor[i,21]
        slip_droit5[(i*5)+4] = prop_sensor[i,22]
        
        slip_gauche5[(i*5)+0] = prop_sensor[i,13]
        slip_gauche5[(i*5)+1] = prop_sensor[i,14]
        slip_gauche5[(i*5)+2] = prop_sensor[i,15]
        slip_gauche5[(i*5)+3] = prop_sensor[i,16]
        slip_gauche5[(i*5)+4] = prop_sensor[i,17]
        
slip_gauche5 = np.clip(slip_gauche5,-1,1)
slip_droit5 = np.clip(slip_droit5,-1,1)
slip_ref = np.ones(np.shape(slip_gauche5))*0.2

fig, axs = plt.subplots(2, 2)
axs[0,0].set_title('Vitesse des roues')
axs[0,0].plot(prop_sensor[:,0]-prop_cmd_costum[1,0], prop_sensor[:,4], label='Avant Gauche')
axs[0,0].plot(prop_sensor[:,0]-prop_cmd_costum[1,0], prop_sensor[:,5], label='Avant Droite')
axs[0,0].plot(prop_sensor[:,0]-prop_cmd_costum[1,0], prop_sensor[:,6], label='Arrière Gauche')
axs[0,0].plot(prop_sensor[:,0]-prop_cmd_costum[1,0], prop_sensor[:,7], label='Arrière Droite')
axs[0,0].legend()

axs[0,1].set_title('Distance')
axs[0,1].plot(prop_sensor[:,0]-prop_cmd_costum[1,0], prop_sensor[:,3], label='Distance')
axs[0,1].plot(prop_cmd_costum[:,0]-prop_cmd_costum[1,0], prop_cmd_costum[:,3], label='Commandes utilisaturs')
axs[0,1].legend()

axs[1,0].set_title('Courant aux roues')
axs[1,0].plot(prop_sensor[:,0]-prop_cmd_costum[1,0], prop_sensor[:,8], label='Avant Gauche')
axs[1,0].plot(prop_sensor[:,0]-prop_cmd_costum[1,0], prop_sensor[:,9], label='Avant Droite')
axs[1,0].plot(prop_sensor[:,0]-prop_cmd_costum[1,0], prop_sensor[:,10], label='Arrière Gauche')
axs[1,0].plot(prop_sensor[:,0]-prop_cmd_costum[1,0], prop_sensor[:,11], label='Arrière Droite')
axs[1,0].legend()

axs[1,1].set_title('Slip')
#axs[1,1].plot(joy[:,0]-joy[1,0], joy[:,22],label='Brake')
axs[1,1].plot(prop_cmd_costum[:,0]-prop_cmd_costum[1,0], prop_cmd_costum[:,3]/14, label='Commandes utilisaturs')
axs[1,1].plot(slip_time-prop_cmd_costum[1,0], slip_gauche5, label='slip')
#axs[1,1].plot(slip_time-prop_cmd_costum[1,0], slip_droit5,label='slip Droit')
axs[1,1].plot(slip_time-prop_cmd_costum[1,0], slip_ref,label='slip Ref')

axs[1,1].legend()

fig, axs = plt.subplots(1,3)
axs[2].set_title('Acceleration')
#axs[2].plot(imu_acc[:,0]-imu_acc[1,0], imu_acc[:,5], label='Acc')
axs[1].set_title('Vitesse des roues')
axs[1].plot(prop_sensor[:,0]-prop_sensor[1,0], prop_sensor[:,4], label='Avant Gauche')
axs[1].plot(prop_sensor[:,0]-prop_sensor[1,0], prop_sensor[:,5], label='Avant Droite')
axs[1].plot(prop_sensor[:,0]-prop_sensor[1,0], prop_sensor[:,6], label='Arrière Gauche')
axs[1].plot(prop_sensor[:,0]-prop_sensor[1,0], prop_sensor[:,7], label='Arrière Droite')
axs[0].set_title('Distance')
axs[0].plot(prop_sensor[:,0]-prop_sensor[1,0], prop_sensor[:,3])

'''
fig = plt.figure()
plt.plot(mu_est[:,0]-mu_est[1,0],mu_est[:,1])
plt.plot(prop_sensor[:,0]-prop_sensor[1,0], prop_sensor[:,4], label='Avant Gauche')
'''
plt.show()
