#!/usr/bin/env python
import rospy
import numpy as np
import xlrd
import time
import math
import sys as s
import os

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

### VALUE ITERATION 
import xmaxx_vi_controller as xvi
import xmaxx_ttc_controller as xttc


#########################################
class racecar_controller(object):

    #######################################    
    def __init__(self):

        # Init subscribers  
        self.sub_joy   = rospy.Subscriber("joy", Joy , self.read_joy , queue_size=1)
        self.sub_imu   = rospy.Subscriber("/imu/data", Imu , self.read_imu , queue_size=1)
        self.sub_sensors    = rospy.Subscriber("prop_sensors", Float32MultiArray , self.read_sensors , queue_size=1)
        
        #front_scan_detection
        self.front_scan_point = 15
        self.sub = rospy.Subscriber("/scan", LaserScan, self.callback_front_scan)
        self.min_distance = 20
        
        # Init publishers
        self.pub_cmd    = rospy.Publisher("prop_cmd", Twist , queue_size=1)
        self.pub_cmd_costum    = rospy.Publisher("prop_cmd_costum", Float32MultiArray, queue_size=1)
        self.prop_cmd_costum_array = [0,0,0,0,0,0,0,0,0]

        # Iniy
        self.first_brake_loop = 1
        self.pos_last = 0.0
        self.v_last = 0.0 
	
        # Gain
        self.max_vel  = rospy.get_param('~max_vel',   6.0) # Max linear velocity (m/s)
        self.max_volt = rospy.get_param('~max_volt',  8)   # Max voltage is set at 6 volts   
        self.maxStAng = rospy.get_param('~max_angle', 24)  # Supposing +/- 40 degrees max for the steering angle
        self.cmd2rad   = self.maxStAng*2*3.1416/360
        
        #4wheel free
        self.roue_avant_droite = 0
        self.roue_avant_gauche = 0
        self.roue_arriere_droite = 0
        self.roue_arriere_gauche = 0
        self.steering = 0
        
        self.keep_moving = True
        
        #IMU
        self.acceleration_vehicule = 0
        
        # CMD input
        # Always set first to No velocity controler + ABS Brake
        self.driving_states = {
        	'ALLWHEEL': [4, 0],
        	'FRONTWHEEL': [5, 0],
        	'FREE4WHEEl': [6, 0],
        	'NOTHING': [0,0],
        	} 
        self.driving_state = self.driving_states['NOTHING']
        self.driving_state_name = 'NOTHING'
        
        self.control_states = {
        	'FIXE': [0,0],
        	'MSD': [1, 0],
        	'TTC': [2, 0],
        	'VI': [3, 0],
        	} 
        self.control_state = self.control_states['FIXE']
        self.control_state_name = 'FIXE'
        
        self.brake_states = {
        	'BRAKE': [11, 0.5],
        	'ABSC': [12, 0.5],
        	'ABSPID': [13, 0.5],
        	'ABSSMC': [14, 0.5],
        	}   
        self.brake_state = self.brake_states['ABSC'] 
        self.brake_state_name = 'ABSC'    
        
        
        # References Inputs
        self.right_joy = 0 
        self.right_joy_steer = 0
        self.left_joy  = 0
        self.left_button  = 0
        self.right_button = 0
        self.A_button = 0
        self.B_button = 0
        self.X_button = 0
        self.Y_button = 0
        self.back_button = 0
        self.start_button = 0
        
        self.left_trigger = 0
        self.right_trigger = 0
        self.steering_offset = -0.
        
        # Ouput commands
        self.propulsion_cmd = 0  # Command sent to propulsion
        self.arduino_mode   = 0  # Control mode
        self.steering_cmd   = 0  # Command sent to the steering servo
        self.pp_delta       = 0

        # Sensing info 
        self.delta = 0
        self.X =0

        # 
        self.v_i = 1
        self.u   = 26
                
        # Distance decision
        self.max_vitesse = 1.0 
        self.brake_flag = False
	self.brake_flag_prio_1 = False
        self.d = 0.1
        self.slip = 0.3
	

	road = 'CobblestoneWet'
	#VI
	self.vi_cmd = np.array([0,0])
	self.vi_controller = xvi.xmaxxvi_controller(road = road)
	while self.vi_controller.ready_to_use_flag == False:
		pass
	#TTC
	self.ttc_cmd = np.array([0,0])
	self.ttc_controller = xttc.xmaxxttc_controller(road = road)
	while self.ttc_controller.ready_to_use_flag == False:
		pass
	self.time_brake = time.time()
        
        # Timer
        self.dt         = 0.02
        self.timer      = rospy.Timer( rospy.Duration( self.dt ), self.timed_controller )
        self.pos_ini =0.0
        self.pos_ref = 4
        self.pos_offset = 4
        self.v_ini =0.0
        
        #Controler States
        self.position_vehicule = 0.
        self.vitesse_vehicule = 0.
        self.acceleration_vehicule = 0.
         



    #######################################
    def callback_front_scan(self,data):
        front_scan_distances = np.asarray(data.ranges)[(len(data.ranges)/2)-(self.front_scan_point/2):(len(data.ranges)/2)+(self.front_scan_point/2)]
        self.min_distance = np.min(front_scan_distances)
	if self.min_distance == np.inf:
		self.min_distance = 0.5
        self.position_vehicule = self.min_distance
	
    def timed_controller(self, timer):
        self.send_arduino_costum()

    def read_imu(self, imu_data):
        self.acceleration_vehicule = imu_data.linear_acceleration.x




    #######################################
    def read_joy(self, joy_msg):
        #Read received references
        self.back_button = joy_msg.buttons[6]
        self.start_button = joy_msg.buttons[7]
        self.A_button = joy_msg.buttons[0]
        self.B_button = joy_msg.buttons[1]
        self.X_button = joy_msg.buttons[2]
        self.Y_button = joy_msg.buttons[3]
        self.left_button  = joy_msg.buttons[4]
        self.right_button  = joy_msg.buttons[5]
        self.R3 = joy_msg.buttons[9]
       	
        self.right_trigger = (((joy_msg.axes[4]*-1)+1)/2)
        self.left_trigger = (((joy_msg.axes[5]*-1)+1)/2)
        self.dpad_left_right = joy_msg.axes[6]
        self.dpad_up_down = joy_msg.axes[7]
        
        self.right_joy  = joy_msg.axes[3] # Up-down Right joystick 
        if (self.right_joy >= -0.1) and (self.right_joy <= 0.1):
            self.right_joy = 0 #Pour enlever les glitchs de la manette de xbox    
        self.left_joy  = joy_msg.axes[1] # Up-down Right joystick 
        if (self.left_joy >= -0.1) and (self.left_joy <= 0.1):
            self.left_joy = 0 #Pour enlever les glitchs de la manette de xbox
        self.right_joy_steer = joy_msg.axes[2]    # Left-right left joystick
        if (self.right_joy_steer>= -0.1) and (self.right_joy_steer <= 0.1):
            self.right_joy_steer = 0 #Pour enlever les glitchs de la manette de xbox
        self.right_joy_steer= 0.4*(self.right_joy_steer + self.steering_offset)
        
        # apply brakes
        if self.R3 == 1:
            self.brake_flag_prio_1 = True
        
        #Change Le controler
        if  self.start_button == 1:
            	if self.A_button == 1:
            		self.control_state = self.control_states['FIXE']
            		self.control_state_name = 'FIXE'
            	if self.B_button == 1:
            		self.control_state = self.control_states['MSD']
            		self.control_state_name = 'MSD'
            	if self.X_button == 1:
            		self.control_state = self.control_states['TTC']
            		self.control_state_name = 'TTC'
            	if self.Y_button == 1:
                  self.control_state = self.control_states['VI']
                  self.control_state_name = 'VI'    
            	print('Controller is : ',self.control_state_name)
        
        #Change Le Mode de Frein
        if  self.start_button == 1:
        	if self.dpad_left_right >= 0.8:
        		self.brake_state = self.brake_states['ABSC'] 
        		self.brake_state_name = 'ABSC'
        	if self.dpad_left_right <= -0.8:
        		self.brake_state = self.brake_states['ABSPID']
        		self.brake_state_name = 'ABSPID'
        	if self.dpad_up_down >= 0.8:
        		self.brake_state = self.brake_states['ABSSMC'] 
        		self.brake_state_name = 'ABSSMC'
        	if self.dpad_up_down <= -0.8:
        		self.brake_state = self.brake_states['BRAKE'] 
        		self.brake_state_name = 'BRAKE'
        	print('Brake Mode is : ',self.brake_state_name)        	
        	
        #Change le mode de conduite
        if  self.back_button == 1:
        	if self.A_button == 1:
        		self.driving_state = self.driving_states['FRONTWHEEL']
        		self.driving_state_name = 'FRONTWHEEL' 
        	if self.B_button == 1:
        		self.driving_state = self.driving_states['ALLWHEEL']
        		self.driving_state_name = 'ALLWHEEL'
        	if self.X_button == 1:
        		self.driving_state = self.driving_states['FREE4WHEEl']
        		self.driving_state_name = 'FREE4WHEEl'
        	if self.Y_button == 1:
        		self.driving_state = self.driving_states['NOTHING']
        		self.driving_state_name = 'NOTHING'
        	print('Driving Mode is : ',self.driving_state_name)

    def read_sensors(self, arduino):
        #Read received references
        self.pos_ini = arduino.data[0]
        self.v_ini   = arduino.data[1]
        self.vitesse_vehicule = arduino.data[2] 

    def driving_function(self):
        if self.left_button == 1:
            self.steering = self.right_joy_steer
            if self.driving_state_name == 'FRONTWHEEL':
                self.prop_cmd_costum_array[0] = self.driving_state[0]
                self.roue_avant_droite = self.left_joy
                self.roue_avant_gauche = self.left_joy
                self.roue_arriere_droite = 0
                self.roue_arriere_gauche = 0
			
            if self.driving_state_name == 'ALLWHEEL':
                self.prop_cmd_costum_array[0] = self.driving_state[0]
                self.roue_avant_droite = self.left_joy
                self.roue_avant_gauche = self.left_joy
                self.roue_arriere_droite = self.left_joy
                self.roue_arriere_gauche = self.left_joy
			
            if self.driving_state_name == 'FREE4WHEEl':
                self.prop_cmd_costum_array[0] = self.driving_state[0]
                self.roue_avant_droite = self.right_joy
                self.roue_avant_gauche = self.left_joy
                self.roue_arriere_droite = self.right_trigger
                self.roue_arriere_gauche = self.left_trigger
			
            if self.driving_state_name == 'NOTHING':
                self.prop_cmd_costum_array[0] = self.driving_state[0]
                self.roue_avant_droite = 0
                self.roue_avant_gauche = 0
                self.roue_arriere_droite = 0
                self.roue_arriere_gauche = 0
        else:
            self.roue_avant_droite = 0
            self.roue_avant_gauche = 0
            self.roue_arriere_droite = 0
            self.roue_arriere_gauche = 0	

    def controler_function(self):
	if self.brake_flag_prio_1 == False:

		if self.control_state_name == 'FIXE':
		    if self.pos_ini > self.pos_ref + self.pos_offset:
			self.slip = 0.2
		        self.brake_flag = True

		if self.control_state_name == 'MSD':
		    if self.min_distance < 2.0 and self.vitesse_vehicule > 0.1:
			self.slip = 0.2
		        self.brake_flag = True

		if self.control_state_name == 'TTC':
		    self.ttc_cmd = self.ttc_controller.commands(self.min_distance, self.vitesse_vehicule, self.acceleration_vehicule)
		    print('Distance = ', self.min_distance, 'Vitesse = ', self.vitesse_vehicule)   
		    if self.ttc_cmd[1] >= 0.5:
			print('BRAKE: ', self.ttc_cmd[0])
			self.slip = np.abs(self.ttc_cmd[0])
			self.slip = 0.2
			self.brake_flag = True
			self.time_brake = time.time()
		    else:
			print('PAS BRAKE')
			if time.time() - self.time_brake > 1.0:
				self.brake_flag = False

		if self.control_state_name == 'VI':
		    self.vi_cmd = self.vi_controller.commands(self.min_distance, self.vitesse_vehicule)
		    print('Distance = ', self.min_distance, 'Vitesse = ', self.vitesse_vehicule)    
		    if self.vi_cmd[1] >= 0.5:
			self.slip = np.abs(self.vi_cmd[0])
			self.slip = 0.2
			print('BRAKE: ', self.vi_cmd[0])
			self.brake_flag = True
			self.time_brake = time.time()
		    else:
			print('PAS BRAKE')
			if time.time() - self.time_brake > 1.0:
				self.brake_flag = False
	else:
		self.brake_flag = True
		
    	
    def brakes_function(self):
    	#Apply Brakes
        if self.brake_flag == True:
            self.prop_cmd_costum_array[0] = self.brake_state[0]
            self.roue_avant_droite = 0
            self.roue_avant_gauche = 0
            self.roue_arriere_droite = 0
            self.roue_arriere_gauche = 0
	#Reset Brakes
        if self.left_button == 1 and self.right_button == 1 and (self.brake_flag == True or self.brake_flag_prio_1 == True):
            self.brake_flag = False
	    self.brake_flag_prio_1 = False
            self.pos_ref = self.pos_ini
    	
    			
    	
    def send_arduino_costum(self):
        cmd_prop = Float32MultiArray()
	
        self.driving_function()
        self.controler_function()
        self.brakes_function()
	
        self.prop_cmd_costum_array[1] = self.steering
        self.prop_cmd_costum_array[2] = self.roue_avant_droite
        self.prop_cmd_costum_array[3] = self.roue_avant_gauche
        self.prop_cmd_costum_array[4] = self.roue_arriere_droite
        self.prop_cmd_costum_array[5] = self.roue_arriere_gauche
        self.prop_cmd_costum_array[6] = self.acceleration_vehicule
        self.prop_cmd_costum_array[7] = self.slip
	self.prop_cmd_costum_array[8] = self.vi_cmd[0]	
        cmd_prop.data = self.prop_cmd_costum_array
        self.pub_cmd_costum.publish(cmd_prop)

	
#########################################

if __name__ == '__main__':

    rospy.init_node('CONTROLLER',anonymous=False)
    node = racecar_controller()
    rospy.spin()
