#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Float64, Float32MultiArray
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from nav_msgs.msg import Odometry

from math import pi


class node:

    def __init__( self ):
        self.t0 = rospy.get_time()
        self.t_last = self.t0
        self.sub_imu = rospy.Subscriber( '/imu/acceleration', Vector3Stamped, self.readImu );
        self.sub_encd = rospy.Subscriber( '/prop_sensors', Float32MultiArray, self.readEncd );
        self.sub_scan = rospy.Subscriber( '/rtabmap/odom', Odometry, self.readScan );
        self.pub_mu = rospy.Publisher( 'mu_est', Float64, queue_size=1 );
        self.pub_debug = rospy.Publisher( 'debug', Float32MultiArray, queue_size=1 );

        
        self.dt = 0.001
        self.t = 0.0
        self.time_last = 0.0
        self.time_last1 = 0.0
        self.acceleration = 0.0
        self.v = 0.0
        self.tension = 0.0
        self.borne_sup = 0.75
        self.borne_inf = 0.1
        self.order_1 = 10
        self.order_2 = 10
        self.order_3 = 8
        self.x_1 = np.zeros([ 1, self.order_1 ])
        self.y_1 = np.zeros([ 1, self.order_1 ])
        self.x_2 = np.zeros([ 1, self.order_2 ])
        self.y_2 = np.zeros([ 1, self.order_2 ])
        self.x_3 = np.zeros([ 1, self.order_3 ])
        self.y_3 = np.zeros([ 1, self.order_3 ])
        self.rho = 0.0
        self.rho_last = 0.0
        self.rho_plot = 0.0
        self.s_x = 0.0
        self.s_x_last = 0.0
        self.s_x_plot = 0.0
        self.omega_R = 0.0
        self.a_last = 0.0
        self.b_last = 0.0
        self.a_est_1 = 0.0
        self.a_est_2 = 0.0
        self.a_est_3 = 0.0
        self.b_est_1 = 0.0
        self.b_est_2 = 0.0
        self.b_est_3 = 0.0
        self.n_1 = 0
        self.n_2 = 0
        self.n_3 = 0
        self.s_x_vector = 0.0
        self.rho_vector = 0.0
        self.s_x_vector_plot = 0.0
        self.rho_vector_plot = 0.0
        self.length_s_x = 0.0
        self.length_s_x_last = 0.0
        self.mu_est_1 = 0.0
        self.mu_est_3 = 0.0
        self.mu_est_1_vector = 0.0
        self.mu_est_3_vector = 0.0
        self.t_vector = 0.0
        self.t_1_vector = 0.0
        self.t_3_vector = 0.0 
        self.mu_estime = 0.0
        self.mu_est_1_last = 0.0
        self.mu_est_3_last = 0.0
        self.mu_estime_vector = 0.0
        self.position_vector = 0.0
        self.v_vector = 0.0
        self.a_vector = 0.0
        self.tension_vector = 0.0
        self.k1 = 0.0
        self.k2 = 0.0
        self.manoeuvre = 0.0
        self.a = 0.0
        self.acc_vector = np.zeros([ 1, 20 ])
        self.acc_filtered_last = 0.0
        self.acc_filtered_last2 = 0.0
        self.omega_vector = np.zeros([ 1, 5 ])
        self.scan_vector = np.zeros([ 1, 5 ])
        self.vit_vector = np.zeros([ 1, 20 ])
        self.m = 6.828 
        self.F_n = 0.57*( self.m*9.81 )
        self.vit_last = 0.0
        self.position_last = 0.0
        self.vit_scan_filtered =0.0
        self.fc =1.0
        self.vit_imu_last = 0.0
        self.vit_imu_filtered_last = 0.0
        self.vit_scan_filtered_last = 0.0 
        self.acc_imu_lp_last = 0.0
        self.acc_imu_hp_last = 0.0
        self.acc_x_raw_last = 0.0
        self.vit_last = 0.0
        self.vit_comp = 0.0
        
 

        
        
	
#%% Callbacks     
    def readImu( self, imu ):
        self.acc_x_raw = imu.vector.x

        self.acc_imu_lp = self.lowPassFilter(1.0/400.0,self.acc_x_raw,self.acc_imu_lp_last, 1.0)
        self.acc_imu_hp = self.highPassFilter(1.0/400.0,self.acc_imu_lp,self.acc_imu_lp_last,self.acc_imu_hp_last, 1.0)
        
        t_now = rospy.get_time()
        dt = t_now-self.t_last
        self.vit_imu = self.timeIntegration(dt, self.acc_x_raw,self.acc_x_raw_last, self.vit_imu_last)
        self.vit_imu_filtered = self.highPassFilter(dt,self.vit_imu,self.vit_imu_last,self.vit_imu_filtered_last, self.fc)
        
        self.vit_comp = self.complementaryFilter()
	
        #Memory
        self.acc_imu_lp_last = self.acc_imu_lp
        self.acc_imu_hp_last = self.acc_imu_hp
        self.acc_x_raw_last = self.acc_x_raw
        
        self.vit_imu_last = self.vit_imu
        self.vit_imu_filtered_last = self.vit_imu_filtered

	self.t_last = t_now


    def readEncd( self, encd ):
        self.omega_raw = encd.data[1]
        self.omega_vector = np.insert( self.omega_vector, 0, self.omega_raw )
        self.omega_vector = self.omega_vector[ :-1 ]
        self.omega_filtered = self.FIR5( self.omega_vector )
        self.core()
        
    def readScan( self, data ):
        self.pos_x = data.pose.pose.position.x      
        self.pos_y = data.pose.pose.position.y
        self.position = np.sqrt( self.pos_x**2 + self.pos_y**2 )
        self.vit_scan = self.timeDerivation(1.0/10.0, self.position, self.position_last)
        self.vit_scan_filtered = self.lowPassFilter(1.0/10.0,self.vit_scan,self.vit_scan_filtered_last, self.fc)
	
        #Memory
        self.position_last = self.position
        self.vit_scan_filtered_last = self.vit_scan_filtered
	
        
#%% FILTERS
    def complementaryFilter(self):
        data = self.vit_imu_filtered+self.vit_scan_filtered
    
    	return data
    	
    def FIR5( self, signal ):
	FIR5_coefficient = np.array([ 0.3, 0.265, 0.175, 0.145, 0.115 ])
	signal_filtered = np.matmul( FIR5_coefficient, np.transpose( signal ))
	return signal_filtered
	
    def FIR20( order, signal ):
    	FIR20_coefficient = np.array([ 0.0057, 0.0103, 0.0150, 0.0196, 0.0243, 0.0290, 0.0336, 0.0383, 0.0429, 0.0476, 0.0523, 0.0569, 0.0616, 0.0662, 0.0709, 0.0756, 0.0802, 0.0849, 0.0895, 0.0942])
    	signal_filtered = np.matmul( FIR20_coefficient, np.transpose( signal ))
	return signal_filtered
	
    def lowPassFilter( self, dt, signal, signal_last, fc ):
	RC = 1.0/( 2.0*np.pi*fc )
	alpha = dt/( RC+dt )
	signal_filtered = ( 1.0-alpha )*signal_last + alpha*signal
	return signal_filtered
 
    def highPassFilter( self, dt, signal, signal_last, signal_last_f, fc ):
	RC = 1.0/( 2.0*np.pi*fc )
	beta = RC/( RC+dt )
	signal_filtered = beta*signal_last_f + beta*(signal-signal_last)
	return signal_filtered
	
#%% INTEGRATION
    def timeDerivation( self, dt, data, data_last ):
	d_data_dt = ( data-data_last )/dt
	return d_data_dt
	
    def timeIntegration( self, dt, data_dot,data_dot_last, data_last ):
	data = (data_dot+data_dot_last)/2*dt+data_last

	return data
                
#%% Fonctions

    def linearParameters( self, rho, rho_last, s_x, s_x_last, a_last ):
	if s_x == s_x_last:
		a = a_last
	else:
		a = ( rho-rho_last )/( s_x-s_x_last )
	
	b = rho-a*s_x

	return a, b

    def leastSquares( self, x, y, n ):
	sum_x_square = ( np.sum( x**2 ))
	sum_x = ( np.sum( x ))
	sum_y = ( np.sum( y ))
	sum_x_y = float( np.matmul( np.transpose( x ), y))

	A = np.array([[ sum_x_square, sum_x ], [ sum_x, n ]])
	B = np.array([[ sum_x_y ], [ sum_y ]])
	
	if sum_x_square == 0.0:
		A_inv = np.linalg.pinv( A )
	else:
		A_inv = np.linalg.pinv( A )

	C = np.matmul( A_inv, B )

	a_est = C[ 0 ]
	b_est = C[ 1 ]

	return a_est, b_est

    def slipRatio( self, omega_R, vitesse ):
	if (np.absolute( omega_R ) <= 0.001 or np.absolute( vitesse ) <= 0.001):
		s_x = 0.0
	else:
		s_x = ( omega_R-vitesse )/np.maximum( np.absolute( omega_R ), np.absolute( vitesse )) 
	
	if ( s_x < -1.0 ):
		s_x = -1.0
	if ( s_x > 1.0 ):
		s_x = 1.0
			
	return s_x

    def compute_rho( self, acceleration ):
	F_x = self.m*acceleration
	rho = F_x/self.F_n
	
	if ( rho < -1.0 ):
		rho = -1.0
	if ( rho > 1.0 ):
		rho = 1.0
		
	return rho

#%% Core Node

    def core( self ):
        self.s_x = self.slipRatio(self.omega_filtered, self.vit_comp*1.55)
        self.rho = self.compute_rho(self.acc_imu_lp)
        time = rospy.get_time() - self.t0
	
#	( a, b ) = self.linearParameters( self.rho, self.rho_last, self.s_x, self.s_x_last, self.a_last )

        if self.rho <= -0.15:

		if ( -1.0 < self.s_x ) and ( self.s_x < -self.borne_inf ) and ( abs( self.vit_comp*1.55 ) > 0.1 ):
			self.rho_plot = self.rho
			self.s_x_plot = self.s_x
			self.n_1 = self.n_1+1
			self.x_1 = np.insert( self.x_1, 0, self.s_x )
			self.x_1 = self.x_1[ :-1 ]
			self.x_1 = np.transpose([ self.x_1 ])

			self.y_1 = np.insert( self.y_1, 0, self.rho )
			self.y_1 = self.y_1[ :-1 ]
			self.y_1 = np.transpose([ self.y_1 ])

			( self.a_est_1, self.b_est_1 ) = self.leastSquares( self.x_1, self.y_1, self.n_1 )
			alpha = 1.0 #0.975
        		self.mu_est_1 = self.y_1[np.nonzero(self.y_1)].mean() # alpha*np.mean( self.y_1 ) + ( 1-alpha )*( self.a_est_1*self.borne_inf+self.b_est_1 )
        		self.mu_est_1_vector = np.append( self.mu_est_1_vector, self.mu_est_1 )	
        		t_1 = time
        		self.t_1_vector = np.append( self.t_1_vector, t_1 )
        				
		if ( -self.borne_inf < self.s_x ) and ( self.s_x <= 0.0 ):# and np.abs( self.omega_R ) >= 0.001:
#			self.y_1 = np.full(( self.order_1, 1 ), -self.mu_estime )
#			self.y_1[ 5:self.order_1 ] = 0.0
#			self.y_3 = np.full(( self.order_1, 1 ), self.mu_estime )
			self.rho_plot = self.rho
			self.s_x_plot = self.s_x
			self.n_2 = self.n_2+1
			self.x_2 = np.insert( self.x_2, 0, self.s_x )
			self.x_2 = self.x_2[ :-1 ]
			self.x_2 = np.transpose([ self.x_2 ])

			self.y_2 = np.insert( self.y_2, 0, self.rho )
			self.y_2 = self.y_2[ :-1 ]
			self.y_2 = np.transpose([ self.y_2 ])

			( self.a_est_2, self.b_est_2 ) = self.leastSquares( self.x_2, self.y_2, self.n_2 )
						
        if self.rho >= 0.15:		
		if ( 0.0 <= self.s_x ) and ( self.s_x <= self.borne_inf ):# and np.abs( self.omega_R ) >= 0.001:
			self.rho_plot = self.rho
			self.s_x_plot = self.s_x
			self.n_2 = self.n_2+1
			self.x_2 = np.insert( self.x_2, 0, self.s_x )
			self.x_2 = self.x_2[ :-1 ]
			self.x_2 = np.transpose([ self.x_2 ])

			self.y_2 = np.insert( self.y_2, 0, self.rho )
			self.y_2 = self.y_2[ :-1 ]
			self.y_2 = np.transpose([ self.y_2 ])

			( self.a_est_2, self.b_est_2 ) = self.leastSquares( self.x_2, self.y_2, self.n_2 )
						
		if ( self.borne_inf < self.s_x ) and ( self.s_x < self.borne_sup ) and ( abs( self.vit_comp*1.55 ) > 0.1 ):
			self.rho_plot = self.rho
			self.s_x_plot = self.s_x
			self.n_3 = self.n_3+1
			self.x_3 = np.insert( self.x_3, 0, self.s_x )
			self.x_3 = self.x_3[ :-1 ]
			self.x_3 = np.transpose([ self.x_3 ])

			self.y_3 = np.insert( self.y_3, 0, self.rho )
			self.y_3 = self.y_3[ :-1 ]
			self.y_3 = np.transpose([ self.y_3 ])

			( self.a_est_3, self.b_est_3 ) = self.leastSquares( self.x_3, self.y_3, self.n_3 )
			#alpha = 1.0 # 0.975
        		self.mu_est_3 = self.y_3[np.nonzero(self.y_3)].mean() #  alpha*np.mean( self.y_3 ) + ( 1-alpha )*( self.a_est_3*self.borne_inf+self.b_est_3 )
        		self.mu_est_3_vector = np.append( self.mu_est_3_vector, self.mu_est_3 )
        		t_3 = time
        		self.t_3_vector = np.append( self.t_3_vector, t_3 )
        
        if ( self.mu_est_1 != self.mu_est_1_last ):
        	self.mu_estime = self.mu_est_1;
        	self.k1 = self.k1+1
        	if ( self.mu_estime <= -1.0 ):
        		self.mu_estime = -1.0
  
        if ( self.mu_est_3 != self.mu_est_3_last ):
        	self.mu_estime = self.mu_est_3
        	self.k2 = self.k2+1
        	if ( self.mu_estime >= 1.0 ):
        		self.mu_estime = 1.0
				
	# Memory
	self.s_x_vector = np.append( self.s_x_vector, self.s_x )
	self.rho_vector = np.append( self.rho_vector, self.rho )
	self.t_vector = np.append( self.t_vector, time )
	self.mu_estime_vector = np.append( self.mu_estime_vector, self.mu_estime )
	print(self.mu_estime)
	self.pub_mu.publish( self.mu_estime )  
	msg = Float32MultiArray()
	msg.data = [self.omega_filtered,self.vit_comp*1.55,self.s_x,self.rho,self.mu_estime,self.acc_imu_lp,self.vit_imu_filtered,self.vit_scan_filtered]
	self.pub_debug.publish( msg )
    
		
	self.length_s_x_last = self.length_s_x
	self.mu_est_1_last = self.mu_est_1
	self.mu_est_3_last = self.mu_est_3
	
if __name__== '__main__':
	rospy.init_node( 'node', anonymous = False )
	node = node()
	rospy.spin()
	
