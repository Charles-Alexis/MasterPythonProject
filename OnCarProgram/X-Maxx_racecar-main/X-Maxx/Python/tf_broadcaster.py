#!/usr/bin/env python
"""
Created on Wed Dec  2 18:02:38 2020

@author: Will Therrien
"""

  
import rospy
 
# Because of transformations
import tf_conversions

import tf2_ros
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import numpy as np

#########################################
class Broadcaster(object):

    #######################################
    #-----------Initialization------------#
    ####################################### 
    
    def __init__(self):
        
        # Init subscribers          
        self.sub_cmd     = rospy.Subscriber('prop_cmd', Twist, self.read_cmd)
        self.sub_sensors = rospy.Subscriber('prop_sensors', Float32MultiArray, self.read_sensors)

        # Init clock-timed callback
        self.dt         = 0.15
        self.timer      = rospy.Timer( rospy.Duration( self.dt ), self.publish)
        
        # Init states and inputs
        self.x     = 0
        self.y     = 0
        self.theta = 0
        self.v     = 0
        self.delta = 0
        self.pos   = 0
        
        #Init memory
        self.x_last     = 0
        self.y_last     = 0
        self.theta_last = 0 
        self.pos_last   = 0
        self.data_last = 0
        
        #Params
        self.L = 0.345
        self.width = 0.225377
        
        
    def lowPassFilter(self, data):
        a = 0.75
        data_filt = self.data_last+a*(data-self.data_last)
        self.data_last = data_filt
        return data_filt
       
    def read_cmd(self, prop):
        #Read steering cmd
        self.delta = -prop.angular.z
        
    def read_sensors(self, sensors):
        #Read encoder speed from arduino
        self.pos = sensors.data[0]
        self.v_raw = (self.pos-self.pos_last)/self.dt
        self.v     = self.lowPassFilter(self.v_raw)
        self.pos_last = self.pos
        
    def publish(self, timer):
        br = tf2_ros.TransformBroadcaster()

        t1 = TransformStamped()  
        t1.header.stamp = rospy.Time.now()
        t1.header.frame_id = "base_link"
        t1.child_frame_id  = "body"
        t1.transform.translation.x = 0.0
        t1.transform.translation.y = 0.0
        t1.transform.translation.z = 0.05
        q1 = tf_conversions.transformations.quaternion_from_euler(0, 0, 0)
        t1.transform.rotation.x = q1[0]
        t1.transform.rotation.y = q1[1]
        t1.transform.rotation.z = q1[2]
        t1.transform.rotation.w = q1[3]
        
        
        t2 = TransformStamped()  
        t2.header.stamp = rospy.Time.now()
        t2.header.frame_id = "body"
        t2.child_frame_id  = "velodyne"
        t2.transform.translation.x = 0.146518
        t2.transform.translation.y = 0.0
        t2.transform.translation.z = 0.31
        q2 = tf_conversions.transformations.quaternion_from_euler(0, 0, 0)
        t2.transform.rotation.x = q2[0]
        t2.transform.rotation.y = q2[1]
        t2.transform.rotation.z = q2[2]
        t2.transform.rotation.w = q2[3]

        
        t3 = TransformStamped() 
        t3.header.stamp = rospy.Time.now()
        t3.header.frame_id = "body"
        t3.child_frame_id  = "left_wheels"
        t3.transform.translation.x = 0.0
        t3.transform.translation.y = -self.width/2
        t3.transform.translation.z = 0.0
        q3 = tf_conversions.transformations.quaternion_from_euler(np.pi/2, 0, 0)
        t3.transform.rotation.x = q3[0]
        t3.transform.rotation.y = q3[1]
        t3.transform.rotation.z = q3[2]
        t3.transform.rotation.w = q3[3]
        
        t4 = TransformStamped()  
        t4.header.stamp = rospy.Time.now()
        t4.header.frame_id = "left_wheels"
        t4.child_frame_id  = "left_rear_wheel"
        t4.transform.translation.x = 0.0
        t4.transform.translation.y = 0.0
        t4.transform.translation.z = 0.0
        q4 = tf_conversions.transformations.quaternion_from_euler(0, 0, -self.pos/0.054)
        t4.transform.rotation.x = q4[0]
        t4.transform.rotation.y = q4[1]
        t4.transform.rotation.z = q4[2]
        t4.transform.rotation.w = q4[3]
        
        t5 = TransformStamped()  
        t5.header.stamp = rospy.Time.now()
        t5.header.frame_id = "left_wheels"
        t5.child_frame_id  = "left_front_wheel_steer"
        t5.transform.translation.x = self.L
        t5.transform.translation.y = 0.0
        t5.transform.translation.z = 0.0
        q5 = tf_conversions.transformations.quaternion_from_euler(0, self.delta, 0)
        t5.transform.rotation.x = q5[0]
        t5.transform.rotation.y = q5[1]
        t5.transform.rotation.z = q5[2]
        t5.transform.rotation.w = q5[3]  

        t6 = TransformStamped()  
        t6.header.stamp = rospy.Time.now()
        t6.header.frame_id = "left_front_wheel_steer"
        t6.child_frame_id  = "left_front_wheel"
        t6.transform.translation.x = 0.0
        t6.transform.translation.y = 0.0
        t6.transform.translation.z = 0.0
        q6 = tf_conversions.transformations.quaternion_from_euler(0, 0, -self.pos/0.054)
        t6.transform.rotation.x = q6[0]
        t6.transform.rotation.y = q6[1]
        t6.transform.rotation.z = q6[2]
        t6.transform.rotation.w = q6[3]  
        
        t7 = TransformStamped()  
        t7.header.stamp = rospy.Time.now()
        t7.header.frame_id = "left_front_wheel"
        t7.child_frame_id  = "left_front_mag"
        t7.transform.translation.x = 0.0
        t7.transform.translation.y = 0.0
        t7.transform.translation.z = 0.0
        q7 = tf_conversions.transformations.quaternion_from_euler(0, 0, 0)
        t7.transform.rotation.x = q7[0]
        t7.transform.rotation.y = q7[1]
        t7.transform.rotation.z = q7[2]
        t7.transform.rotation.w = q7[3]

        t8 = TransformStamped()  
        t8.header.stamp = rospy.Time.now()
        t8.header.frame_id = "left_rear_wheel"
        t8.child_frame_id  = "left_rear_mag"
        t8.transform.translation.x = 0.0
        t8.transform.translation.y = 0.0
        t8.transform.translation.z = 0.0
        q8 = tf_conversions.transformations.quaternion_from_euler(0, 0, 0)
        t8.transform.rotation.x = q8[0]
        t8.transform.rotation.y = q8[1]
        t8.transform.rotation.z = q8[2]
        t8.transform.rotation.w = q8[3]
        
        t9 = TransformStamped() 
        t9.header.stamp = rospy.Time.now()
        t9.header.frame_id = "body"
        t9.child_frame_id  = "right_wheels"
        t9.transform.translation.x = 0.0
        t9.transform.translation.y = self.width/2
        t9.transform.translation.z = 0.0
        q9 = tf_conversions.transformations.quaternion_from_euler(-np.pi/2, 0, 0)
        t9.transform.rotation.x = q9[0]
        t9.transform.rotation.y = q9[1]
        t9.transform.rotation.z = q9[2]
        t9.transform.rotation.w = q9[3]
        
        t10 = TransformStamped()  
        t10.header.stamp = rospy.Time.now()
        t10.header.frame_id = "right_wheels"
        t10.child_frame_id  = "right_rear_wheel"
        t10.transform.translation.x = 0.0
        t10.transform.translation.y = 0.0
        t10.transform.translation.z = 0.0
        q10 = tf_conversions.transformations.quaternion_from_euler(0, 0, self.pos/0.054)
        t10.transform.rotation.x = q10[0]
        t10.transform.rotation.y = q10[1]
        t10.transform.rotation.z = q10[2]
        t10.transform.rotation.w = q10[3]
        
        t11 = TransformStamped()  
        t11.header.stamp = rospy.Time.now()
        t11.header.frame_id = "right_wheels"
        t11.child_frame_id  = "right_front_wheel_steer"
        t11.transform.translation.x = self.L
        t11.transform.translation.y = 0.0
        t11.transform.translation.z = 0.0
        q11 = tf_conversions.transformations.quaternion_from_euler(0, -self.delta, 0)
        t11.transform.rotation.x = q11[0]
        t11.transform.rotation.y = q11[1]
        t11.transform.rotation.z = q11[2]
        t11.transform.rotation.w = q11[3]  

        t12 = TransformStamped()  
        t12.header.stamp = rospy.Time.now()
        t12.header.frame_id = "right_front_wheel_steer"
        t12.child_frame_id  = "right_front_wheel"
        t12.transform.translation.x = 0.0
        t12.transform.translation.y = 0.0
        t12.transform.translation.z = 0.0
        q12 = tf_conversions.transformations.quaternion_from_euler(0, 0, self.pos/0.054)
        t12.transform.rotation.x = q12[0]
        t12.transform.rotation.y = q12[1]
        t12.transform.rotation.z = q12[2]
        t12.transform.rotation.w = q12[3]  
        
        t13 = TransformStamped()  
        t13.header.stamp = rospy.Time.now()
        t13.header.frame_id = "right_front_wheel"
        t13.child_frame_id  = "right_front_mag"
        t13.transform.translation.x = 0.0
        t13.transform.translation.y = 0.0
        t13.transform.translation.z = 0.0
        q13 = tf_conversions.transformations.quaternion_from_euler(0, 0, 0)
        t13.transform.rotation.x = q13[0]
        t13.transform.rotation.y = q13[1]
        t13.transform.rotation.z = q13[2]
        t13.transform.rotation.w = q13[3]

        t14 = TransformStamped()  
        t14.header.stamp = rospy.Time.now()
        t14.header.frame_id = "right_rear_wheel"
        t14.child_frame_id  = "right_rear_mag"
        t14.transform.translation.x = 0.0
        t14.transform.translation.y = 0.0
        t14.transform.translation.z = 0.0
        q14 = tf_conversions.transformations.quaternion_from_euler(0, 0, 0)
        t14.transform.rotation.x = q14[0]
        t14.transform.rotation.y = q14[1]
        t14.transform.rotation.z = q14[2]
        t14.transform.rotation.w = q14[3]
        

        
        

        
        br.sendTransform(t1)
        br.sendTransform(t2)
        br.sendTransform(t3)
        br.sendTransform(t4)
        br.sendTransform(t5)
        br.sendTransform(t6)
        br.sendTransform(t7)
        br.sendTransform(t8)
        br.sendTransform(t9)
        br.sendTransform(t10)
        br.sendTransform(t11)
        br.sendTransform(t12)
        br.sendTransform(t13)
        br.sendTransform(t14)


  
if __name__ == '__main__':
    rospy.init_node('TF_Broadcaster',anonymous=False)
    node = Broadcaster()
    rospy.spin()
