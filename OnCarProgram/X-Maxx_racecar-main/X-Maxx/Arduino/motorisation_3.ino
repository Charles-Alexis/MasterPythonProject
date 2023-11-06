//=========================HEADER=============================================================
// Firmware for the Arduino managing the propulsion of the slash platform (UdeS Racecar)
//============================================================================================

#include "Arduino.h"
#include <SPI.h>
#include <Servo.h>
#define USB_USBCON
#include <ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Vector3.h>
#include <math.h>


// Slave Select pins for encoder
const int slaveSelectEnc = 45;

// These hold the current encoder count.
signed long enc_now = 0;

// Servo objects for PWM control of
// Sterring servo-motor
Servo steeringServo;
Servo Maxon_1;
Servo Maxon_2;
Servo Maxon_3;
Servo Maxon_4;

// ROS
ros::NodeHandle  nodeHandle;

//Publisher
const int prop_sensors_msg_length = 21;
float prop_sensors_data[ prop_sensors_msg_length ];

std_msgs::Float32MultiArray prop_sensors_msg;
ros::Publisher prop_sensors_pub("prop_sensors", &prop_sensors_msg);

// Serial Communication
const unsigned long baud_rate = 115200;

float slip_gauche[5];
float slip_droit[5];

// Pins pour outputs
const int enable_pin_front   = 8;       // Pin for enable signal
const int ser_pin            = 9;      // Servo 
const int enable_pin_rear    = 11;
const int ledPin             = 13;

const int maxon_1_pin        = 2;// g
const int maxon_2_pin        = 3;// d
const int maxon_3_pin        = 4;
const int maxon_4_pin        = 5;

// Pins for analog inputs
const int courant_pin_avant_gauche = 0;
const int vitesse_pin_avant_gauche = 2;
const int courant_pin_avant_droit  = 1;
const int vitesse_pin_avant_droit  = 3;

const int courant_pin_arriere_gauche  = 4;
const int vitesse_pin_arriere_gauche  = 6;
const int courant_pin_arriere_droit   = 5;
const int vitesse_pin_arriere_droit   = 7;

// Loop period
const unsigned long time_period_low   = 2;    //2 500 Hz for internal PID loop
const unsigned long time_period_high  = 10;   //10 100 Hz  for ROS communication
const unsigned long time_period_com   = 1000; // 1000 ms = max com delay (watchdog)

// Hardware min-zero-max range for the steering servo and the drive
const int pwm_min_ser = 30  ;
const int pwm_zer_ser = 90  ;
const int pwm_max_ser = 150 ;

const int dri_wakeup_time = 20; // micro second

// Units Conversion
const double maxAngle  = 40 * (2 * 3.1416) / 360; //max steering angle in rad
const double rad2pwm   = (pwm_zer_ser - pwm_min_ser) / maxAngle;
const double tick2m    = 0.000079; //A tester
const float k_c = 0.0273 ; // constante de couple = 27.3 mNm/A

///////////PID///////////////
/////////////////////////////
//Specify the links and initial tuning parameters
double kp = 4, ki = 0.1 , kd =2; //ces valeur sont auusi à régler par l'expérience
unsigned long currentTime, OldTime;
double elapsedTime ; //dt
float PIDvalue ;
double Setpoint , Input,  pid_brake ;
float error = 0 ;
float D = 0;
float I = 0;
float P = 0;
float previousError = 0; //erreur_old
/////////////////////////////////
/////////////////////////////////

///////////SMC///////////////
/////////////////////////////
// Parametres 
float g    = 9.81;
float r    = 0.1; // 10cm
float m    = 20; // 15 kg
//float mu = 0.9 ; // road asphalt
float mu = 0.6 ; // road wet
//float mu = 0.4 ; // snow

// Init
float acceleration     = 0; // smc
float Fn_f = 0;
float Fn_r = 0;
float Fx = 0 ;
float Ji = 0; // moment d'inertie
float error_smc= 0 ;
float D_error_smc = 0 ;
float s = 0;
float sgn = 0;
float T_eq = 0 ;
float T_bh= 0 ;
float vitesse_arriere_droit = 0;
float previousError_smc = 0 ; //erreur_old
float k_smc =2;
float SMC_brake= 0;
/////////////////////////////////
/////////////////////////////////

// Inputs
float ser_ref          = 0;
float consigne_RPM     = 0;
float consigne_PWM     = 0;
float consigne_current = 0;
float consigne_ser     = 0;
float consigne_ser_FR     = 0;
float consigne_ser_FL     = 0;
float consigne_ser_RR     = 0;
float consigne_ser_RL     = 0;

float consigne_ser_br     = 0;
float dri_ref_FR          = 0;
float dri_ref_FL          = 0;
float dri_ref_RR          = 0;
float dri_ref_RL          = 0;
float acceleration_vehicule = 0;
int   ctl_mode              = 0; // discrete control mode
float slip_arriere_droit    = 0;
float slip_arriere_gauche   = 0;
float vitesse_vehicule      = 0;
float slip_ref              = 0.2;
float k = 1;

//float delta = abs(map(ser_pin,1000,2000,-0.418,0.418)) ; // angle de braquage en rad(24 deg=0.4188rad)
float delta =0;
float teta = 0 ; //angle de tangageen en rad
float L = 0.7 , Lf = 0.35 , Lr = 0.35 , H = 0.4 ; ; // 70 cm longueur de vehicule
float Hf = H , Hr = H ;



// Ouputs
int   ser_pwm  = 0;
float SMCvalue = 0; //output SMC Value

// Loop timing
unsigned long time_now       = 0;
unsigned long time_last_low  = 0;
unsigned long time_last_high = 0;
unsigned long time_last_com  = 0; //com watchdog

// input ref for braking (amp)
float input_low  =  0.0;
float input_high = -5.0;

void cmdCallback ( const std_msgs::Float32MultiArray&  propCmdMsg ) {
  
  ser_ref               = -propCmdMsg.data[1]; //rad steering
  dri_ref_FR            = propCmdMsg.data[2];  // volt or m/s or m
  dri_ref_FL            = propCmdMsg.data[3];  // volt or m/s or m
  dri_ref_RR            = propCmdMsg.data[4];  // volt or m/s or m
  dri_ref_RL            = propCmdMsg.data[5];  // volt or m/s or m
  acceleration_vehicule = propCmdMsg.data[6];  
  ctl_mode              = propCmdMsg.data[0];  // 1   or 2   or 3
  slip_ref              = propCmdMsg.data[7];
  
  time_last_com = millis(); // for watchdog
}
// ROS suscriber
ros::Subscriber<std_msgs::Float32MultiArray> cmdSubscriber("prop_cmd_costum", &cmdCallback) ;

void setup() {
  
  Serial.begin(115200);
  
  
  Setpoint = slip_ref; //Setpoint = 0.3 // PID
  // Init PWM output Pins
  steeringServo.attach(ser_pin);
  Maxon_1.attach(maxon_1_pin, 1000, 2000);
  Maxon_2.attach(maxon_2_pin, 1000, 2000);
  Maxon_3.attach(maxon_3_pin, 1000, 2000);
  Maxon_4.attach(maxon_4_pin, 1000, 2000);

  pinMode(maxon_1_pin, OUTPUT);
  pinMode(maxon_2_pin, OUTPUT);
  pinMode(maxon_3_pin, OUTPUT);
  pinMode(maxon_4_pin, OUTPUT);

  pinMode(enable_pin_front, OUTPUT);
  pinMode(enable_pin_rear, OUTPUT);
  pinMode(ledPin, OUTPUT);

  pinMode(courant_pin_avant_gauche, INPUT);
  pinMode(vitesse_pin_avant_gauche, INPUT);
  pinMode(courant_pin_avant_droit, INPUT);
  pinMode(vitesse_pin_avant_droit, INPUT);

  pinMode(courant_pin_arriere_gauche, INPUT);
  pinMode(vitesse_pin_arriere_gauche, INPUT);
  pinMode(courant_pin_arriere_droit, INPUT);
  pinMode(vitesse_pin_arriere_droit, INPUT);

  // Init Communication
  nodeHandle.getHardware()->setBaud(baud_rate);

  // Init ROS
  nodeHandle.initNode();
  nodeHandle.subscribe(cmdSubscriber) ; // Subscribe to the steering and throttle messages
  nodeHandle.advertise(prop_sensors_pub);

  // Initialize Steering and drive cmd to neutral
  steeringServo.write(pwm_zer_ser) ;

  delay(3000) ;
  
  // Clear and INIT encoders
  initEncoder();
  clearEncoderCount();
  nodeHandle.spinOnce(); 

}

void loop() {

  time_now = millis();

  enc_now = readEncoder();
  //float offset_i = -425.0;
  float offset_i = -409.2;
  float offset_i2 = -1 * 1023.0 * 2.0 / 5.0;
  float V2i = 8.0 / (1023.0 * 4.0 / 5.0) * 2.0 / 1.12;
  float v2i = 16.0 / (1023.0 * 4.0 / 5.0);

  float RPM2m_s = 2.0 * PI * .1 / 60.0 / 10.6;
  float V2m_s = 5000.0 / (1023.0 * 4.0 / 5.0) * RPM2m_s / 1.12;
  float offset_v = -409.2;
  float v2m_s = 5000 / (1023.0 * 4.0 / 5.0) * RPM2m_s;

  float courant_avant_gauche = (analogRead( courant_pin_avant_gauche ) + offset_i) * V2i;
  float vitesse_avant_gauche = (analogRead( vitesse_pin_avant_gauche ) + offset_v) * v2m_s;
  float courant_avant_droit = (analogRead( courant_pin_avant_droit ) + offset_i) * V2i;
  float vitesse_avant_droit = (analogRead( vitesse_pin_avant_droit ) + offset_v) * v2m_s;

  float courant_arriere_gauche = (analogRead( courant_pin_arriere_gauche ) + offset_i) * V2i;
  float vitesse_arriere_gauche = (analogRead( vitesse_pin_arriere_gauche ) + offset_v) * v2m_s;
  float courant_arriere_droit = (analogRead( courant_pin_arriere_droit ) + offset_i) * V2i;
  float vitesse_arriere_droit = (analogRead( vitesse_pin_arriere_droit ) + offset_v) * v2m_s;

  vitesse_vehicule = vitesse_arriere_gauche;
  
  // Glissement 
  
 
    slip_arriere_gauche = ( vitesse_arriere_gauche - vitesse_avant_gauche ) / max(vitesse_avant_gauche, vitesse_arriere_gauche);
    slip_arriere_gauche=constrain(slip_arriere_gauche,-1.0,1.0);
    slip_arriere_droit = ( vitesse_arriere_droit - vitesse_avant_droit ) / max(vitesse_avant_droit, vitesse_arriere_droit);
    slip_arriere_droit=constrain(slip_arriere_droit,-1.0,1.0); 
  

  
  float pos_arriere_droit  = enc_now * tick2m;
  


  if (( time_now - time_last_low ) > time_period_low ) {
     //Read encoders
    
    slip_gauche[0] = slip_gauche[1];
    slip_gauche[1] = slip_gauche[2];
    slip_gauche[2] = slip_gauche[3];
    slip_gauche[3] = slip_gauche[4];
    slip_gauche[4] = slip_arriere_gauche;

    slip_droit[0] = slip_droit[1];
    slip_droit[1] = slip_droit[2];
    slip_droit[2] = slip_droit[3];
    slip_droit[3] = slip_droit[4];
    slip_droit[4] = slip_arriere_droit;
  //////////PID///////////
  //Input = slip_arriere_gauche;
  pid_brake = calculatePID();
  
  //////////SMC///////////
  //Input = slip_arriere_droit;
  SMC_brake = calculateSMC();
 
    ctl(); // one control tick 
    time_last_low = time_now ;
  
  }

  unsigned long dt = time_now - time_last_high;
  if (dt > time_period_high ) {
    
    // Feedback loop
    prop_sensors_data[0] = pos_arriere_droit;
    
    prop_sensors_data[1] = vitesse_avant_gauche; //
    prop_sensors_data[2] = vitesse_avant_droit; //
    prop_sensors_data[3] = vitesse_arriere_gauche; // set point received by arduino
    prop_sensors_data[4] = vitesse_arriere_droit; // drive set point in volts

    prop_sensors_data[5] = courant_avant_gauche; //
    prop_sensors_data[6] = courant_avant_droit; //
    prop_sensors_data[7] = courant_arriere_gauche; //
    prop_sensors_data[8] = courant_arriere_droit; //

    prop_sensors_data[9] = k;

    prop_sensors_data[10] = slip_arriere_gauche;
    prop_sensors_data[11] = slip_gauche[1];
    prop_sensors_data[12] = slip_gauche[2];
    prop_sensors_data[13] = slip_gauche[3];
    prop_sensors_data[14] = slip_gauche[4];

    prop_sensors_data[15] = slip_arriere_droit;
    prop_sensors_data[16] = slip_droit[1];
    prop_sensors_data[17] = slip_droit[2];
    prop_sensors_data[18] = slip_droit[3];
    prop_sensors_data[19] = slip_droit[4];

    prop_sensors_data[20] = acceleration;


    prop_sensors_msg.data        = &prop_sensors_data[0];
    prop_sensors_msg.data_length = prop_sensors_msg_length;
    prop_sensors_pub.publish( &prop_sensors_msg );

    // Process ROS Events
    nodeHandle.spinOnce();
    time_last_high = time_now ;
  }
}
////// FUNCTION ///////

/////////PID/////////////
////////////////////////
  double calculatePID () {
        currentTime = millis();
        elapsedTime = currentTime - OldTime; // dt (Duree ecoulee)
        error = Setpoint - slip_arriere_gauche;
        P = error;
        I = I + error ;
         D = error - previousError ;
        //I = (I + error)*elapsedTime ;
        //D = (error - previousError)/elapsedTime ;
        PIDvalue = (kp * P) + (ki * I) + (kd * D);
        previousError = error;
        OldTime = currentTime;
        PIDvalue=(-PIDvalue*62.5)+1500;
        PIDvalue = constrain(PIDvalue,1300,1500);
       
       // PIDvalue = map(PIDvalue , input_high , input_low , 1500, 1000); // Output drives "servo.writeMicroseconds"
        return PIDvalue;
      }
//////////////////////////////
/////////////////////////////

///////////SMC///////////////
/////////////////////////////
  double calculateSMC () {
    
     currentTime = millis();
     elapsedTime = (double)(currentTime - OldTime);

      Hf = H - Lf * sin(teta); // hauteur avant
      Hr = H + Lr * sin(teta); // hauteur arriere
      // Force normale de la roue avant
      Fn_f = m * acceleration * (Hr / L) + m * g * (Lr / L);
      // force normale de la roue arrière
      Fn_r = -m * g * (Lf / L) + m * acceleration* (Hr / L);
      // Mouvement longitudinal
      Fx = 4* mu * Fn_f;
      // the inertia of the vehicle
      Ji = 4 * m * sq(r) ;
      //k_smc = ((vitesse_arriere_gauche * Ji)/r)*2;// averfier
      error_smc = slip_ref - slip_arriere_gauche;
      D_error_smc = error_smc - previousError_smc;
      
      float lam = 20; // a changer par une equation
      s = (error_smc * lam) + D_error_smc;
      // fonction de signe 
      if (s < 0) {sgn = -1.0;}
      if (s = 0) {sgn = 0.0;}
      if (s > 0) {sgn = 1.0;}
      // couple de commutation
      T_bh = k_smc * sgn   ;
      // Couple équivalent
      T_eq = Fx * r - ((1 - slip_arriere_gauche) * (acceleration* Ji / r) ) ;
      // final SMC value
      SMCvalue = (T_eq + T_bh)/k_c;
      previousError_smc = error_smc;
      SMCvalue =(-SMCvalue*62.5)+1500;
      SMCvalue = constrain(SMCvalue,1300,1500);
      //SMCvalue = map(SMCvalue ,-4.0, 0.0, 1000, 1500); // Output drives "servo.writeMicroseconds"
      return SMCvalue;
}

//////////////////////////////
/////////////////////////////
