// Convertion function : Servo Angle --> PWM
double ser2pwm (double cmd) {
  
  // Scale and offset
  double pwm_d = cmd * rad2pwm + (double) pwm_zer_ser;
  
  // Rounding and conversion
  int pwm = (int) ( pwm_d + 0.5 );
  
  // Saturations
  if (pwm > pwm_max_ser) {
    pwm = pwm_max_ser;
  }
  if (pwm < pwm_min_ser) { 
    pwm = pwm_min_ser;
  }
  
  return pwm;
}

// Convertion function : Meter/s --> RPM
double m_s2rpm (double dri_ref_right) {
  
  // Scale and offset
  double RPM = dri_ref_right/( 2.0*3.14159*.1/60.0 );
  
  return RPM;
}

// Convertion function : RPM --> PWM
double rpm2pwm (double RPM) {
  
  // Scale and offset
  double pwm = RPM*511.0 / 5000.0;
  
  return pwm;
}

// Convertion function : torque --> current
double torque2i (double dri_ref) {
  
  // Scale and offset
  double i= dri_ref/k_c;
  
  return i;
}

// Convertion function : i --> PWM
double i2pwm (double i) {
  
  // Scale and offset
 double pwm = i*62.5 + 1500.0; // interpolation linéaire 
  //double pwm = i*13.65 + 1500.0; // interpolation linéaire 
  return pwm;
}

// Convertion function : PWM --> servo
double pwm2ser (double PWM) {
  
  // Scale and offset
  double ser = PWM/511.0*90.0+90.0;
  
  return ser;
}
