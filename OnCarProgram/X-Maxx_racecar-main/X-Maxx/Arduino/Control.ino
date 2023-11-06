void ctl() {

  // Servo Open-Loop fonction
  ser_pwm      = ser2pwm( ser_ref ) ;
  steeringServo.write(ser_pwm) ;

  // Propulsion Controllers

  if (ctl_mode == 4) {
    //Same command for every wheels
    Maxon_1.write(1500 + (dri_ref_FR * 500));
    Maxon_2.write(1500 + (dri_ref_FL * 500));
    Maxon_3.write(1500 + (dri_ref_RR * 500));
    Maxon_4.write(1500 + (dri_ref_RL * 500));
    /*if (vitesse_vehicule > 3){

        digitalWrite(enable_pin_front,LOW);
        digitalWrite(enable_pin_rear,LOW);
      }
      else{
      digitalWrite(enable_pin_front,HIGH);
      digitalWrite(enable_pin_rear,HIGH);

      }*/
    digitalWrite(enable_pin_front, HIGH);
    digitalWrite(enable_pin_rear, HIGH);
    digitalWrite(ledPin, HIGH);
  }

  else if (ctl_mode == 5) {
    //Only front Wheels
    digitalWrite(enable_pin_rear, LOW);
    Maxon_1.write(1500 + (dri_ref_FR * 500)); //gauche
    Maxon_2.write(1500 + (dri_ref_FL * 500)); //droit
    digitalWrite(enable_pin_front, HIGH);
    digitalWrite(ledPin, HIGH);
  }

  else if (ctl_mode == 6) {
    //All Wheels, different commands
    Maxon_1.write(1500 + (dri_ref_FR * 500));
    Maxon_2.write(1500 + (dri_ref_FL * 500));
    Maxon_3.write(1500 + (dri_ref_RR * 500));
    Maxon_4.write(1500 + (dri_ref_RL * 500));
    digitalWrite(enable_pin_front, HIGH);
    digitalWrite(enable_pin_rear, HIGH);
    digitalWrite(ledPin, HIGH);
  }

  else if (ctl_mode == 11) { //100%brake
    //Consigne de frein 1125 = -8A
    consigne_ser_br = (-6 * 62.5) + 1500;
    if (vitesse_vehicule > 0.2) {
      Maxon_1.write(consigne_ser_br);
      Maxon_2.write(consigne_ser_br);
      Maxon_3.write(consigne_ser_br);
      Maxon_4.write(consigne_ser_br);
      digitalWrite(enable_pin_front, HIGH);
      digitalWrite(enable_pin_rear, HIGH);
    }
    else {
      Maxon_1.write(1500);
      Maxon_2.write(1500);
      Maxon_3.write(1500);
      Maxon_4.write(1500);
      digitalWrite(enable_pin_front, LOW);
      digitalWrite(enable_pin_rear, LOW);
    }
  }
  else if (ctl_mode == 12) { //ABSC
    //Consigne de frein 1062 = -7A
    consigne_ser_br = (-6 * 62.5) + 1500; // pwm
    Maxon_1.write(consigne_ser_br);
    Maxon_2.write(consigne_ser_br);
    Maxon_3.write(1470);
    Maxon_4.write(1470);
    digitalWrite(enable_pin_front, HIGH);
    digitalWrite(enable_pin_rear, HIGH);
    if (vitesse_vehicule > 0.2) {

      if (slip_arriere_gauche < slip_ref) {
        Maxon_1.write( consigne_ser_br);
        //digitalWrite(enable_pin_front,HIGH);
        k = 1;
      }
      else {
        Maxon_1.write(1470);
        //digitalWrite(enable_pin_front,LOW);
        k = 0;
      }
      if (slip_arriere_droit < slip_ref) {
        Maxon_2.write( consigne_ser_br);
        // digitalWrite(enable_pin_front,HIGH);
        k = 1;
      }
      else {
        Maxon_2.write(1470);
        //digitalWrite(enable_pin_front,LOW);
        k = 0;
      }
    }
    else {
      digitalWrite(enable_pin_front, LOW);
      digitalWrite(enable_pin_rear, LOW);
    }
  }
  else if (ctl_mode == 13) { // PID
    //consigne_ser_br = pid_brake;
    //Maxon_1.write( consigne_ser_br);
    //Maxon_2.write( consigne_ser_br);
    Maxon_3.write(1450);
    Maxon_4.write(1450);
    digitalWrite(enable_pin_front, HIGH);
    digitalWrite(enable_pin_rear, HIGH);
    if (vitesse_vehicule > 0.2) {

      if (slip_arriere_gauche < slip_ref) {
        Maxon_1.write( pid_brake);
        k = 1;
      }
      else {
        Maxon_1.write(1450);
        k = 0;
      }
      if (slip_arriere_droit < slip_ref) {
        Maxon_2.write( pid_brake);
        k = 1;
      }
      else {
        Maxon_2.write(1450);
        k = 0;
      }
    }
    else {
      digitalWrite(enable_pin_front, LOW);
      digitalWrite(enable_pin_rear, LOW);
    }
    digitalWrite(ledPin, HIGH);
  }
  else if (ctl_mode == 14) { // SMC
    //consigne_ser_br = SMC_brake;
    Maxon_1.write( consigne_ser_br);
    Maxon_2.write( consigne_ser_br);
    Maxon_3.write(1450);
    Maxon_4.write(1450);
    digitalWrite(enable_pin_front, HIGH);
    digitalWrite(enable_pin_rear, HIGH);
    if (vitesse_vehicule > 0.2) {

      if (slip_arriere_gauche < slip_ref) {
        Maxon_1.write( SMC_brake);
        k = 1;
      }
      else {
        Maxon_1.write(1450);
        k = 0;
      }
      if (slip_arriere_droit < slip_ref) {
        Maxon_2.write( SMC_brake);
        k = 1;
      }
      else {
        Maxon_2.write(1450);
        k = 0;
      }
    }
    else {
      digitalWrite(enable_pin_front, LOW);
      digitalWrite(enable_pin_rear, LOW);
    }
    digitalWrite(ledPin, HIGH);
  }
  else {
    clearEncoderCount();
    consigne_RPM = 0.0;
    consigne_PWM = 0.0;
    consigne_current = 0.0;
    consigne_ser_FL = pwm2ser(consigne_PWM);
    consigne_ser_FR = pwm2ser(consigne_PWM);
    consigne_ser_RL = pwm2ser(consigne_PWM);
    consigne_ser_RR = pwm2ser(consigne_PWM);

    Maxon_1.write(consigne_ser_FL);
    Maxon_2.write(consigne_ser_FR);
    Maxon_3.write(consigne_ser_RL);
    Maxon_4.write(consigne_ser_RR);

    digitalWrite(enable_pin_front, LOW);
    digitalWrite(enable_pin_rear, LOW);
    digitalWrite(ledPin, LOW);
  }

}
