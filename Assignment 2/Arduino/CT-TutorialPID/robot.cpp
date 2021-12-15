/*
 * MECOTRON TUTORIAL
 *
 * This is a template to get you started in the course of the tutorial on the
 * control theory platforms, a.k.a. the MECOtrons.s
 * The tasks of the tutorial session will guide you through this template and
 * ask you to make use of the platform's capabilities in a step-by-step fashion.
 *
 * Every function in this template comes with an opening comment that describes
 * its purpose and functionality. Please also pay attention to the remarks that
 * are made in comment blocks.
 *
 */

#include "robot.h"

bool Robot::init() {
  MECOtron::init(); // Initialize the MECOtron

  // Initializing the robot's specific variables
    errorA = 0.0;
    errorB = 0.0;
    controlA = 0.0;
    controlB = 0.0;
    a1 = 0.8;
    a2 = 0.6692;
    a3 = 1;
    a4 = 1;
    b1 = 0.775;
    b2 = 0.6442;
    b3 = 1;
    b4 = 1;
    counter = 0.0;
    wdes = -6.0;
    
  return true;
}

void Robot::control() {

  // Compute update of motor voltages if controller is enabled (triggered by
  // pushing 'Button 0' in QRoboticsCenter)
  if(controlEnabled()) {
    // Fill your control law here to conditionally update the motor voltage...
    LED1(ON);
    LED2(OFF);
    float wA = getSpeedMotorA();
    float wB = getSpeedMotorB();        
    float eA = wdes-wA;                 //  calculate the position error of motor A (in radians)
    float eB = wdes-wB;                 //  calculate the position error of motor B (in radians)

    // the actual control algorithm
    float uA = a4/a3*controlA + a1/a3*eA -a2/a3*errorA; // equation (2), the difference equation
    float uB = b4/b3*controlB + b1/b3*eB -b2/b3*errorB; // equation (2), the difference equation

 
    errorA = eA; errorB = eB; controlA = uA; controlB = uB;    // append the new values

    // apply the control signal
    setVoltageMotorA(uA);
    setVoltageMotorB(uB);
    counter = counter + 1;
    writeValue(0, uA);
    writeValue(1, uB);
    writeValue(4, getSpeedMotorA());
    writeValue(5, getSpeedMotorB());
    writeValue(10,eA);
    
  } else {
    // If the controller is disabled, you might want to do something else...
    LED1(OFF);
    LED2(ON);
    counter = 0.0;
    setVoltageMotorA(0.0);
    setVoltageMotorB(0.0);
    writeValue(0, 0.0);
    writeValue(1, 0.0);
    writeValue(4, getSpeedMotorA());
    writeValue(5, getSpeedMotorB());
    errorA = 0.0;
    errorB = 0.0;
    controlA = 0.0;
    controlB = 0.0;
    

  }
  writeValue(11, errorA);
  float pa = getPositionMotorA();
  writeValue(2, pa);
  float pb = getPositionMotorB();
  writeValue(3, pb);
  float fd = getFrontDistance();
  writeValue(6, fd);
  float pangle = getPendulumAngle();
  writeValue(7, pangle);
  time = counter * 0.01;
  writeValue(8, time);


}

bool Robot::controlEnabled() {
  return _button_states[0];       // The control is enabled if the state of button 0 is true
}

void Robot::button0callback() {
  if(toggleButton(0)) {           // Switches the state of button 0 and checks if the new state is true
    message("Robot enabled.");    // Display a message in the status bar of QRoboticsCenter
  }
  else {
    message("Robot disabled.");
  }
}

void Robot::button1callback() {
  toggleButton(1);
  init();                         // Reset the MECOtron and reinitialize the Robot object
  message("Reset.");
}
