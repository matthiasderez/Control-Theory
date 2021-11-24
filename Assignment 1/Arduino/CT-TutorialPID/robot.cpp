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
  for(int k=0; k<2; k++){
    x[k] = 0.0;   // Set all components of the vector (float array) x to 0 as initialization
    e[k] = 0.0;
    u[k] = 0.0;
  }
  
  previousTime = 0;
  voltage = 6.0;
  out = voltage; 
  counter = 0;
  counterThousand = 0;
  sign = 1;

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Robot::control() {

  // Compute update of motor voltages if controller is enabled (triggered by
  // pushing 'Button 0' in QRoboticsCenter)
  if(controlEnabled()) {
    // Fill your control law here to conditionally update the motor voltage...
    LED1(ON);
    LED2(OFF);

//    unsigned long currentTime = millis();

//    if (currentTime - previousTime >= eventInterval) {
//      /* Event code */
//      LED1(OFF);
//      out = -out;
//      /* Update the timing for the next time around */
//      previousTime = currentTime;
//    }
    
    if (counter - counterThousand >= 800) {
      /* Event code */
      LED1(OFF);
      if (out != 0){
        sign = -sign;
      }
      out = out + sign * voltage;
      counterThousand = counter;
    }
    
    
    counter = counter + 1;
    
    
    setVoltageMotorA(out);
    setVoltageMotorB(out);
 



// ------------------------------------------------------------------------------------------------------

    


  
      

//    float setpoint = pb;
//    float error = setpoint - pa;
//    
//    e[1] = e[0]; e[0] = error;
//    float output = 2.05*e[0] - 1.95*e[1] + u[1];
//    u[1] = u[0]; u[0] = output; 
//  
//    setVoltageMotorA(u[0]);

// -----------------------------------------------------------------------------------------------------
    
  } else {
    // If the controller is disabled, you might want to do something else...
    LED1(OFF);
    LED2(ON);
    setVoltageMotorA(0.0); // Apply 0.0 volts to motor A if the control is disabled
    setVoltageMotorB(0.0); // Apply 0.0 volts to motor B if the control is disabled
    
    writeValue(0, getVoltageMotorA());
    writeValue(3, getSpeedMotorA());
    writeValue(4, getSpeedMotorB());
    out = voltage; 
    counter = 0;
    counterThousand = 0;
    sign = 1;
    
  }
  
    float vol_A = readValue(0);
    float vol_B = readValue(1);
    
    //setVoltageMotorA(vol_A); // Apply 6.0 volts to motor A if the control is enabled
    //setVoltageMotorB(vol_B); // Apply 2.0 volts to motor B if the control is enabled
    float voltage_A = getVoltageMotorA();
    writeValue(0, voltage_A); 
    float voltage_B = getVoltageMotorB();
    writeValue(1, voltage_B);

    float pa = getPositionMotorA();
    writeValue(2, pa);
    float pb = getPositionMotorB();
    writeValue(3, pb);
  
    float va = getSpeedMotorA();    // Get the wheel speed of motor A (in radians/second)
    x[1] = x[0]; x[0] = va;         // Memorize the last two samples of the speed of motor A (in fact, a shift register)
    writeValue(4, va);
    float vb = getSpeedMotorB();  
    writeValue(5, vb);
  
    float fd = getFrontDistance(); 
    writeValue(6, fd);
    float pangle = getPendulumAngle(); 
    writeValue(7, pangle);
    time = counter * 0.01;
    writeValue(8,time);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Robot::button1callback() {
  toggleButton(1);
  init();                         // Reset the MECOtron and reinitialize the Robot object
  message("Reset.");
}
