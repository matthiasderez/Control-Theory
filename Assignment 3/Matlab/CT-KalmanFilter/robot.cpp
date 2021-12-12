/*
 * KALMAN FILTER TEMPLATE
 *
 * This is a template to get you started with the implementation of the Kalman filter
 * on your own cart.
 *
 */

#include "robot.h"

bool Robot::init() {
  MECOtron::init(); // Initialize the MECOtron

  desired_velocity(0) = 0;
  return true;
}

void Robot::control() {

  float volt_A = 0.0;
  float volt_B = 0.0;
  Matrix<1> desired_velocity; //control signal
  desired_velocity.Fill(0); //Initialize matrix with zeros

  // Kalman filtering
  if(KalmanFilterEnabled()) {   // only do this if controller is enabled (triggered by pushing 'Button 1' in QRoboticsCenter)
    // Correction step
    Matrix<1> distance_measurement;                                     // define a vector of length 1
    distance_measurement(0) = getFrontDistance();                       // front distance
    CorrectionUpdate(distance_measurement, _xhat, _Phat, _nu, _S);     // do the correction step -> update _xhat, _Phat, _nu, _S
  }
  writeValue(8, _xhat(0)); // a posteriori state estimate
  writeValue(9, _Phat(0)); // a posteriori state covariance
  writeValue(10, _nu(0)); // innovation
  writeValue(11, _S(0));  // innovation covariance

  if(controlEnabled()) {   // only do this if controller is enabled (triggered by pushing 'Button 0' in QRoboticsCenter)

    // // UNCOMMENT AND COMPLETE LINES BELOW TO IMPLEMENT POSITION CONTROLLER
    //float desired_position = readValue(0);      // use channel 0 to provide the constant position reference
    //xref(0) = ? ;                               // transform desired_position to the state reference (make sure units are consistent)
    //K(0) = ? ;                                  // state feedback gain K, to design
    //desired_velocity = K * (xref - _xhat);      // calculate the state feedback signal, (i.e. the input for the velocity controller)

    //// UNCOMMENT AND COMPLETE LINES BELOW TO IMPLEMENT VELOCITY CONTROLLER
    // ...
    // ...                                          // implement your velocity controller here (assignment 2), such that the motors track velocity v
    // ...
    //volt_A =
    //volt_B =

    //// COMMENT OR REMOVE LINES BELOW ONCE YOU IMPLEMENT THE VELOCITY CONTROLLER
    volt_A = 0.0;
    volt_B = 0.0;

    // Send wheel speed command
    setVoltageMotorA(volt_A);
    setVoltageMotorB(volt_B);
  }
  else                      // do nothing since control is disables
  {
    desired_velocity(0) = 0.0;
    setVoltageMotorA(0.0);
    setVoltageMotorB(0.0);
  }

  // Kalman filtering
  if(KalmanFilterEnabled()) {   // only do this if controller is enabled (triggered by pushing 'Button 1' in QRoboticsCenter)
    // Prediction step
    PredictionUpdate(desired_velocity, _xhat, _Phat);                    // do the prediction step -> update _xhat and _Phat
  }
  // writeValue(8, _xhat(0)); // a priori state estimate
  // writeValue(9, _Phat(0)); // a priori state covariance
  
  // Send useful outputs to QRC
  writeValue(0, volt_A);
  writeValue(1, volt_B);
  writeValue(2, desired_velocity(0));
  writeValue(3, getPositionMotorA());
  writeValue(4, getPositionMotorB());
  writeValue(5, getSpeedMotorA());
  writeValue(6, getSpeedMotorB());
  writeValue(7, getFrontDistance());


}

void Robot::resetController(){
  // Set all errors and control signals in the memory back to 0
  // ...
  // ...
}

void Robot::resetKalmanFilter() {
  // // UNCOMMENT AND MODIFIES LINES BELOW TO IMPLEMENT THE RESET OF THE KALMAN FILTER
  // // Initialize state covariance matrix
  // _Phat.Fill(0);      // Initialize the covariance matrix
  // _Phat(0,0) = ?;     // Fill the initial covariance matrix, you can change this according to your experiments
  //
  // // Initialize state estimate
  // _xhat(0) = 0.0;     // Change this according to your experiments
}

bool Robot::controlEnabled() {
  return _button_states[0];       // The control is enabled if the state of button 0 is true
}

bool Robot::KalmanFilterEnabled() {
  return _button_states[1];
}

void Robot::button0callback() {
  if(toggleButton(0)) {           // Switches the state of button 0 and checks if the new state is true
    resetController();
    message("Controller resed and enabled.");    // Display a message in the status bar of QRoboticsCenter
  }
  else {
    message("Control disabled.");
  }
}

void Robot::button1callback() {
  if(toggleButton(1)){
      resetKalmanFilter();            // Reset the Kalman filter
      message("Kalman filter reset and enabled.");
  }
  else
  {
    message("Kalman filter disabled.");
  }
}
