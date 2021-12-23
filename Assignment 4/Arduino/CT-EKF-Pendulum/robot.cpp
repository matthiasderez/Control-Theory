/*
 * EXTENDED KALMAN FILTER TEMPLATE
 *
 * This is a template to get you started with the implementation of the Kalman filter
 * on your own cart.
 *
 */

#include "robot.h"

bool Robot::init() {
  MECOtron::init(); // Initialize the MECOtron

  desiredVelocityCart(0) = 0.0;
  r = 0.033;
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
    ref = 0;
  return true;
  
}

void Robot::control() {

  float volt_A = 0.0;
  float volt_B = 0.0;
  float desiredVelocityMotorA = 0.0;
  float desiredVelocityMotorB = 0.0;
  Matrix<1> desiredVelocityCart;  // control signal
  desiredVelocityCart.Fill(0); //Initialize matrix with zeros
  Matrix<2> measurements;

  // Kalman filtering
  if(KalmanFilterEnabled()) { // only do this if Kalman Filter is enabled (triggered by pushing 'Button 1' in QRoboticsCenter)
      
     // UNCOMMENT AND MODIFY LINES BELOW TO IMPLEMENT THE KALMAN FILTER
     // Correction step

     measurements(0) = getPositionMotorA()*r; //transform encoders measurement (getPositionMotorA() and getPositionMotorB()) to cart position measurement
     measurements(1) = -getPendulumAngle();
     CorrectionUpdate(measurements, _xhat, _Phat, _nu, _S);     // do the correction step -> update _xhat, _Phat, _nu, _S

    // // Useful outputs to QRC for assignment questions
//     writeValue(7, _xhat(0));
//     writeValue(8, _xhat(1));
//     writeValue(9, _xhat(2));
//     writeValue(1, _Phat(0,0));
//     writeValue(2, _Phat(1,0));
//     // writeValue(4, _Phat(1,0));
//     writeValue(4, _Phat(1,1));
//     writeValue(3, _Phat(2,0));
//     writeValue(5, _Phat(2,1));
//     writeValue(6, _Phat(2,2));
//     writeValue(10, measurements(0));
//     writeValue(11, measurements(1));
  }

  if(controlEnabled()) {   // only do this if controller is enabled (triggered by pushing 'Button 0' in QRoboticsCenter)
//    // UNCOMMENT AND COMPLETE LINES BELOW TO IMPLEMENT THE FEEDFORWARD INPUTS (ASSIGNMENT 4.2, no state feedback here)
//    // COMMENT OR REMOVE LINES BELOW ONCE YOU IMPLEMENT THE POSITION STATE FEEDBACK CONTROLLER
//    // Compute the feedforward input of the cart
//    // The feedforward is here returned by the built-in trajectory: trajectory.v()
//    desiredVelocityCart(0) = trajectory.v();  //desired forward velocity of the cart (in m/s)
//    // The trajectory must be started by pushing 'Button 2' in QRoboticsCenter, otherwise will return zeros
//    // after any experiment the trajectory must be reset pushing 'Button 3' in QRoboticsCenter
//    //
//     // apply the static transformation between velocity of the cart and velocity of the motors
//     desiredVelocityMotorA = desiredVelocityCart(0)/r;        // calculate the angular velocity of the motor A using desiredVelocityCart
//     desiredVelocityMotorB = desiredVelocityCart(0)/r;        // calculate the angular velocity of the motor B using desiredVelocityCart


     // UNCOMMENT AND COMPLETE LINES BELOW TO IMPLEMENT POSITION CONTROLLER (ASSIGNMENT 4.4)
     // step reference in the desired pendulum mass position
     ref(0)= readValue(0);         // r [m] - use channel 0 to provide the step reference
    
     // State feedback controller
     float arrayKfb[1][3]{{3.0597,   -0.4433,    0.2208}};  // state feedback gain Kfb, to design
     Matrix<1, 3> Kfb = arrayKfb;
    
     // Compute feedback signal ufb = -Kfb*x
     Matrix<1> ufb = -Kfb * _xhat;
    
     // Feedforward controller
     float arrayKff[1][1]{3.0597};  // feedforward gain Kff, to design
     Matrix<1> Kff = arrayKff;
    
     Matrix<1> uff = Kff*ref;
    
     // Compute the control action u = uff + ufb
     desiredVelocityCart(0) = uff(0) + ufb(0);  // desired forward velocity of the cart from the feedforward and state feedback controller
//
     desiredVelocityMotorA = desiredVelocityCart(0)/r;        // calculate the angular velocity of the motor A using desiredVelocityCart
     desiredVelocityMotorB = desiredVelocityCart(0)/r;        // calculate the angular velocity of the motor B using desiredVelocityCart
    float wA = getSpeedMotorA();
    float wB = getSpeedMotorB();        
    float eA = desiredVelocityMotorA-wA;      // Here wdes vervangen door desired_velocity(0)/R [rad/s]       //  calculate the position error of motor A (in radians)
    float eB = desiredVelocityMotorB-wB;                 //  calculate the position error of motor B (in radians)

    // the actual control algorithm
    float uA = a4/a3*controlA + a1/a3*eA -a2/a3*errorA; // equation (2), the difference equation
    float uB = b4/b3*controlB + b1/b3*eB -b2/b3*errorB; // equation (2), the difference equation

 
    errorA = eA; errorB = eB; controlA = uA; controlB = uB;    // append the new values

    volt_A = uA;
    volt_B = uB;
    counter = counter +1;
    
  

    // Send wheel speed command
    setVoltageMotorA(volt_A);
    setVoltageMotorB(volt_B);
    writeValue(1, uA);
    
  }
  else                      // do nothing since control is disables
  {
   desiredVelocityCart(0) = 0.0;
   setVoltageMotorA(0.0);
   setVoltageMotorB(0.0);
  }

  // Kalman filtering
  if(KalmanFilterEnabled()) { // only do this if Kalman Filter is enabled (triggered by pushing 'Button 1' in QRoboticsCenter)
    // Prediction step
    PredictionUpdate(desiredVelocityCart, _xhat, _Phat);                    // do the prediction step -> update _xhat and _Phat
  }

  // Send useful outputs to QRC
  // to check functioning of trajectory and feedforward
  writeValue(0, trajectory.v());
//  writeValue(1, trajectory.X());
//  writeValue(2, trajectory.hasMeasurements());
//  writeValue(3, getSpeedMotorA());
//  writeValue(4, getSpeedMotorB());
//  writeValue(5, volt_A);
//  writeValue(6, volt_B);
//    writeValue(5, desiredVelocityMotorA); // Puur voor test even
//    writeValue(6, volt_B);

   // Useful outputs to QRC for assignment questions
     writeValue(7, _xhat(0));
     writeValue(8, _xhat(1));
     writeValue(9, _xhat(2));
     //writeValue(1, _Phat(0,0));
     
     writeValue(2, _Phat(1,0));
     // writeValue(4, _Phat(1,0));
     writeValue(4, _Phat(1,1));
     writeValue(3, _Phat(2,0));
     writeValue(5, _Phat(2,1));
     writeValue(6, _Phat(2,2));
     writeValue(10, measurements(0));
     writeValue(11, measurements(1));

  //triggers the trajectory to return the next values during the next cycle
    trajectory.update();
}

void Robot::resetController(){
  errorA = 0.0;
    errorB = 0.0;
    eA = 0.0;
    eB = 0.0;
    controlA = 0.0;
    controlB = 0.0;
    uA = 0.0;
    uB = 0.0;
}

void Robot::resetKalmanFilter() {
   // UNCOMMENT AND MODIFY LINES BELOW TO IMPLEMENT THE RESET OF THE KALMAN FILTER
   // Initialize state covariance matrix
   _Phat.Fill(0);       // Initialize the covariance matrix
   _Phat(0,0) = 2.3511e-6;      // Fill the initial covariance matrix, you can change this according to your experiments
   _Phat(1,1) = 1e-6;
   _Phat(2,2) = 1e-6;
  
   // Initialize state estimate
   _xhat(0) = 0;       // Change this according to your experiments
   _xhat(1) = 0;
   _xhat(2) = 0;
  
   // Reset innovation and its covariance matrix
   _S.Fill(0);
   _nu.Fill(0);
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
    message("Controller reset and enabled.");    // Display a message in the status bar of QRoboticsCenter
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

void Robot::button2callback() {
  if(toggleButton(2)) {
    trajectory.start();
    message("Trajectory started/resumed.");
  } else {
    trajectory.stop();
    message("Trajectory stopped.");
  }
}

void Robot::button3callback() {
     _button_states[2] = 0;
    trajectory.reset();
    message("Trajectory reset.");
}
