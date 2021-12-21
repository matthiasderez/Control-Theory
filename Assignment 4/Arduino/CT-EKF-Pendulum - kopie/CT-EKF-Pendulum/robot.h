#ifndef ROBOT_H
#define ROBOT_H

/*
 * ROBOT Class
 *
 * Class incorporating the robot. This class is used to define state machines,
 * control algorithms, sensor readings,...
 * It should be interfaced with the communicator to send data to the world.
 *
 */

#include "mecotron.h" // Include MECOTRON header
#include <BasicLinearAlgebra.h> // Include BasicLinearAlgebra to make matrix manipulations easier
#include "extended_kalman_filter.h" // Include template to make extended Kalman filter implementation easier

#define PENDULUM
#include <trajectory.h> // Include trajectory, for assignment 4

class Robot : public MECOtron {
  private:

    // Class variables
    Trajectory trajectory; // define the reference trajectory object

    // Kalman filter
    Matrix<3> _xhat;       // state estimate vector
    Matrix<3,3> _Phat;     // state estimate covariance
    Matrix<1> _nu;         // innovation vector
    Matrix<1,1> _S;        // innovation covariance

    // Position controller
    Matrix<3> xref;        // reference state
    Matrix<1> desiredVelocityCart;  // control signal
    float r;
    // Velocity controller
    float time;
    float a1;
    float a2;
    float a3;
    float a4;
    float b1;
    float b2;
    float b3;
    float b4;
    float uA;
    float uB;
    float errorA;
    float errorB;
    float counter;
    float controlA;
    float controlB;
    float eA;
    float eB;
    Matrix<1,1> Newmeas;
  public:
    // Constructor
    Robot() { }

    void control();

    // General functions
    bool init();  // Set up the robot

    bool controlEnabled();
    bool KalmanFilterEnabled();

    void resetController();
    void resetKalmanFilter();

    void button0callback();
    void button1callback();
    void button2callback();
    void button3callback();

};

#endif // ROBOT_H
