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
#include "kalman_filter.h" // Include template to make Kalman filter implementation easier

class Robot : public MECOtron {
  private:

    // Class variables

    // Kalman filter
    Matrix<1> _xhat;      // state estimate vector
    Matrix<1,1> _Phat;    // state estimate covariance
    Matrix<1> _nu;        // innovation vector
    Matrix<1,1> _S;       // innovation covariance

    // Position controller
    Matrix<1> xref;       // reference state
    Matrix<1,1> K;        // state feedback gain
    Matrix<1> desired_velocity; //control signal

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

};

#endif // ROBOT_H
