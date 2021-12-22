#include "extended_kalman_filter.h"

void PredictionUpdate(const Matrix<1> &u, Matrix<3> &xhat, Matrix<3,3> &Phat) {
  // // UNCOMMENT AND COMPLETE LINES BELOW TO IMPLEMENT PredictionUpdate OF THE EXTENDED KALMAN FILTER
  // // Define useful constant
   const float L = 0.1259;      //Pendulum length [m]
   const float c = 0;      //Damping coefficient [Nm/s]
   const float g = 9.81;   //Gravitational acceleration [m/s^2]
   const float Ts = 0.01; 
   // Tuning parameter
   float arrayQ[3][3]{ { 2.7878e-9,  0,  0},    //Provide here the element values of weight Q
                       { 0,  9.4124e-9,  0},
                       { 0,  0,  1e-5}};
  
   Matrix<3, 3> Q = arrayQ;
  

//  // Linear Kalman filter
//  float arrayF[3][3]{{1,0,0},
//                      {0,1,0.01/L},
//                      {0,-0.01*g,1}};
//  Matrix <3,3> F = arrayF;
//
//  float arrayG [3][1]{{0.01},{-0.01/L},{0}};
//  Matrix <3,1> G = arrayG;
//  xhat = F * xhat + G*u;
//  Phat = F*Phat*F.Transpose() + Q;
//

  // Extended kalman filter
   // Compute Jacobian of system dynamics
   float arrayJf[3][3]{{1, 0, 0},   //Provide here the element values of the Jacobian of system dynamics
                       {0, 1+Ts*u(0)*sin(xhat(1))/L, Ts/L},
                       {0,  Ts*(-g*cos(xhat(1)) - (xhat(2)-u(0)*cos(xhat(1)))/L*u(0)*cos(xhat(1))-pow(u(0)*sin(xhat(1)),2)/L), 1-Ts*u(0)*sin(xhat(1))/L}};
   Matrix<3, 3> A = arrayJf;
  
   // Evaluate discrete-time nonlinear system dynamics
   float arrayf[3][1]{{ Ts*u(0)+xhat(0) }, //Provide the nonlinear dynamics equation for each state
                      { Ts*(xhat(2)-u(0)*cos(xhat(1)))/L + xhat(1) },
                      { Ts*(-g*sin(xhat(1)) - u(0)*sin(xhat(1))*(xhat(2)-u(0)*cos(xhat(1)))/L)+xhat(2) }};
   xhat = arrayf;    //state prediction is equal to the nonlinear dynamics calculated in arrayf
  
   // Update state covariance: P = APAt + Q, with A equal to the Jacobian of system dynamics
   Phat = A * Phat * A.Transpose() + Q;
}

void CorrectionUpdate(const Matrix<2> &y, Matrix<3> &xhat, Matrix<3,3> &Phat, Matrix<2> &nu, Matrix<2,2> &S) {
   // UNCOMMENT AND COMPLETE LINES BELOW TO IMPLEMENT CorrectionUpdate OF THE EXTENDED KALMAN FILTER
  // Tuning parameter
   float arrayR[2][2]{{2.7878e-9, 0},    //Provide here the element values of weight R
                      {0, 9.4124e-6}};
   Matrix<2, 2> R = arrayR;
  // System C-matrix - Compute Jacobian of measurement equation
   float arrayJh[2][3]{{1, 0, 0}, {0, 1, 0}}; //Provide here the element values of the Jacobian of measurement equation
                       
   Matrix<2,3> C = arrayJh;
  
   // Evaluate measurement equation
   Matrix<2> h = C*xhat;
  
   // Compute innovation
   nu = y - h;
  
   // Compute innovation covariance
   S = C * Phat * C.Transpose() + R;
  
   // Compute optimal Kalman filter gain
   Matrix<3,2> L = Phat * C.Transpose() * S.Inverse();
  
   // Compute corrected system state estimate
   xhat += L * nu;
  
   // Compute corrected state estimate covariance
   Phat -= L * C * Phat;
}
