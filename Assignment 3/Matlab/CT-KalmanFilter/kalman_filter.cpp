#include "kalman_filter.h"

void PredictionUpdate(const Matrix<1> &u, Matrix<1> &xhat, Matrix<1,1> &Phat) {
  // // UNCOMMENT AND COMPLETE LINES BELOW TO IMPLEMENT PredictionUpdate OF THE KALMAN FILTER
  // Tuning parameter
  // float arrayQ[1][1]{{ ? )}}; //Provide here the element values of weight Q
  // Matrix<1,1> Q = arrayQ;
  //
  // // System A&B-matrix
  // float arrayA[1][1]{{ ? }}; //Provide here the element values of state-space matrix A
  // Matrix<1,1> A = arrayA;
  // float arrayB[1][1]{{ ? }}; //Provide here the element values of state-space matrix B
  // Matrix<1,1> B = arrayB;
  //
  // // Evaluate discrete-time system dynamics
  // xhat = A*xhat + B*u;
  //
  // // Update state covariance: P = APAt + Q
  // Phat = A * Phat * A.Transpose() + Q;
}

void CorrectionUpdate(const Matrix<1> &y, Matrix<1> &xhat, Matrix<1,1> &Phat, Matrix<1> &nu, Matrix<1,1> &S) {
  // // UNCOMMENT AND COMPLETE LINES BELOW TO IMPLEMENT CorrectionUpdate OF THE KALMAN FILTER
  // // Tuning parameter
  // float arrayR[1][1]{{ ? }}; //Provide here the element values of weight R
  // Matrix<1,1> R = arrayR;
  //
  // // System C-matrix - measurement equation
  // float arrayC[1][1]{{ ? }}; //Provide here the element values of state-space matrix C
  // Matrix<1,1> C = arrayC;
  //
  // // Compute innovation
  // nu = y - C*xhat;
  //
  // // Compute innovation covariance
  // S = C * Phat * C.Transpose() + R;
  //
  // // Compute optimal Kalman filter gain
  // Matrix<1,1> L = Phat * C.Transpose() * S.Inverse();
  //
  // // Compute corrected system state estimate
  // xhat += L * nu;
  //
  // // Compute corrected state estimate covariance
  // Phat -= L * C * Phat;
}
