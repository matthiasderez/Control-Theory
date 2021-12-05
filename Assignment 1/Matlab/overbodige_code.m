% %%% Motor A
% 
% figure, hold on
% bode(sys_1A)
% bode(sys_2A)
% bode(sys_SK_1A)
% bode(sys_SK_2A)
% legend('complex model', 'simple model', 'complex model with SK', 'simple model with SK', 'Location','SouthWest')
% hold off

% --------------------------------------------------------------------------

% % Motor A
% va_est_square = lsim(sys_1A,voltageA,t);
% figure
% subplot(2,1,1),plot(t,[va va_est_square]);
% legend('empirical','estimated','Location','SouthWest')
% xlabel('time [s]')
% ylabel('\omega_a [rad/s]')
% axis tight
% subplot(2,1,2),plot(t,abs(va - va_est_square))
% legend('absolute error')
% xlabel('time [s]')
% ylabel('omega_{a empirical} - \omega_{a estimated} [rad/s]')
% axis tight
% sgtitle('Response complex model to square wavefunction input motor a')
% print -depsc square_square_wave_response_complex_a.eps
% 
% % Motor B
% vb_est_square = lsim(sys_1A,voltageA,t);
% figure
% subplot(2,1,1),plot(t,[vb vb_est_square]);
% legend('empirical','estimated','Location','SouthWest')
% xlabel('time [s]')
% ylabel('\omega_b [rad/s]')
% axis tight
% subplot(2,1,2),plot(t,abs(vb - vb_est_square))
% legend('absolute error')
% xlabel('time [s]')
% ylabel('omega_{b empirical} - \omega_{b estimated} [rad/s]')
% axis tight
% sgtitle('Response complex model to square wavefunction input motor b')
% print -depsc square_square_wave_response_complex_b.eps

% --------------------------------------------------------------------------

% % Motor A square wave input
% va_est1 = lsim(sys_2A,voltageA,t);
% figure 
% subplot(2,1,1),plot(t,[va va_est1]);
% legend('empirical','estimated','Location','SouthWest')
% xlabel('time [s]')
% ylabel('\omega_a [rad/s]')
% axis tight
% subplot(2,1,2),plot(t,abs(va - va_est1))
% legend('absolute error')
% xlabel('time [s]')
% ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
% axis tight
% sgtitle('response simple model to square wavefunction input motor a')
% print -depsc square_square_wave_response_simple_a.eps
% 
% % Motor B square wave input
% va_est1 = lsim(sys_2A,voltageA,t);
% figure 
% subplot(2,1,1),plot(t,[vb va_est1]);
% legend('empirical','estimated','Location','SouthWest')
% xlabel('time [s]')
% ylabel('\omega_b [rad/s]')
% axis tight
% subplot(2,1,2),plot(t,abs(vb - va_est1))
% legend('absolute error')
% xlabel('time [s]')
% ylabel('\omega_b(empirical) - \omega_b(estimated) [rad/s]')
% axis tight
% sgtitle('response simple model to square wavefunction input motor b')
% print -depsc square_square_wave_response_simple_b.eps

% --------------------------------------------------------------------------









