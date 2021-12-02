
clear all
close all
clear global
clc

%% Loading data from assignment 1

load('C:\Users\matth\Documents\IW4\Control Theory\Project\Control_Theory\Assignment 1\Matlab\assignment1_def.mat')

%Selected model assignment 1
sys = sys_g_SK;

%% 1)b) Design of the PI controller
[GM,PM,Wpi,Wc] = margin(sys);
w = logspace(-2,log10(fs/2),100)'*2*pi;  % *2pi so it's in radian
% % Bode diagram
% figure
% bode(sys)
% grid on
% title('Bodeplot original system')

% System numerator and denumerator in vector format
[sys_num,sys_den] = tfdata(sys, 'v');

% The exact magnitude and phase from the bodeplot in vector format
[mag, phase] = bode(sys,w);
phase_ori = zeros(length(w),1);
mag_ori = zeros(length(w),1);
phase_ori(:) = phase(:,:,:);
mag_ori(:) = mag(:,:,:);

% Bodeplot original system
figure
subplot(211)
semilogx(w, 20*log10(mag_ori))
grid on, axis tight
ylabel('|P(j\omega)| [dB]')
title('Bodeplot original system')
subplot(212)
semilogx(w,phase_ori)
grid on, axis tight
ylabel('\phi(P(j\omega)) [^o]')
xlabel('\omega [rad/s]')

% Specifications
PM_des = 45;		%desired phase margin [degrees]
Dphi_PI = 11;       % allowed phase lag of the PI controller at the crossover frequency 

% Determine the new cross-over pulsation wco
phase_crossover = -180 + PM_des + Dphi_PI;
wco = interp1(phase_ori,w,phase_crossover);	

% Determine Ti [s], such that the phase lag of the PI controller at wco equals Dphi_PI
Ti = 1/(wco * tan(Dphi_PI*pi/180));      % Matlab uses radians!
num_PI = [Ti 1];
den_PI = [Ti 0];

% Transforming the control TF from continuous time to discrete time
contr_PI_c = tf(num_PI, den_PI);
contr_PI_d = c2d(contr_PI_c, Ts, 'zoh') ;

% Open loop system with PI controller
sys_PI = sys * contr_PI_d;
[num_loopgain,den_loopgain] = tfdata(sys_PI, 'v');

% Calculate the gain such that the amplitude at wco equals 1
[mag, phase] = bode(sys_PI,w);
phase_PI = zeros(length(w),1);
mag_PI = zeros(length(w),1);
phase_PI(:) = phase(:,:,:);
mag_PI(:) = mag(:,:,:);
gain = 1/interp1(w,mag_PI,wco);

% Controller and open loop system with correct gain
num_loopgain = num_loopgain*gain;
num_PI = num_PI * gain;
contr_PI_g = tf(num_PI, den_PI, Ts)
sys_PI_g = tf(num_loopgain, den_loopgain, Ts)

[mag, phase] = bode(sys_PI_g,w);
phase_PI_g = zeros(length(w),1);
mag_PI_g = zeros(length(w),1);
phase_PI_g(:) = phase(:,:,:);
mag_PI_g(:) = mag(:,:,:);

% Bodeplot controller
figure
bode(contr_PI_g)
title('Frequency respons PI controller in discrete time')
% Bodeplot open loop systems
figure
subplot(211)
semilogx(w, [20*log10(mag_ori), 20*log10(mag_PI),20*log10(mag_PI_g)])
grid on, axis tight
ylabel('|P(j\omega)| [dB]')
title('Bodeplot original open loop systems')
legend('Original system', 'System with PI controller', 'System with PI with correct gain', 'location', 'NorthEast')
subplot(212)
semilogx(w, [phase_ori, phase_PI, phase_PI_g])
grid on, axis tight
ylabel('\phi(P(j\omega)) [^o]')
xlabel('\omega [rad/s]')

% Parameters open loop system with PI controller 
[GM_PI_g,PM_PI_g,Wpi_PI_g,Wc_PI_g] = margin(sys_PI_g);


% Bode plot closed loop system
sys_cl = feedback(sys_PI_g,1)

[mag, phase] = bode(sys_cl,w);
phase_cl = zeros(length(w),1);
mag_cl = zeros(length(w),1);
phase_cl(:) = phase(:,:,:);
mag_cl(:) = mag(:,:,:);

figure
subplot(211)
semilogx(w, 20*log10(mag_cl))
grid on, axis tight
ylabel('|P(j\omega)| [dB]')
title('Bodeplot closed loop system with PI controller')
subplot(212)
semilogx(w,phase_cl)
grid on, axis tight
ylabel('\phi(P(j\omega)) [^o]')
xlabel('\omega [rad/s]')

%% Step response closed loop system

t = [0:0.01:10];
[sim_step] = step(sys_cl,t);
figure
plot(t, sim_step)
figure
plot(t,1-sim_step)
title('ddphi = 11')
error11 = sum(abs(1-sim_step))
