
clear all
close all
clear global
clc

%% Loading data from assignment 1

load('C:\Users\matth\Documents\IW4\Control Theory\Project\Control_Theory\Assignment 1\Matlab\assignment1_def.mat')
sys = sys_g_SK;
%% 1)b) Design of the PI controller
[Gm,Pm,Wpi,Wc] = margin(sys);
Wpi = Wpi; % in rad/s
Wc = Wc; % in rad/s
w = logspace(-2,log10(fs/2),100)*2*pi;
% Bode diagram
figure
bode(sys)
grid on
title('Bodeplot original system')


[mag, phase] = bode(sys,w);
phase_P = zeros(length(w),1);
phase_P(:) = phase(:,:,:);
mag_P(:) = mag(:,:,:);

% controle
figure
subplot(211)
semilogx(w, 20*log10(mag_P))
grid on, axis tight
ylabel('|P(j\omega)| [dB]')
title('Bodeplot original system')
subplot(212)
semilogx(w,phase_P)
grid on, axis tight
ylabel('\phi(P(j\omega)) [^o]')
xlabel('w [rad/s]')

% Specifications
PM_des = 45;		%desired phase margin [degrees]
Dphi_PI = 12;       % allowed phase lag of the PI controller at the crossover frequency 

% Determine the new cross-over pulsation wco
phase_crossover = -180 + PM_des + Dphi_PI;
wco = interp1(phase_P,w,phase_crossover);	

% Determine Ti [s], such that the phase lag of the PI controller at wco equals Dphi_PI
Ti = 1/(wco * tan(Dphi_PI*pi/180));      % Matlab uses radians!
num_PI = [Ti 1];
den_PI = [Ti 0];

% Calculate the gain such that the amplitude at wco equals 1
[sys_num,sys_den] = tfdata(sys, 'v');

% Bodeplot of the compensated system

contr_PI_c = tf(num_PI, den_PI)
contr_PI_d = c2d(contr_PI_c, Ts, 'zoh') 

%contr_PI_2 = tf(num_PI, den_PI, Ts)


sys_tot = sys * contr_PI_d
[num_loopgain,den_loopgain] = tfdata(sys_tot, 'v')
[mag, phase] = bode(sys_tot,w);
phase_loop = zeros(length(w),1);
mag_loop = zeros(length(w),1);
phase_loop(:) = phase(:,:,:);
mag_loop(:) = mag(:,:,:);
gain = 1/interp1(w,mag_loop,wco)


figure
hold on
bode(sys)
bode(sys_tot)
grid on
legend('Original system', 'System with PI controller', 'location', 'SouthWest')
hold off


num_loopgain = num_loopgain*gain;
num_PI = num_PI * gain;

sys_tot2 = tf(num_loopgain, den_loopgain, Ts)
[mag, phase] = bode(sys_tot2,w);
phase_loop2 = zeros(length(w),1);
mag_loop2 = zeros(length(w),1);
phase_loop2(:) = phase(:,:,:);
mag_loop2(:) = mag(:,:,:);


sys_PI = tf(num_loopgain, den_loopgain);

figure
hold on
bode(sys)
bode(sys_tot)
bode(sys_tot2)
grid on
legend('Original system', 'System with PI controller', 'System with PI with correct gain', 'location', 'SouthWest')
hold off


