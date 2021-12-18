clear all
close all
clear global
clc



%% Selected models assignment 1
Ts = 0.01;
fs = 1/Ts;
numA =  0.5689;
denA = [1 -0.7191 0];
sysA = tf(numA, denA, Ts);
numB = 0.6164;
denB = [1 -0.6928 0];
sysB = tf(numB, denB, Ts);

%% 1)b) Design of the PI controller
[GMA,PMA,WpiA,WcA] = margin(sysA);
[GMB,PMB,WpiB,WcB] = margin(sysB);

w = logspace(-2,log10(fs/2),100)'*2*pi;  % *2pi so it's in radian

% The exact magnitude and phase from the bodeplot in vector format
[mag, phase] = bode(sysA,w);
phase_oriA = zeros(length(w),1);
mag_oriA = zeros(length(w),1);
phase_oriA(:) = phase(:,:,:);
mag_oriA(:) = mag(:,:,:);

[mag, phase] = bode(sysB,w);
phase_oriB = zeros(length(w),1);
mag_oriB = zeros(length(w),1);
phase_oriB(:) = phase(:,:,:);
mag_oriB(:) = mag(:,:,:);


% Bodeplot original system A
figure
subplot(211)
semilogx(w, 20*log10(mag_oriA))
grid on, axis tight
ylabel('|P(j\omega_a)| [dB]')
set(gca, 'FontSize', 11)
title('Bodeplot original system motor A')
subplot(212)
semilogx(w,phase_oriA)
grid on, axis tight
ylabel('\phi(P(j\omega_a)) [^o]')
xlabel('\omega_a [rad/s]')
print -depsc bodeplot_original_A.eps


% Bodeplot original system B
figure
subplot(211)
semilogx(w, 20*log10(mag_oriB))
grid on, axis tight
ylabel('|P(j\omega_b)| [dB]')
set(gca, 'FontSize', 11)
title('Bodeplot original system motor B')
subplot(212)
semilogx(w,phase_oriB)
grid on, axis tight
ylabel('\phi(P(j\omega_b)) [^o]')
xlabel('\omega_B [rad/s]')
print -depsc bodeplot_original_B.eps


% Specifications
PM_des = 50;		%desired phase margin [degrees]
Dphi_PI = 15;       % allowed phase lag of the PI controller at the crossover frequency 

% Determine the new cross-over pulsation wco
phase_crossoverA = -180 + PM_des + Dphi_PI;
wcoA = interp1(phase_oriA,w,phase_crossoverA);	

phase_crossoverB = -180 + PM_des + Dphi_PI;
wcoB = interp1(phase_oriB,w,phase_crossoverB);

% Determine Ti [s], such that the phase lag of the PI controller at wco equals Dphi_PI
Ti_A = 1/(wcoA * tan(Dphi_PI*pi/180));      
num_PI_A = [Ti_A 1];
den_PI_A = [Ti_A 0];

Ti_B = 1/(wcoB * tan(Dphi_PI*pi/180));      
num_PI_B = [Ti_B 1];
den_PI_B = [Ti_B 0];

% Transforming the control TF from continuous time to discrete time
% continu D(s) = K/s(s+1/Ti) = V(s)/E(s) = K(1 + 1/(Ti*s))
% V(s) = E(s) (K + K/(s*Ti))
contrA_PI_c = tf(num_PI_A, den_PI_A);
contrA_PI_d = c2d(contrA_PI_c, Ts, 'zoh') ;
[num_PI_A,den_PI_A] = tfdata(contrA_PI_d, 'v');

contrB_PI_c = tf(num_PI_B, den_PI_B);
contrB_PI_d = c2d(contrB_PI_c, Ts, 'zoh') ;
[num_PI_B,den_PI_B] = tfdata(contrB_PI_d, 'v');


% Open loop system with PI controller
sysA_PI = sysA * contrA_PI_d;
[num_loopgainA,den_loopgainA] = tfdata(sysA_PI, 'v')

sysB_PI = sysB * contrB_PI_d;
[num_loopgainB,den_loopgainB] = tfdata(sysB_PI, 'v');

% Calculate the gain such that the amplitude at wco equals 1
[mag, phase] = bode(sysA_PI,w);
phase_PI_A = zeros(length(w),1);
mag_PI_A = zeros(length(w),1);
phase_PI_A(:) = phase(:,:,:);
mag_PI_A(:) = mag(:,:,:);

[mag, phase] = bode(sysB_PI,w);
phase_PI_B = zeros(length(w),1);
mag_PI_B = zeros(length(w),1);
phase_PI_B(:) = phase(:,:,:);
mag_PI_B(:) = mag(:,:,:);

gainA = 1/interp1(w,mag_PI_A,wcoA);
gainB = 1/interp1(w,mag_PI_B,wcoB);

% Controller and open loop system A with correct gain
sysA_PI_wogain = sysA_PI;
contrA_PI_wogain = contrA_PI_d;
num_loopgainA = num_loopgainA*gainA;
num_PI_A = num_PI_A * gainA;
contrA_PI = tf(num_PI_A, den_PI_A, Ts);
sysA_PI = tf(num_loopgainA, den_loopgainA, Ts);

[mag, phase] = bode(sysA_PI,w);
phase_PI_g_A = zeros(length(w),1);
mag_PI_g_A = zeros(length(w),1);
phase_PI_g_A(:) = phase(:,:,:);
mag_PI_g_A(:) = mag(:,:,:);

sysB_PI_wogain = sysB_PI;
contrB_PI_wogain = contrB_PI_d;
num_loopgainB = num_loopgainB*gainB;
num_PI_B = num_PI_B * gainB;
contrB_PI = tf(num_PI_B, den_PI_B, Ts);
sysB_PI = tf(num_loopgainB, den_loopgainB, Ts);

[mag, phase] = bode(sysB_PI,w);
phase_PI_g_B = zeros(length(w),1);
mag_PI_g_B = zeros(length(w),1);
phase_PI_g_B(:) = phase(:,:,:);
mag_PI_g_B(:) = mag(:,:,:);


% Bodeplot controller
figure
bode(contrA_PI,w)
grid on
set(gca, 'FontSize', 11)
title('Frequency respons PI controller motor A')
print -depsc bodeplot_controllerA_method1.eps

figure
bode(contrB_PI,w)
grid on
set(gca, 'FontSize', 11)
title('Frequency respons PI controller motor B')
print -depsc bodeplot_controllerB_method1.eps

% Bodeplot open loop systems
figure
subplot(211)
semilogx(w, [20*log10(mag_oriA), 20*log10(mag_PI_A),20*log10(mag_PI_g_A)])
grid on, axis tight
ylabel('|P(j\omega_A)| [dB]')
set(gca, 'FontSize', 11)
title('Bodeplot open loop systems motor A')
legend('Original system motor A', 'System motor A with PI controller', 'System motor A with PI with correct gain', 'location', 'NorthEast')
subplot(212)
semilogx(w, [phase_oriA, phase_PI_A, phase_PI_g_A])
grid on, axis tight
ylabel('\phi(P(j\omega_A)) [^o]')
xlabel('\omega_A [rad/s]')
print -depsc bodeplotA_openloop_method1.eps

figure
subplot(211)
semilogx(w, [20*log10(mag_oriB), 20*log10(mag_PI_B),20*log10(mag_PI_g_B)])
grid on, axis tight
ylabel('|P(j\omega_B)| [dB]')
set(gca, 'FontSize', 11)
title('Bodeplot open loop systems motor B')
legend('Original system motor B', 'System motor B with PI controller', 'System motor B with PI with correct gain', 'location', 'NorthEast')
subplot(212)
semilogx(w, [phase_oriB, phase_PI_B, phase_PI_g_B])
grid on, axis tight
ylabel('\phi(P(j\omega_B)) [^o]')
xlabel('\omega_B [rad/s]')
print -depsc bodeplotB_openloop_method1.eps

figure
margin(sysA_PI)
print -depsc bodeplot_contr_sysA_openloop_method1.eps

figure
margin(sysB_PI)
print -depsc bodeplot_contr_sysB_openloop_method1.eps

% Parameters open loop system with PI controller 
[GM_PI_g_A,PM_PI_g_A,Wpi_PI_g_A,Wc_PI_g_A] = margin(sysA_PI);

[GM_PI_g_B,PM_PI_g_B,Wpi_PI_g_B,Wc_PI_g_B] = margin(sysB_PI);


% Bode plot closed loop system A
sysA_cl = feedback(sysA_PI,1);

[mag, phase] = bode(sysA_cl,w);
phase_clA = zeros(length(w),1);
mag_clA = zeros(length(w),1);
phase_clA(:) = phase(:,:,:);
mag_clA(:) = mag(:,:,:);

figure
subplot(211)
semilogx(w, 20*log10(mag_clA))
grid on, axis tight
ylabel('|P(j\omega_A)| [dB]')
set(gca, 'FontSize', 11)
title('Bodeplot closed loop system A with PI controller')
subplot(212)
semilogx(w,phase_clA)
grid on, axis tight
ylabel('\phi(P(j\omega_A)) [^o]')
xlabel('\omega_A [rad/s]')
print -depsc bodeplotA_cl_method1.eps

% Bode plot closed loop system B
sysB_cl = feedback(sysB_PI,1);

[mag, phase] = bode(sysB_cl,w);
phase_clB = zeros(length(w),1);
mag_clB = zeros(length(w),1);
phase_clB(:) = phase(:,:,:);
mag_clB(:) = mag(:,:,:);

figure
subplot(211)
semilogx(w, 20*log10(mag_clB))
grid on, axis tight
ylabel('|P(j\omega_B)| [dB]')
set(gca, 'FontSize', 11)
title('Bodeplot closed loop system motor B with PI controller')
subplot(212)
semilogx(w,phase_clB)
grid on, axis tight
ylabel('\phi(P(j\omega_B)) [^o]')
xlabel('\omega_B [rad/s]')
print -depsc bodeplotB_cl_method1.eps


%% Step response closed loop system 

t = [0:0.01:3];
[sim_stepA] = step(sysA_cl,t);
figure
plot(t, sim_stepA)
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
set(gca, 'FontSize', 11)
title('Respons motor A for a step input as reference velocity')
print -depsc stepresponsA_cl_method1.eps


error1A = sum(abs(1-sim_stepA));
BW_1A = bandwidth(sysA_cl);

[sim_stepB] = step(sysB_cl,t);
figure
plot(t, sim_stepB)
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
set(gca, 'FontSize', 11)
title('Respons motor B for a step input as reference velocity')
print -depsc stepresponsB_cl_method1.eps


error1B = sum(abs(1-sim_stepB));
BW_1B = bandwidth(sysB_cl);

%% Same procedure but with adapted gain

% Specifications
PM_des = 50;		%desired phase margin [degrees]
Dphi_PI = 15;       % allowed phase lag of the PI controller at the crossover frequency 

% Determine the new cross-over pulsation wco
phase_crossoverA = -180 + PM_des + Dphi_PI;
wcoA = interp1(phase_oriA,w,phase_crossoverA);	

phase_crossoverB = -180 + PM_des + Dphi_PI;
wcoB = interp1(phase_oriB,w,phase_crossoverB);

% Determine Ti [s], such that the phase lag of the PI controller at wco equals Dphi_PI
Ti_A = 1/(wcoA * tan(Dphi_PI*pi/180));      % Matlab uses radians!
num_PI_A = [Ti_A 1];
den_PI_A = [Ti_A 0];

Ti_B = 1/(wcoB * tan(Dphi_PI*pi/180));      % Matlab uses radians!
num_PI_B = [Ti_B 1];
den_PI_B = [Ti_B 0];

% Transforming the control TF from continuous time to discrete time
contrA_PI_c = tf(num_PI_A, den_PI_A);
contrA_PI_d = c2d(contrA_PI_c, Ts, 'zoh') ;
[num_PI_A,den_PI_A] = tfdata(contrA_PI_d, 'v');

contrB_PI_c = tf(num_PI_B, den_PI_B);
contrB_PI_d = c2d(contrB_PI_c, Ts, 'zoh') ;
[num_PI_B,den_PI_B] = tfdata(contrB_PI_d, 'v');


% Open loop system A with PI controller
sysA_PI = sysA * contrA_PI_d;
[num_loopgainA,den_loopgainA] = tfdata(sysA_PI, 'v');

sysB_PI = sysB * contrB_PI_d;
[num_loopgainB,den_loopgainB] = tfdata(sysB_PI, 'v');

% Calculate the gain such that the amplitude at wco equals 1
[mag, phase] = bode(sysA_PI,w);
phase_PI_A = zeros(length(w),1);
mag_PI_A = zeros(length(w),1);
phase_PI_A(:) = phase(:,:,:);
mag_PI_A(:) = mag(:,:,:);

[mag, phase] = bode(sysB_PI,w);
phase_PI_B = zeros(length(w),1);
mag_PI_B = zeros(length(w),1);
phase_PI_B(:) = phase(:,:,:);
mag_PI_B(:) = mag(:,:,:);

gainA = 0.8;
gainB = 0.775;
% Controller and open loop system A with correct gain
sysA_PI_wogain = sysA_PI;
contrA_PI_wogain = contrA_PI_d;
num_loopgainA = num_loopgainA*gainA;
num_PI_A = num_PI_A * gainA;
contrA_PI = tf(num_PI_A, den_PI_A, Ts);
sysA_PI = tf(num_loopgainA, den_loopgainA, Ts);

[mag, phase] = bode(sysA_PI,w);
phase_PI_g_A = zeros(length(w),1);
mag_PI_g_A = zeros(length(w),1);
phase_PI_g_A(:) = phase(:,:,:);
mag_PI_g_A(:) = mag(:,:,:);

sysB_PI_wogain = sysB_PI;
contrB_PI_wogain = contrB_PI_d;
num_loopgainB = num_loopgainB*gainB;
num_PI_B = num_PI_B * gainB;
contrB_PI = tf(num_PI_B, den_PI_B, Ts);
sysB_PI = tf(num_loopgainB, den_loopgainB, Ts);

[mag, phase] = bode(sysB_PI,w);
phase_PI_g_B = zeros(length(w),1);
mag_PI_g_B = zeros(length(w),1);
phase_PI_g_B(:) = phase(:,:,:);
mag_PI_g_B(:) = mag(:,:,:);


% Bodeplot controller
figure
bode(contrA_PI,w)
grid on
set(gca, 'FontSize', 11)
title('Frequency respons PI controller motor A')
print -depsc bodeplot_controllerA_method2.eps

figure
bode(contrB_PI,w)
grid on
set(gca, 'FontSize', 11)
title('Frequency respons PI controller motor B')
print -depsc bodeplot_controllerB_method2.eps

% Bodeplot open loop systems
figure
subplot(211)
semilogx(w, [20*log10(mag_oriA), 20*log10(mag_PI_A),20*log10(mag_PI_g_A)])
grid on, axis tight
ylabel('|P(j\omega_A)| [dB]')
set(gca, 'FontSize', 11)
title('Bodeplot open loop systems motor A')
legend('Original system motor A', 'System motor A with PI controller', 'System motor A with PI with correct gain', 'location', 'NorthEast')
subplot(212)
semilogx(w, [phase_oriA, phase_PI_A, phase_PI_g_A])
grid on, axis tight
ylabel('\phi(P(j\omega_A)) [^o]')
xlabel('\omega_A [rad/s]')
print -depsc bodeplotA_openloop_method2.eps

figure
subplot(211)
semilogx(w, [20*log10(mag_oriB), 20*log10(mag_PI_B),20*log10(mag_PI_g_B)])
grid on, axis tight
ylabel('|P(j\omega_B)| [dB]')
set(gca, 'FontSize', 11)
title('Bodeplot open loop systems motor B')
legend('Original system motor B', 'System motor B with PI controller', 'System motor B with PI with correct gain', 'location', 'NorthEast')
subplot(212)
semilogx(w, [phase_oriB, phase_PI_B, phase_PI_g_B])
grid on, axis tight
ylabel('\phi(P(j\omega_B)) [^o]')
xlabel('\omega_B [rad/s]')
print -depsc bodeplotB_openloop_method2.eps

figure
margin(sysA_PI)
print -depsc bodeplot_contr_sysA_openloop_method2.eps

figure
margin(sysB_PI)
print -depsc bodeplot_contr_sysB_openloop_method2.eps

% Parameters open loop system A with PI controller 
[GM_PI_g_A,PM_PI_g_A,Wpi_PI_g_A,Wc_PI_g_A] = margin(sysA_PI);

[GM_PI_g_B,PM_PI_g_B,Wpi_PI_g_B,Wc_PI_g_B] = margin(sysB_PI);


% Bode plot closed loop system A
sysA_cl = feedback(sysA_PI,1);

[mag, phase] = bode(sysA_cl,w);
phase_clA = zeros(length(w),1);
mag_clA = zeros(length(w),1);
phase_clA(:) = phase(:,:,:);
mag_clA(:) = mag(:,:,:);

figure
subplot(211)
semilogx(w, 20*log10(mag_clA))
grid on, axis tight
ylabel('|P(j\omega_A)| [dB]')
set(gca, 'FontSize', 11)
title('Bodeplot closed loop system A with PI controller')
subplot(212)
semilogx(w,phase_clA)
grid on, axis tight
ylabel('\phi(P(j\omega_A)) [^o]')
xlabel('\omega_A [rad/s]')
print -depsc bodeplotA_cl_method2.eps

% Bode plot closed loop system B
sysB_cl = feedback(sysB_PI,1);

[mag, phase] = bode(sysB_cl,w);
phase_clB = zeros(length(w),1);
mag_clB = zeros(length(w),1);
phase_clB(:) = phase(:,:,:);
mag_clB(:) = mag(:,:,:);

figure
subplot(211)
semilogx(w, 20*log10(mag_clB))
grid on, axis tight
ylabel('|P(j\omega_B)| [dB]')
set(gca, 'FontSize', 11)
title('Bodeplot closed loop system motor B with PI controller')
subplot(212)
semilogx(w,phase_clB)
grid on, axis tight
ylabel('\phi(P(j\omega_B)) [^o]')
xlabel('\omega_B [rad/s]')
print -depsc bodeplotB_cl_method2.eps


%% Step response closed loop system 

t = [0:0.01:3];
[sim_stepA] = step(sysA_cl,t);
figure
plot(t, sim_stepA)
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
set(gca, 'FontSize', 11)
title('Respons motor A for a step input as reference velocity')
print -depsc stepresponsA_cl_method2.eps


error2A = sum(abs(1-sim_stepA));
BW_2A = bandwidth(sysA_cl);

[sim_stepB] = step(sysB_cl,t);
figure
plot(t, sim_stepB)
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
set(gca, 'FontSize', 11)
title('Respons motor B for a step input as reference velocity')
print -depsc stepresponsB_cl_method2.eps


error2B = sum(abs(1-sim_stepB));
BW_2B = bandwidth(sysB_cl);

%% Resultaten

% %Motor a: K = 0.8, PM = 50°, 6rad/s
% %Motor b: K = 0.775, PM = 50°, 6rad/s
% csvfile = '../Data/step_controller12.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% step_controller12 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save step_controller12
load step_controller12.mat

% %Motor a: K = 0.8, PM = 50°, 6rad/s
% %Motor b: K = 0.775, PM = 50°, 6rad/s
% %Placed on a slope of 3.518°
% csvfile = '../Data/step_controller13.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% step_controller13 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save step_controller13
load step_controller13.mat

step_controller = step_controller12;
voltageA = step_controller(1:400,2);
voltageB = step_controller(1:400,3);
va  = step_controller(1:400, 6); % velocity motor a
vb = step_controller(1:400, 7); % velocity motor b
t = step_controller(1:400,10);
eA = step_controller(1:400,12);
wdes = zeros(length(t),1);
wdes(2:length(t)) = 6;
[sim_stepA] = 6*step(sysA_cl,t);
[sim_stepB] = 6*step(sysB_cl,t);

% Adding an extra zero in front because we have our first non zero value
% for wdesired at t = 0.01, while the step function has the first non zero
% input for t = 0. 
[sim_stepA] = [0 sim_stepA(1:(length(t)-1))']';
[sim_stepB] = [0 sim_stepB(1:(length(t)-1))']';



figure
plot(t,[va,sim_stepA,wdes])
legend('measured', 'simulated', 'step reference', 'Location', 'SouthEast')
set(gca, 'FontSize', 11)
title('Different step responses motor A')
ylabel('\omega_a[rad/s]')
xlabel('time [s]')
print -depsc comparison_stepresponseA.eps


figure
plot(t,[va,sim_stepA,wdes])
legend('measured', 'simulated', 'step reference', 'Location', 'SouthEast')
set(gca, 'FontSize', 11)
title('Different step responses motor A')
xlim([0 0.5])
ylim([2.8 7])
ylabel('\omega_a[rad/s]')
xlabel('time [s]')
print -depsc comparison_stepresponseA_zoom.eps


figure
plot(t,[wdes-va,wdes-sim_stepA])
legend('measured', 'simulated', 'Location', 'NorthEast')
set(gca, 'FontSize', 11)
title('Tracking error of the step reference of motor A')
ylabel('step reference - \omega_a [rad/s]')
xlabel('time [s]')
print -depsc trackingerror_stepresponseA.eps


figure
plot(t,[wdes-va,wdes-sim_stepA])
legend('measured', 'simulated', 'Location', 'NorthEast')
set(gca, 'FontSize', 11)
xlim([0 0.5])
ylim([-0.7 6.5])
title('Tracking error of the step reference of motor A')
ylabel('step reference - \omega_a[rad/s]')
xlabel('time [s]')
print -depsc trackingerror_stepresponseA_zoom.eps



figure
plot(t,[vb,sim_stepB,wdes])
legend('measured', 'simulated', 'step reference', 'Location', 'SouthEast' )
set(gca, 'FontSize', 11)
title('Different step responses motor B')
ylabel('\omega_b[rad/s]')
xlabel('time [s]')
print -depsc comparison_stepresponseB.eps


figure
plot(t,[vb,sim_stepB,wdes])
legend('measured', 'simulated', 'step reference', 'Location', 'SouthEast' )
set(gca, 'FontSize', 11)
title('Different step responses motor B')
xlim([0 0.5])
ylim([2.8 7])
ylabel('\omega_b[rad/s]')
xlabel('time [s]')
print -depsc comparison_stepresponseB_zoom.eps



figure
plot(t,[wdes-vb,wdes-sim_stepB])
legend('measured', 'simulated', 'Location', 'NorthEast')
set(gca, 'FontSize', 11)
title('Tracking error of the step reference of motor B')
ylabel('step reference - \omega_b[rad/s]')
xlabel('time [s]')
print -depsc trackingerror_stepresponseB.eps


figure
plot(t,[wdes-vb,wdes-sim_stepB])
legend('measured', 'simulated', 'Location', 'NorthEast')
set(gca, 'FontSize', 11)
xlim([0 0.5])
ylim([-0.7 6.5])
ylabel('step reference - \omega_b[rad/s]')
xlabel('time [s]')
title('Tracking error of the step reference of motor B')
print -depsc trackingerror_stepresponseB_zoom.eps




[sim_stepA] = 6*step(sysA_cl,t);
[sim_stepB] = 6*step(sysB_cl,t);
[sim_voltageA] = lsim(contrA_PI, 6-sim_stepA,t);
[sim_voltageB] = lsim(contrB_PI, 6-sim_stepB,t);
% Adding an extra zero in front because we have our first non zero value
% for wdesired at t = 0.01, while the step function has the first non zero
% input for t = 0.
[sim_voltageA] = [0 sim_voltageA(1:(length(t)-1))']';
[sim_voltageB] = [0 sim_voltageB(1:(length(t)-1))']';

figure
plot(t,[voltageA, sim_voltageA])
legend('measured', 'simulated', 'Location', 'SouthEast' )
set(gca, 'FontSize', 11)
title('Voltage motor A for step response')
ylabel('voltage Motor A [V]')
xlabel('time [s]')
print -depsc comparison_voltage_stepresponseA.eps


figure
plot(t,[voltageB, sim_voltageB])
legend('measured', 'simulated', 'Location', 'SouthEast' )
set(gca, 'FontSize', 11)
title('Voltage motor B for step response')
ylabel('voltage Motor B [V]')
xlabel('time [s]')
print -depsc comparison_voltage_stepresponseB.eps



% figure
% plot(t,eA)
% ylabel('eA')
% xlabel('time [s]')


% Force disturbance
step_controller = step_controller13;
voltageA = step_controller(1:400,2);
voltageB = step_controller(1:400,3);
va  = step_controller(1:400, 6); % velocity motor a
vb = step_controller(1:400, 7); % velocity motor b
t = step_controller(1:400,10);
eA = step_controller(1:400,12);
[sim_stepA] = 6*step(sysA_cl,time);
[sim_stepB] = 6*step(sysB_cl,time);
% Adding an extra zero in front because we have our first non zero value
% for wdesired at t = 0.01, while the step function has the first non zero
% input for t = 0.
[sim_stepA] = [0 sim_stepA(1:(length(t)-1))']';
[sim_stepB] = [0 sim_stepB(1:(length(t)-1))']';
wdes = zeros(length(t),1);
wdes(2:length(t)) = 6;

figure
plot(t,[va,sim_stepA,wdes])
legend('measured', 'simulated', 'step reference', 'Location', 'SouthEast')
set(gca, 'FontSize', 11)
title({'Different step responses motor A',' with force disturbance'})
ylabel('\omega_a[rad/s]')
xlabel('time [s]')
print -depsc comparison_stepresponseA_FD.eps


figure
plot(t,[wdes-va,wdes-sim_stepA])
legend('measured', 'simulated', 'Location', 'NorthEast')
set(gca, 'FontSize', 11)
title({'Tracking error of the step reference of motor A',' with force disturbance'})
ylabel('step reference - \omega_a[rad/s]')
xlabel('time [s]')
print -depsc trackingerror_stepresponseA_FD.eps


figure
plot(t,[vb,sim_stepB,wdes])
legend('measured', 'simulated', 'step reference', 'Location', 'SouthEast' )
set(gca, 'FontSize', 11)
title({'Different step responses motor B',' with force disturbance'})
ylabel('\omega_b[rad/s]')
xlabel('time [s]')
print -depsc comparison_stepresponseB_FD.eps


figure
plot(t,[wdes-vb,wdes-sim_stepB])
legend('measured', 'simulated', 'Location', 'NorthEast')
set(gca, 'FontSize', 11)
title({'Tracking error of the step reference of motor B',' with force disturbance'})
ylabel('step reference - \omega_a[rad/s]')
xlabel('time [s]')
print -depsc trackingerror_stepresponseB_FD.eps


[sim_stepA] = 6*step(sysA_cl,t);
[sim_stepB] = 6*step(sysB_cl,t);
[sim_voltageA] = lsim(contrA_PI, 6-sim_stepA,t);
[sim_voltageB] = lsim(contrB_PI, 6-sim_stepB,t);
% Adding an extra zero in front because we have our first non zero value
% for wdesired at t = 0.01, while the step function has the first non zero
% input for t = 0.
[sim_voltageA] = [0 sim_voltageA(1:(length(t)-1))']';
[sim_voltageB] = [0 sim_voltageB(1:(length(t)-1))']';

figure
plot(t,[voltageA, sim_voltageA])
legend('measured', 'simulated', 'Location', 'SouthEast' )
set(gca, 'FontSize', 11)
title({'Voltage motor A for step response' , 'with force disturbance'})
ylabel('voltage Motor A [V]')
xlabel('time [s]')
print -depsc comparison_voltage_stepresponseA_FD.eps


figure
plot(t,[voltageB, sim_voltageB])
legend('measured', 'simulated', 'Location', 'SouthEast' )
set(gca, 'FontSize', 11)
title({'Voltage motor B for step response', 'with force disturbance'})
ylabel('voltage Motor B [V]')
xlabel('time [s]')
print -depsc comparison_voltage_stepresponseB_FD.eps






