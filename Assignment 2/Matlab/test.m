
clear all
%close all
clear global
clc


%% Loading data from assignment 1

%Selected models assignment 1
Ts = 0.01;
fs = 1/Ts;
numB = 0.6164;
denB = [1 -0.6928 0];
sysA = tf(numB, denB, Ts)


%% 1)b) Design of the PI controller
[GMA,PMA,WpiA,WcA] = margin(sysA);

w = logspace(-2,log10(fs/2),100)'*2*pi;  % *2pi so it's in radian

% The exact magnitude and phase from the bodeplot in vector format
[mag, phase] = bode(sysA,w);
phase_oriA = zeros(length(w),1);
mag_oriA = zeros(length(w),1);
phase_oriA(:) = phase(:,:,:);
mag_oriA(:) = mag(:,:,:);



% Bodeplot original system A
figure
subplot(211)
semilogx(w, 20*log10(mag_oriA))
grid on, axis tight
ylabel('|P(j\omeg_A)| [dB]')
title('Bodeplot original system motor A')
subplot(212)
semilogx(w,phase_oriA)
grid on, axis tight
ylabel('\phi(P(j\omeg_A)) [^o]')
xlabel('\omeg_A [rad/s]')
print -depsc bodeplot_original_A.eps
% Specifications
PM_des = 50;		%desired phase margin [degrees]
Dphi_PI = 15;       % allowed phase lag of the PI controller at the crossover frequency 

% Determine the new cross-over pulsation wco
phase_crossoverA = -180 + PM_des + Dphi_PI;
wcoA = interp1(phase_oriA,w,phase_crossoverA);	

% Determine Ti [s], such that the phase lag of the PI controller at wco equals Dphi_PI
Ti_A = 1/(wcoA * tan(Dphi_PI*pi/180));      % Matlab uses radians!
num_PI_A = [Ti_A 1];
den_PI_A = [Ti_A 0];

% Transforming the control TF from continuous time to discrete time
contrA_PI_c = tf(num_PI_A, den_PI_A)
contrA_PI_d = c2d(contrA_PI_c, Ts, 'zoh') 
[num_PI_A,den_PI_A] = tfdata(contrA_PI_d, 'v');


% Open loop system A with PI controller
sysA_PI = sysA * contrA_PI_d;
[num_loopgainA,den_loopgainA] = tfdata(sysA_PI, 'v');

% Calculate the gain such that the amplitude at wco equals 1
[mag, phase] = bode(sysA_PI,w);
phase_PI_A = zeros(length(w),1);
mag_PI_A = zeros(length(w),1);
phase_PI_A(:) = phase(:,:,:);
mag_PI_A(:) = mag(:,:,:);

gainA = 1/interp1(w,mag_PI_A,wcoA);
gainA = 0.775;
% Controller and open loop system A with correct gain
sysA_PI_wogain = sysA_PI;
contrA_PI_wogain = contrA_PI_d;
num_loopgainA = num_loopgainA*gainA;
num_PI_A = num_PI_A * gainA;
contrA_PI = tf(num_PI_A, den_PI_A, Ts)
sysA_PI = tf(num_loopgainA, den_loopgainA, Ts)

[mag, phase] = bode(sysA_PI,w);
phase_PI_g_A = zeros(length(w),1);
mag_PI_g_A = zeros(length(w),1);
phase_PI_g_A(:) = phase(:,:,:);
mag_PI_g_A(:) = mag(:,:,:);

% Bodeplot controller
figure
bode(contrA_PI)
title('Frequency respons PI controller in discrete time')
print -depsc bodeplot_controller.eps

% Bodeplot open loop system As
figure
subplot(211)
semilogx(w, [20*log10(mag_oriA), 20*log10(mag_PI_A),20*log10(mag_PI_g_A)])
grid on, axis tight
ylabel('|P(j\omeg_A)| [dB]')
title('Bodeplot open loop system As')
legend('Original system motor A', 'system motor A with PI controller', 'system motor A with PI with correct gain', 'location', 'NorthEast')
subplot(212)
semilogx(w, [phase_oriA, phase_PI_A, phase_PI_g_A])
grid on, axis tight
ylabel('\phi(P(j\omeg_A)) [^o]')
xlabel('\omeg_A [rad/s]')
print -depsc bodeplot_openloop.eps

% Bodeplot of only serial connection of controller and system A
figure
subplot(211)
semilogx(w, 20*log10(mag_PI_g_A))
grid on, axis tight
ylabel('|P(j\omeg_A)| [dB]')
title('Bodeplot serial connection of PI controller and system motor A')
subplot(212)
semilogx(w, phase_PI_g_A)
grid on, axis tight
ylabel('\phi(P(j\omeg_A)) [^o]')
xlabel('\omeg_A [rad/s]')
print -depsc bodeplot_contr_sysA_openloop.eps


figure
bode(sysA_PI)
% Parameters open loop system A with PI controller 
[GM_PI_g_A,PM_PI_g_A,Wpi_PI_g_A,Wc_PI_g_A] = margin(sysA_PI);


% Bode plot closed loop system A
sysA_cl = feedback(sysA_PI,1)

[mag, phase] = bode(sysA_cl,w);
phase_clA = zeros(length(w),1);
mag_clA = zeros(length(w),1);
phase_clA(:) = phase(:,:,:);
mag_clA(:) = mag(:,:,:);

figure
subplot(211)
semilogx(w, 20*log10(mag_clA))
grid on, axis tight
ylabel('|P(j\omeg_A)| [dB]')
title('Bodeplot closed loop system A with PI controller, PM = 60°')
subplot(212)
semilogx(w,phase_clA)
grid on, axis tight
ylabel('\phi(P(j\omeg_A)) [^o]')
xlabel('\omeg_A [rad/s]')
print -depsc bodeplot_cl.eps

%% Step response closed loop system A

t = [0:0.01:10];
[sim_step] = step(sysA_cl,t);
figure
plot(t, sim_step)
title('Step respons for PM = 60°')
print -depsc steprespons_cl.eps
figure
plot(t,1-sim_step)
title('ddphi = 15')

error11 = sum(abs(1-sim_step))
gainA
GM_PI_g_A
PM_PI_g_A
Ti_A
1/Ti_A
Wc_PI_g_A
bandwidth(sysA_cl)

%% Resultaten

% csvfile = '../Data/step_controller2.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% step_controller2 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save step_controller2

load step_controller3.mat
% csvfile = '../Data/step_controller3.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% step_controller3 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save step_controller3
load step_controller3.mat

% csvfile = '../Data/step_controller4.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% step_controller4 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save step_controller4
load step_controller4.mat

% csvfile = '../Data/step_controller5.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% step_controller5 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save step_controller5
load step_controller5.mat

% PM = 85° ddphi = 15°
% csvfile = '../Data/step_controller6.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% step_controller6 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save step_controller6
load step_controller6.mat

% PM = 70° ddphi = 15°
% csvfile = '../Data/step_controller7.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% step_controller7 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save step_controller7
load step_controller7.mat

% PM = 87.4° ddphi = 15°
% csvfile = '../Data/step_controller8.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% step_controller8 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save step_controller8
load step_controller8.mat

% K = 0.8
% csvfile = '../Data/step_controller9.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% step_controller9 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save step_controller9
load step_controller9.mat

% K = 0.8, PM = 45°, 5rad/s
% csvfile = '../Data/step_controller10.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% step_controller10 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save step_controller10
load step_controller10.mat

% K = 0.8, PM = 50°, 6rad/s
% csvfile = '../Data/step_controller11.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% step_controller11 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save step_controller11
load step_controller11.mat

step_controller = step_controller11;
voltageA = step_controller(:,2);
voltageB = step_controller(:,3);
va  = step_controller(:, 6); % velocity motor a
vb = step_controller(:, 7); % velocity motor b
t = step_controller(:,10);
eA = step_controller(:,12);
figure
plot(t,va)
legend('\omeg_A_a', '\omeg_A_b')
ylabel('\omeg_A [rad/s]')
xlabel('time [s]')

figure
plot(t,voltageA)
ylabel('voltageA [V]')
xlabel('time [s]')


figure
plot(t,eA)
ylabel('eA')
xlabel('time [s]')

