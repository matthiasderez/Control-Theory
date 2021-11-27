%% Explanation naming
% 2 different models are used in this script: 1 refers to the complex model, 2 refers to the simple model. 
% Different filters are used: BW stands for Butterworth, SK stands for Sanathan Koerner

%% Loading data
close all
clear
% csvfile = '../Data/recording9.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% data = dlmread(csvfile, ',', 2, 0); % Data follows the labels
% 
% save data

load data.mat

% csvfile = '../Data/recording10(4V).csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% data10 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
% 
% save data10

load data10.mat


% csvfile = '../Data/recording11(10V).csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% data11 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
% 
% save data11

load data11.mat


% csvfile = '../Data/ground_step.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% ground_step = dlmread(csvfile, ',', 2, 0); % Data follows the labels
% 
% save ground_step

load ground_step.mat

% csvfile = '../Data/ground_blokfuncties.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% ground_blokfuncties = dlmread(csvfile, ',', 2, 0); % Data follows the labels
% 
% save ground_blokfuncties

load ground_blokfuncties.mat

% csvfile = '../Data/air_step_input6.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% air_step = dlmread(csvfile, ',', 2, 0); % Data follows the labels
% 
% save air_step

load air_step.mat

%% Defining variables

% Data alternating +6V, 0V, -6V, 0V, input
voltageA = data(:,2);
voltageB = data(:,3);
positionA = data(:,4);
positionB = data(:,5);
va  = data(:, 6); % velocity motor a
vb = data(:, 7); % velocity motor b
t = data(:,10);
N = length(t);
num_periods = 4;
points_per_period = N/num_periods;
Ts = 0.01;
fs = 1/Ts;
f = [0:N-1]'*(fs/N); % arrays of frequencies, 0 to f_s Hz

% Data step input 6V
voltageA_step = air_step(:,2);
voltageB_step = air_step(:,3);
va_step  = air_step(:, 6);
vb_step = air_step(:, 7);
t_step = air_step(:,10);

% Data cart on ground (same input as before)
va_gs  = ground_step(:, 6);
vb_gs = ground_step(:, 7);
voltageA_gs = ground_step(:,2);
voltageB_gs = ground_step(:,3);
va_gb  = ground_blokfuncties(:, 6);
vb_gb = ground_blokfuncties(:, 7);
t_gs = ground_step(:,10);




%% Plotting data
% motor velocity plots
figure
subplot(121)
plot(t, va)
ylabel('\omega_a [rad/s]')
xlabel('t [s]')

subplot(122)
plot(t, vb)
ylabel('\omega_b [rad/s]')
xlabel('t [s]')
sgtitle('Motor velocity')
print -depsc motor_velocity.eps

figure
plot(t,[va, vb])
legend('Motor A','Motor B')
ylabel('Motor velocity [rad/s]')
xlabel('t [s]')
print -depsc motor_velocities_1plot.eps


figure
plot(t, va-vb)
ylabel('\omega_a - \omega_b [rad/s]')
xlabel('t [s]')
sgtitle('Difference between motor velocities')
print -depsc difference_motor_velocities.eps

% motor voltage plot
figure
plot(t, voltageA);
ylabel('Input voltage [V]')
xlabel('t [s]')
sgtitle('Input voltage')
print -depsc input_voltage.eps

%% overlap the plot for different periods to appreciate the noise (less important)

% separate the different periods (one per column)
va_matrix = reshape(va,points_per_period,num_periods); %points_per_period= #rows, num_periods = #columns
vb_matrix = reshape(vb,points_per_period,num_periods);
voltageA_matrix = reshape(voltageA,points_per_period,num_periods);

% lets compute the mean of the signals across the periods to have a point of comparison to assess the noise
va_mean = mean(va_matrix,2); %average over dimension 2 (horizontal = over columns)
vb_mean = mean(vb_matrix,2);
voltageA_mean = mean(voltageA_matrix,2);

%repmat creates a large matrix containing of 1 by num_periods copies of
%va_mean, so 4 times next to each othes, always 1 vector high
dva_matrix = va_matrix - repmat(va_mean,1,num_periods); 
dvb_matrix = vb_matrix - repmat(vb_mean,1,num_periods); 
dvoltageA_matrix = voltageA_matrix - repmat(voltageA_mean,1,num_periods);

% plotting some interesting comparisons
figure,hold on
subplot(2,1,1),plot(t(1:points_per_period), va_matrix, 'LineWidth', 1) 
grid on
axis tight
xlabel('t  [s]')
ylabel('\omega_a  [rad/s]')
subplot(2,1,2),plot(t(1:points_per_period), dva_matrix, 'LineWidth', 1)
grid on
axis tight
xlabel('t  [s]')
ylabel('\Delta \omega_a  [rad/s]')
hold off
print -depsc omegaA_deltaomegaA.eps

figure,hold on
subplot(2,1,1),plot(t(1:points_per_period), vb_matrix, 'LineWidth', 1) 
grid on
axis tight
xlabel('t  [s]')
ylabel('\omega_b  [rad/s]')
subplot(2,1,2),plot(t(1:points_per_period), dvb_matrix, 'LineWidth', 1)
grid on
axis tight
xlabel('t  [s]')
ylabel('\Delta \omega_b  [rad/s]')
hold off
print -depsc omegaB_deltaomegaB.eps


figure,hold on
subplot(2,1,1),plot(t(1:points_per_period), voltageA_matrix, 'LineWidth', 1) 
grid on
axis tight
xlabel('t  [s]')
ylabel('voltageA  [V]')
subplot(2,1,2),plot(t(1:points_per_period), dvoltageA_matrix, 'LineWidth', 1)
grid on
axis tight
xlabel('t  [s]')
ylabel('\Delta voltageA  [V]')
hold off
print -depsc V_deltaV.eps

%% 2.b) Least square method on data motor A, no filtering: COMPLEX MODEL

% H(z) = (b1*z+b2)/(z(z^2-a1z-a2))
% 
% collect the signals appearing in the difference equation
b1 = va(4:end); 
phi1 = [-va(3:end-1), -va(2:end-2), voltageA(2:end-2), voltageA(1:end-3)]; 
% perform the fit to get the desired parameters
theta1 = phi1\b1;

% build the identified model
Num1 = [0, theta1(3), theta1(4)];
Den1 = [1, theta1(1), theta1(2), 0];
sys_1 = tf(Num1, Den1, Ts);

% compute the frequency response of the identified model
FRF1 = squeeze(freqresp(sys_1,2*pi*f));
mag_1 = 20*log10(abs(FRF1));
phs_1 = 180/pi*unwrap(angle(FRF1)); 
phs_1 = 360*ceil(-phs_1(1)/360) + phs_1;

% plot the results
figure,hold on,
subplot(2,1,1),semilogx(f, mag_1)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [rad/s]')
legend('estimated','Location','SouthWest')
subplot(2,1,2),semilogx(f, phs_1)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ]')
legend('estimated','Location','SouthWest')
sgtitle('FRF complex model')
print -depsc FRF_complex_model.eps

%% 2.b) Least square method on data motor A, no filtering: SIMPLE MODEL

% suppose L = 0
% H(z) = b1/(z(z-a1))
% collect the signals appearing in the difference equation
b2 = va(3:end); 
phi2 = [va(2:end-1), voltageA(1:end-2)]; 

% perform the fit to get the desired parameters
theta2 = phi2\b2;

% build the identified model
Num2 = [theta2(2)];
Den2 = [1, -theta2(1), 0];
sys_2 = tf(Num2, Den2, Ts);

% compute the frequency response of the identified model
FRF2 = squeeze(freqresp(sys_2,2*pi*f));
mag_2 = 20*log10(abs(FRF2));
phs_2 = 180/pi*unwrap(angle(FRF2)); 
phs_2 = 360*ceil(-phs_2(1)/360) + phs_2;

% plot the results
figure,hold on,
subplot(2,1,1),semilogx(f, mag_2)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [rad/s]')
legend('estimated','Location','SouthWest')
subplot(2,1,2),semilogx(f, phs_2)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ] simple model')
legend('estimated','Location','SouthWest')
sgtitle('FRF simple model')
print -depsc FRF_simple_model.eps

% THIS IS THE MODEL WE WILL USE

%% 2.c) Filtering with Butterworth filter: COMPLEX MODEL 
% orde hoger dan orde systeem
% cutoff freq = bandwith van ongefilterde LSE
% te lage orde: te zwakke attenuation van hoge freq
% te hoge orde: te grote delay
% => kies 4e orde

% ******* Motor A *******

% define a low(band)-pass filter
cutoff = bandwidth(sys_1)/(2*pi);
[B_BW1,A_BW1] = butter(6, cutoff*(2/fs));
h = fvtool(B_BW1, A_BW1);

% apply the filter to both input and output
va_BW1 = filter(B_BW1, A_BW1, va); 
voltageA_BW1 = filter(B_BW1, A_BW1, voltageA);

%repeat the identification
b_BW1 = va_BW1(4:end); 
phi_BW1 = [-va_BW1(3:end-1),-va_BW1(2:end-2), voltageA_BW1(2:end-2), voltageA_BW1(1:end-3)]; 
theta_BW1 = phi_BW1\b_BW1;
Num_BW1 = [0,theta_BW1(3),theta_BW1(4)];
Den_BW1 = [1, theta_BW1(1),theta_BW1(2), 0];
sys_BW1 = tf(Num_BW1, Den_BW1, Ts);

% compute the frequency response of the new identified model
FRF_BW1 = squeeze(freqresp(sys_BW1,2*pi*f));
mag_BW1 = 20*log10(abs(FRF_BW1));
phs_BW1 = 180/pi*unwrap(angle(FRF_BW1)); 
phs_BW1 = 360*ceil(-phs_BW1(1)/360) + phs_BW1;

% plot results
figure, hold on
sgtitle("LLS with of complex model low-pass filter applied to the input and output data")
subplot(2,1,1),semilogx(f, mag_BW1)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [m]')
legend('estimated','Location','SouthWest')
axis tight
subplot(2,1,2)
semilogx(f, phs_BW1)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ]')
legend('estimated','Location','SouthWest')
print -depsc FRF_complex_filter.epsv

%% 2.c) Filtering with Butterworth filter: SIMPLE MODEL

% Butterworth
% orde hoger dan orde systeem
% cutoff freq = bandwith van ongefilterde LSE
% te lage orde: te zwakke attenuation van hoge freq
% te hoge orde: te grote delay
% => kies 4e orde

% ******* Motor A *******

% define a low(band)-pass filter
cutoff = bandwidth(sys_2)/(2*pi);
[B_BW2,A_BW2] = butter(6, cutoff*(2/fs));
h = fvtool(B_BW2, A_BW2);

% apply the filter to both input and output
va_BW2 = filter(B_BW2, A_BW2, va); 
voltageA_BW2 = filter(B_BW2, A_BW2, voltageA);


%repeat the identification
b_BW2 = va_BW2(3:end); 
phi_BW2 = [va_BW2(2:end-1), voltageA_BW2(1:end-2)]; 
theta1_BW2 = phi_BW2\b_BW2;
Num_BW2 = [theta1_BW2(2)];
Den_BW2 = [1, -theta1_BW2(1), 0];
sys_BW2 = tf(Num_BW2, Den_BW2, Ts);

% compute the frequency response of the new identified model
FRF_BW2 = squeeze(freqresp(sys_BW2,2*pi*f));
mag_BW2 = 20*log10(abs(FRF_BW2));
phs_BW2 = 180/pi*unwrap(angle(FRF_BW2)); 
phs_BW2 = 360*ceil(-phs_BW2(1)/360) + phs_BW2;

% plot results
figure, hold on
sgtitle("LLS of simple model with Butterworth filter applied to the input and output data")
subplot(2,1,1),semilogx(f, mag_BW2)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [m]')
legend('estimated','Location','SouthWest')
axis tight
subplot(2,1,2)
semilogx(f, phs_BW2)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ]')
legend('estimated','Location','SouthWest')
print -depsc FRF_simple_BW.eps

%% 2.c) Filtering Sanathan Koerner procedure: COMPLEX MODEL

% Denumerator of complex model without filter: Den1
sys_SK1 = sys_1;
error_SK1 = [100 100 100 100];
Den_SK1_2 = Den1;
Den_SK1_1 = 0;
va_SK1 = va;
voltageA_SK1 = voltageA;
 
while abs(error_SK1) > [eps eps eps eps]
    Den_SK1_1 = Den_SK1_2;
    va_SK1 = filter(1, Den_SK1_2, va_SK1);
    voltageA_SK1 = filter(1, Den_SK1_2, voltageA_SK1);
    b_SK1 = va_SK1(4:end); 
    phi_SK1 = [-va_SK1(3:end-1),-va_SK1(2:end-2), voltageA_SK1(2:end-2), voltageA_SK1(1:end-3)]; 
    theta_SK1 = phi_SK1\b_SK1;
    Num_SK1 = [0, theta_SK1(3), theta_SK1(4) ];
    Den_SK1_2 = [1, theta_SK1(1),  theta_SK1(2), 0];
    sys_SK1= tf(Num_SK1, Den_SK1_2, Ts);
    error_SK1 = Den_SK1_2-Den_SK1_1;
end


    
FRF_SK1 = squeeze(freqresp(sys_SK1,2*pi*f));
mag_SK1 = 20*log10(abs(FRF_SK1));
phs_SK1 = 180/pi*unwrap(angle(FRF_SK1)); 
phs_SK1 = 360*ceil(-phs_SK1(1)/360) + phs_SK1;

% plot results
figure, hold on
sgtitle("LLS of complex model with SK filter applied to the input and output data")
subplot(2,1,1),semilogx(f, mag_SK1)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [m]')
legend('estimated','Location','SouthWest')
axis tight
subplot(2,1,2)
semilogx(f, phs_SK1)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ]')
legend('estimated','Location','SouthWest')
print -depsc FRF_SK_filter_complex.eps

%% 2.c) Filtering Sanathan Koerner procedure: SIMPLE MODEL

% Denumerator of simple model without filter: Den2
sys_SK2 = sys_2;
error2 = [100 100 100];
Den_SK2_2 = Den2;
Den_SK2_1 = 0;
va_SK = va;
voltageA_SK = voltageA;
 
while abs(error2) > [eps eps eps]
    Den_SK2_1 = Den_SK2_2;
    va_SK = filter(1, Den_SK2_2, va_SK);
    voltageA_SK = filter(1, Den_SK2_2, voltageA_SK);
    b_SK2 = va_SK(3:end); 
    phi_SK2 = [va_SK(2:end-1), voltageA_SK(1:end-2)]; 
    theta_SK2 = phi_SK2\b_SK2;
    Num_SK2 = [0 theta_SK2(2)];
    Den_SK2_2 = [1, -theta_SK2(1), 0];
    sys_SK2= tf(Num_SK2, Den_SK2_2, Ts);
    error2 = Den_SK2_2-Den_SK2_1;
end


    
FRF_SK2 = squeeze(freqresp(sys_SK2,2*pi*f));
mag_SK2 = 20*log10(abs(FRF_SK2));
phs_SK2 = 180/pi*unwrap(angle(FRF_SK2)); 
phs_SK2 = 360*ceil(-phs_SK2(1)/360) + phs_SK2;

% plot results
figure, hold on
sgtitle("LLS fo simple model with SK filter applied to the input and output data")
subplot(2,1,1),semilogx(f, mag_SK2)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [m]')
legend('estimated','Location','SouthWest')
axis tight
subplot(2,1,2)
semilogx(f, phs_SK2)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ]')
legend('estimated','Location','SouthWest')
print -depsc FRF_SK_filter_simpel.eps

%% 2.d) Difference between response of the simulated model and the emperical values: NO FILTER, COMPLEX MODEL 

% Motor A
va_est1 = lsim(sys_1,voltageA_step(106:505),t_step(106:505));               % start at 106, moment of step
figure
subplot(2,1,1),plot(t(1:400),[va_step(106:505) va_est1]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(106:505) - va_est1))
legend('error')
xlabel('time [s]')
ylabel('omega_{a empirical} - \omega_{a estimated} [rad/s]')
axis tight
sgtitle('Step response complex model motor a')
print -depsc step_response_complex_a.eps

% Motor B
vb_est1 = lsim(sys_1,voltageA_step(106:505),t_step(106:505));
figure
subplot(2,1,1),plot(t(1:400),[vb_step(106:505) vb_est1]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(vb_step(106:505) - vb_est1))
legend('error')
xlabel('time [s]')
ylabel('\omega_{b empirical} - \omega_{b estimated} [rad/s]')
axis tight
sgtitle('Step response complex model motor b')
print -depsc step_response_complex_b.eps

% Motor A
va_est_square = lsim(sys_1,voltageA,t);
figure
subplot(2,1,1),plot(t,[va va_est_square]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t,abs(va - va_est_square))
legend('error')
xlabel('time [s]')
ylabel('omega_{a empirical} - \omega_{a estimated} [rad/s]')
axis tight
sgtitle('Response complex model to square wavefunction input motor a')
print -depsc square_square_wave_response_complex_a.eps

% Motor B
vb_est_square = lsim(sys_1,voltageA,t);
figure
subplot(2,1,1),plot(t,[vb vb_est_square]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t,abs(vb - vb_est_square))
legend('error')
xlabel('time [s]')
ylabel('omega_{b empirical} - \omega_{b estimated} [rad/s]')
axis tight
sgtitle('Response complex model to square wavefunction input motor b')
print -depsc square_square_wave_response_complex_b.eps

figure, hold on
pzmap(sys_1)
print -depsc p&z_complex.eps


%% 2.d) Difference between response of the simulated model and the real system: SIMPLE MODEL, NO FILTER

% Motor A
figure
va_estA = lsim(sys_2,voltageA_step(106:505),t_step(106:505));
subplot(2,1,1),plot(t(1:400),[va_step(106:505) va_estA]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(106:505) - va_estA))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('Step response simple model motor a')
print -depsc step_response_simple_a.eps

% Motor B
figure
va_estB = lsim(sys_2,voltageA_step(106:505),t_step(106:505));
subplot(2,1,1),plot(t(1:400),[vb_step(106:505) va_estB]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(vb_step(106:505) - va_estB))
legend('error')
xlabel('time [s]')
ylabel('\omega_b(empirical) - \omega_b(estimated) [rad/s]')
axis tight
sgtitle('Step response simple model motor b')
print -depsc step_response_simple_b.eps

% Motor A
va_est1 = lsim(sys_2,voltageA,t);
figure 
subplot(2,1,1),plot(t,[va va_est1]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t,abs(va - va_est1))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('response simple model to square wavefunction input motor a')
print -depsc square_square_wave_response_simple_a.eps

% Motor B
va_est1 = lsim(sys_2,voltageA,t);
figure 
subplot(2,1,1),plot(t,[vb va_est1]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t,abs(vb - va_est1))
legend('error')
xlabel('time [s]')
ylabel('\omega_b(empirical) - \omega_b(estimated) [rad/s]')
axis tight
sgtitle('response simple model to square wavefunction input motor b')
print -depsc square_square_wave_response_simple_b.eps


figure, hold on
pzmap(sys_2)
print -depsc p&z_simple.eps



%% 2.d) Difference between response of the simulated model and the real system: COMPLEX MODEL, BUTTERWORTH FILTER


%empirical
va_est_BW1 = lsim(sys_BW1, voltageA_step(106:505), t_step(106:505));
figure
subplot(2,1,1),plot(t(1:400),[va_step(106:505) va_est_BW1]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(106:505) - va_est_BW1))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('Step response of complex model with Butterworth filter')
print -depsc step_response_complex_BW_a.eps

figure,hold on
pzmap(sys_BW1)
print -depsc p&z_complex_BW.eps



%% 2.d) Difference between response of the simulated model and the real system: SIMPLE MODEL, BUTTERWORTH FILTER
%empirical
va_est_BW2 = lsim(sys_BW2, voltageA_step(106:505), t_step(106:505));
figure
subplot(2,1,1),plot(t(1:400),[va_step(106:505) va_est_BW2]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(106:505) - va_est_BW2))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('Step response of simple model with Butterworth filter')
print -depsc step_response_simple_BW_a.eps

figure,hold on
pzmap(sys_BW2)
print -depsc p&z_simple_BW.eps

%% 2.d) Difference between response of the simulated model and the real system: COMPLEX MODEL, SK FILTER

%empirical
va_est_SK1 = lsim(sys_SK1, voltageA_step(106:505), t_step(106:505));
figure
subplot(2,1,1),plot(t(1:400),[va_step(106:505) va_est_SK1]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(106:505) - va_est_SK1))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('Step response of complex model with SK filter')
print -depsc step_response_SK_filter__complex_a.eps

figure,hold on
pzmap(sys_SK1)
print -depsc p&z_complex_SK.eps

error_SK1_tot= sum(abs(abs(va_step(106:505) - va_est_SK1)-abs(va_step(106:505) - va_estA)));




%% 2.d) Difference between response of the simulated model and the real system: SIMPLE MODEL, SK FILTER
%empirical
va_est_SK2 = lsim(sys_SK2, voltageA_step(106:505), t_step(106:505));
figure
subplot(2,1,1),plot(t(1:400),[va_step(106:505) va_est_SK2]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(106:505) - va_est_SK2))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('Step response of simple model with SK filter')
print -depsc step_response_SK_filter__simple_a.eps

figure,hold on
pzmap(sys_SK2)
print -depsc p&z_simple_SK.eps

error_SK2_tot= sum(abs(abs(va_step(106:505) - va_est_SK2)-abs(va_step(106:505) - va_estA)));




%% 2.d) Superposition principle

va10  = data10(:, 6);
vb10 = data10(:, 7);
va11  = data11(:, 6);
vb11 = data11(:, 7);

figure
subplot(121)
plot(t, [va + va10, va11])
ylabel('\omega_a [rad/s]')
xlabel('t [s]')
legend('va (6V) + va (4V)','va (10V)')

subplot(122)
plot(t, [vb + vb10, vb11])
ylabel('\omega_b [rad/s]')
xlabel('t [s]')
legend('vb (6V) + vb (4V)','vb (10V)')
sgtitle('Prove system is non-linear')
print -depsc superposition.eps


%% 3.a) Step input to cart on ground

% Remark: only worked with simple model, no 2 used in naming anymore

% Data cart on ground

% _gs --> Ground Stepinput
% _gb --> Ground Blokfunctie input (square wave)
va_gs  = ground_step(:, 6);
vb_gs = ground_step(:, 7);
voltageA_gs = ground_step(:,2);
voltageB_gs = ground_step(:,3);
va_gb  = ground_blokfuncties(:, 6);
vb_gb = ground_blokfuncties(:, 7);
t_gs = ground_step(:,10);

figure
subplot(121)
plot(t_gs, va_gs)
ylabel('\omega_a [rad/s]')
xlabel('t [s]')

subplot(122)
plot(t_gs, vb_gs)
ylabel('\omega_b [rad/s]')
xlabel('t [s]')
sgtitle('Motor velocity for step input with cart on the ground')
print -depsc motor_velocity_groundstep.eps

figure
subplot(121)
plot(t, va_gb)
ylabel('\omega_a [rad/s]')
xlabel('t [s]')

subplot(122)
plot(t, vb_gb)
ylabel('\omega_b [rad/s]')
xlabel('t [s]')

sgtitle('Motor velocity for square square wave input with cart on the ground')
print -depsc motor_velocity_ground_square_wave.eps


% Motor A
figure
va_est1 = lsim(sys_2,voltageA_gs(149:548),t_gs(149:548));
subplot(2,1,1),plot(t(1:400),[va_gs(149:548) va_est1]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_gs(149:548) - va_est1))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('Step response motor a old model cart on ground')
print -depsc step_response_ground_simple_a.eps

% Motor B
figure
va_est1 = lsim(sys_2,voltageA_gs(149:548),t_gs(149:548));
subplot(2,1,1),plot(t(1:400),[vb_gs(149:548) va_est1]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(vb_gs(149:548) - va_est1))
legend('error')
xlabel('time [s]')
ylabel('\omega_b(empirical) - \omega_b(estimated) [rad/s]')
axis tight
sgtitle('Step response motor b old model cart on ground')
print -depsc step_response_ground_simple_b.eps

% Emperical reacts slower -> logic, inertia cart


%%% Identify new model%%% 


% suppose L = 0
% H(z) = b1/(z(z-a1))
% collect the signals appearing in the difference equation
b_gs = va_gs(3:end); 
phi_gs = [va_gs(2:end-1), voltageA_gs(1:end-2)]; 
% perform the fit to get the desired parameters

theta_gs = phi_gs\b_gs;

% build the identified model
Num_gs = [theta_gs(2)];
Den_gs = [1, -theta_gs(1), 0];
sys_gs = tf(Num_gs, Den_gs, Ts);

% compute the frequency response of the identified model
FRF_gs = squeeze(freqresp(sys_gs,2*pi*f));
mag_gs = 20*log10(abs(FRF_gs));
phs_gs = 180/pi*unwrap(angle(FRF_gs)); 
phs_gs = 360*ceil(-phs_gs(1)/360) + phs_gs;

% plot the results
figure,hold on,
subplot(2,1,1),semilogx(f, mag_gs)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [rad/s]')
legend('estimated','Location','SouthWest')
subplot(2,1,2),semilogx(f, phs_gs)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ] ')
legend('estimated','Location','SouthWest')
sgtitle('FRF new estimated model for cart on ground without filter')
print -depsc FRF_ground_newmodel_nofilter.eps

% Motor A
figure
va_est_gs = lsim(sys_gs,voltageA_gs(149:548),t_gs(149:548));
subplot(2,1,1),plot(t(1:400),[va_gs(149:548) va_est_gs]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_gs(149:548) - va_est_gs))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('Step response new model motor a for cart on ground without filter')
print -depsc step_response_ground__newmodel_nofilter_a.eps

error_ground_NF = sum(abs(va_gs(149:548) - va_est_gs));


%%% Filtering the new model with SK filter %%%


% Denumerator of simple model cart on ground without filter: Den_gs
sys_gs_SK = sys_gs;
error_gs_SK = [100 100 100];
Den_gs_SK_2 = Den_gs;
Den_gs_SK_1 = 0;
va_gs_SK = va_gs;
voltageA_gs_SK = voltageA_gs;
 
while abs(error_gs_SK) > [eps eps eps]
    Den_gs_SK_1 = Den_gs_SK_2;
    va_gs_SK = filter(1, Den_gs_SK_2, va_gs_SK);
    voltageA_gs_SK = filter(1, Den_gs_SK_2, voltageA_gs_SK);
    b_gs_SK = va_gs_SK(3:end); 
    phi_gs_SK = [va_gs_SK(2:end-1), voltageA_gs_SK(1:end-2)]; 
    theta_gs_SK = phi_gs_SK\b_gs_SK;
    Num_gs_SK = [0 theta_gs_SK(2)];
    Den_gs_SK_2 = [1, -theta_gs_SK(1), 0];
    sys_gs_SK= tf(Num_gs_SK, Den_gs_SK_2, Ts);
    error_gs_SK = Den_gs_SK_2-Den_gs_SK_1;
end


    
FRF_gs_SK = squeeze(freqresp(sys_gs_SK,2*pi*f));
mag_gs_SK = 20*log10(abs(FRF_gs_SK));
phs_gs_SK = 180/pi*unwrap(angle(FRF_gs_SK)); 
phs_gs_SK = 360*ceil(-phs_gs_SK(1)/360) + phs_gs_SK;

% plot results
figure, hold on
sgtitle("LLS of simple model with SK filter applied to the input and output data")
subplot(2,1,1),semilogx(f, mag_gs_SK)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [m]')
legend('estimated','Location','SouthWest')
axis tight
subplot(2,1,2)
semilogx(f, phs_gs_SK)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ]')
legend('estimated','Location','SouthWest')
print -depsc FRF_SK_filter_ground.eps

% Motor A
figure
va_est_gs_SK = lsim(sys_gs_SK,voltageA_gs(149:548),t_gs(149:548));
subplot(2,1,1),plot(t(1:400),[va_gs(149:548) va_est_gs_SK]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_gs(149:548) - va_est_gs_SK))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('Step response new estimated model motor a for cart on ground with SK filter')
print -depsc step_response_ground__newmodel_SKfilter_a.eps

error_ground_SK = sum(abs(va_gs(149:548) - va_est_gs_SK));

%%% Filtering the new model with BW filter %%%

%cutoff = bandwidth(sys_gs_BW)/(2*pi);
cutoff = 35;
[B_gs_BW,A_gs_BW] = butter(6, cutoff*(2/fs));
h = fvtool(B_gs_BW, A_gs_BW);

% apply the filter to both input and output
va_gs_BW = filter(B_gs_BW, A_gs_BW, va_gs); 
voltageA_gs_BW = filter(B_gs_BW, A_gs_BW, voltageA_gs);

%repeat the identification
b_gs_BW = va_gs_BW(3:end); 
phi_gs_BW = [va_gs_BW(2:end-1), voltageA_gs_BW(1:end-2)]; 
theta1_gs_BW = phi_gs_BW\b_gs_BW;
Num_gs_BW = [theta1_gs_BW(2)];
Den_gs_BW = [1, -theta1_gs_BW(1), 0];
sys_gs_BW = tf(Num_gs_BW, Den_gs_BW, Ts);

% compute the frequency response of the identified model
FRF_gs_BW = squeeze(freqresp(sys_gs_BW,2*pi*f));
mag_gs_BW = 20*log10(abs(FRF_gs_BW));
phs_gs_BW = 180/pi*unwrap(angle(FRF_gs_BW)); 
phs_gs_BW = 360*ceil(-phs_gs_BW(1)/360) + phs_gs_BW;

% plot the results
figure,hold on,
subplot(2,1,1),semilogx(f, mag_gs_BW)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [rad/s]')
legend('estimated','Location','SouthWest')
subplot(2,1,2),semilogx(f, phs_gs_BW)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ]')
legend('estimated','Location','SouthWest')
sgtitle('FRF new estimated model for cart on ground with Butterworth filter')
print -depsc FRF_ground_newmodel_filter.eps

% Motor A
figure
va_est_gs_BW = lsim(sys_gs_BW,voltageA_gs_BW(149:548),t_gs(149:548));
subplot(2,1,1),plot(t(1:400),[va_gs_BW(149:548) va_est_gs_BW]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_gs_BW(149:548) - va_est_gs_BW))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('Step response new estimated model motor a for cart on ground with Butterworth filter')
print -depsc step_response_ground__newmodel_filter_a.eps

figure, hold on
pzmap(sys_gs_BW)

error_ground_BW = sum(abs(va_gs(149:548) - va_est_gs_BW));