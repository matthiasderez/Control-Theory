close all
clear all
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
% Data alternating input
voltageA = data(:,2);
voltageB = data(:,3);
positionA = data(:,4);
positionB = data(:,5);
va  = data(:, 6);
vb = data(:, 7);
t = data(:,10);
N = length(t);
num_periods = 4;
points_per_period = N/num_periods;
Ts = 0.01;
fs = 1/Ts;
f = [0:N-1]'*(fs/N); % arrays of frequencies, 0 to f_s Hz

% Data step input
voltageA_step = air_step(:,2);
voltageB_step = air_step(:,3);
va_step  = air_step(:, 6);
vb_step = air_step(:, 7);
t_step = air_step(:,10);

% Data cart on ground
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

%% overlap the plot for different periods to appreciate the noise
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

%% 2.b) Least square method, no filtering (on motor A) COMPLEX MODEL

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
sys_d1 = tf(Num1, Den1, Ts);

% compute the frequency response of the identified model
FRF1 = squeeze(freqresp(sys_d1,2*pi*f));
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
sgtitle('FRF estimated model (complex)')
print -depsc FRF_complex_model.eps






%% 2.d) Difference between response of the simulated model and the real system COMPLEX MODEL, NO FILTER


% Motor A
va_est = lsim(sys_d1,voltageA_step(106:505),t_step(106:505));               % start at 134, moment of step
figure
subplot(2,1,1),plot(t(1:400),[va_step(106:505) va_est]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(106:505) - va_est))
legend('error')
xlabel('time [s]')
ylabel('omega_{a empirical} - \omega_{a estimated} [rad/s]')
axis tight
sgtitle('Step response complex model')
print -depsc step_response_complex_a.eps

% Motor B
vb_est = lsim(sys_d1,voltageA_step(106:505),t_step(106:505));
figure
subplot(2,1,1),plot(t(1:400),[vb_step(106:505) vb_est]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(vb_step(106:505) - vb_est))
legend('error')
xlabel('time [s]')
ylabel('\omega_{b empirical} - \omega_{b estimated} [rad/s]')
axis tight
sgtitle('Step response complex model')
print -depsc step_response_complex_b.eps

% Motor A
x1 = lsim(sys_d1,voltageA,t);
figure
subplot(2,1,1),plot(t,[va x1]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t,abs(va - x1))
legend('error')
xlabel('time [s]')
ylabel('omega_{a empirical} - \omega_{a estimated} [rad/s]')
axis tight
sgtitle('response complex model to wavefunction input')
print -depsc wave_response_complex_a.eps

% Motor B
x1 = lsim(sys_d1,voltageA,t);
figure
subplot(2,1,1),plot(t,[vb x1]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t,abs(vb - x1))
legend('error')
xlabel('time [s]')
ylabel('omega_{b empirical} - \omega_{b estimated} [rad/s]')
axis tight
sgtitle('response complex model to wavefunction input')
print -depsc wave_response_complex_b.eps

figure, hold on
pzmap(sys_d1)
print -depsc p&z_complex.eps

%% 2.b) Least square method, no filtering (on motor A) SIMPLE MODEL

% suppose J = 0
% H(z) = b1/(z(z-a1))
% collect the signals appearing in the difference equation
b2 = va(3:end); 
phi2 = [va(2:end-1), voltageA(1:end-2)]; 
% perform the fit to get the desired parameters

theta2 = phi2\b2;

% build the identified model
Num2 = [theta2(2)];
Den2 = [1, -theta2(1), 0];
sys_d2 = tf(Num2, Den2, Ts);

% compute the frequency response of the identified model
FRF2 = squeeze(freqresp(sys_d2,2*pi*f));
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
sgtitle('FRF estimated model (simple)')
print -depsc FRF_simple_model.eps

% THIS IS THE MODEL WE USE FROM NOW ON

%% 2.d) Difference between response of the simulated model and the real system: SIMPLE MODEL, NO FILTER

% Motor A
figure
va_estA = lsim(sys_d2,voltageA_step(106:505),t_step(106:505));
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
va_est = lsim(sys_d2,voltageA_step(106:505),t_step(106:505));
subplot(2,1,1),plot(t(1:400),[vb_step(106:505) va_est]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(vb_step(106:505) - va_est))
legend('error')
xlabel('time [s]')
ylabel('\omega_b(empirical) - \omega_b(estimated) [rad/s]')
axis tight
sgtitle('Step response simple model motor b')
print -depsc step_response_simple_b.eps

% Motor A
va_est = lsim(sys_d2,voltageA,t);
figure 
subplot(2,1,1),plot(t,[va va_est]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t,abs(va - va_est))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('response simple model to wavefunction input')
print -depsc wave_response_simple_a.eps

% Motor B
va_est = lsim(sys_d2,voltageA,t);
figure 
subplot(2,1,1),plot(t,[vb va_est]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t,abs(vb - va_est))
legend('error')
xlabel('time [s]')
ylabel('\omega_b(empirical) - \omega_b(estimated) [rad/s]')
axis tight
sgtitle('response simple model to wavefunction input')
print -depsc wave_response_simple_b.eps


figure, hold on
pzmap(sys_d2)

%% %% 2.c) Filtering (complex model) % Butterworth
% orde hoger dan orde systeem
% cutoff freq = bandwith van ongefilterde LSE
% te lage orde: te zwakke attenuation van hoge freq
% te hoge orde: te grote delay
% => kies 4e orde

% ******* Motor A *******

% define a low(band)-pass filter
cutoff = bandwidth(sys_d1)/(2*pi);
[B_filtc,A_filtc] = butter(6, cutoff*(2/fs));
h = fvtool(B_filtc, A_filtc);

% apply the filter to both input and output
va_filtc = filter(B_filtc, A_filtc, va); 
voltageA_filtc = filter(B_filtc, A_filtc, voltageA);

%repeat the identification
b1_filt = va_filtc(4:end); 
phi1_filt = [-va_filtc(3:end-1),-va_filtc(2:end-2), voltageA_filtc(2:end-2), voltageA_filtc(1:end-3)]; 
theta1_filtc = phi1_filt\b1_filt;
Num1_filt = [0,theta1_filtc(3),theta1_filtc(4)];
Den1_filt = [1, theta1_filtc(1),theta1_filtc(2), 0];
sys_d1_filt = tf(Num1_filt, Den1_filt, Ts);

% compute the frequency response of the new identified model
FRF1_filt = squeeze(freqresp(sys_d1_filt,2*pi*f));
mag_1_filt = 20*log10(abs(FRF1_filt));
phs_1_filt = 180/pi*unwrap(angle(FRF1_filt)); 
phs_1_filt = 360*ceil(-phs_1_filt(1)/360) + phs_1_filt;

% plot results
figure, hold on
sgtitle("LLS with of complex model low-pass filter applied to the input and output data")
subplot(2,1,1),semilogx(f, mag_1_filt)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [m]')
legend('estimated','Location','SouthWest')
axis tight
subplot(2,1,2)
semilogx(f, phs_1_filt)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ]')
legend('estimated','Location','SouthWest')
print -depsc FRF_complex_filter.eps

%% 2.d) Difference between response of the simulated model and the real system: COMPLEX MODEL, WITH FILTER


%empirical
va_est_filt = lsim(sys_d1_filt, voltageA_step(106:505), t_step(106:505));
figure
subplot(2,1,1),plot(t(1:400),[va_step(106:505) va_est_filt]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(106:505) - va_est_filt))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('Step response of complex model with low-pass filter')
print -depsc step_response_complex_filter_a.eps

figure,hold on
pzmap(sys_d1_filt)

%% 2.c) Filtering BUTTERWORTH (simple model)

% Butterworth
% orde hoger dan orde systeem
% cutoff freq = bandwith van ongefilterde LSE
% te lage orde: te zwakke attenuation van hoge freq
% te hoge orde: te grote delay
% => kies 4e orde

% ******* Motor A *******

% define a low(band)-pass filter
cutoff = bandwidth(sys_d2)/(2*pi);
[B_filts,A_filts] = butter(6, cutoff*(2/fs));
h = fvtool(B_filts, A_filts);

% apply the filter to both input and output
va_filts = filter(B_filts, A_filts, va); 
voltageA_filts = filter(B_filts, A_filts, voltageA);

%repeat the identification
b1_filt = va_filts(3:end); 
phi1_filt = [va_filts(2:end-1), voltageA_filts(1:end-2)]; 
theta1_filts = phi1_filt\b1_filt;
Num1_filt = [theta1_filts(2)];
Den1_filt = [1, -theta1_filts(1), 0];
sys_d2_filt = tf(Num1_filt, Den1_filt, Ts);

% compute the frequency response of the new identified model
FRF2_filt = squeeze(freqresp(sys_d2_filt,2*pi*f));
mag_2_filt = 20*log10(abs(FRF2_filt));
phs_2_filt = 180/pi*unwrap(angle(FRF2_filt)); 
phs_2_filt = 360*ceil(-phs_2_filt(1)/360) + phs_2_filt;

% plot results
figure, hold on
sgtitle("LLS fo simple model with low-pass filter applied to the input and output data")
subplot(2,1,1),semilogx(f, mag_2_filt)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [m]')
legend('estimated','Location','SouthWest')
axis tight
subplot(2,1,2)
semilogx(f, phs_2_filt)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ]')
legend('estimated','Location','SouthWest')
print -depsc FRF_simple_filter.eps


%% 2.d) Difference between response of the simulated model and the real system: SIMPLE MODEL, WITH FILTER
%empirical
va_est_filt2 = lsim(sys_d2_filt, voltageA_step(106:505), t_step(106:505));
figure
subplot(2,1,1),plot(t(1:400),[va_step(106:505) va_est_filt2]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(106:505) - va_est_filt2))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('Step response of simple model with low-pass filter')
print -depsc step_response_simple_filter_a.eps

figure,hold on
pzmap(sys_d2_filt)
%% 2.c) Filtering Sanathan Koerner procedure (simple model)

% Denumerator of simple model without filter: Den2
sys2_SK = sys_d2;
error = [0 0 0];
Den2_SK2 = Den2;
Den2_SK1 = 0;
va_SK = va;
voltageA_SK = voltageA;
 
while abs(error) > [eps eps eps]
    Den2_SK1 = Den2_SK2;
    va_SK = filter(1, Den2_SK2, va_SK);
    voltageA_SK = filter(1, Den2_SK2, voltageA_SK);
    b2_SK = va_SK(3:end); 
    phi2_SK = [va_SK(2:end-1), voltageA_SK(1:end-2)]; 
    theta2_SK = phi2_SK\b2_SK;
    Num2_SK = [0 theta2_SK(2)];
    Den2_SK2 = [1, -theta2_SK(1), 0];
    sys2_SK= tf(Num2_SK, Den2_SK2, Ts);
    error = Den2_SK2-Den2_SK1;
end


    
FRF2_SK = squeeze(freqresp(sys2_SK,2*pi*f));
mag_2_SK = 20*log10(abs(FRF2_SK));
phs_2_SK = 180/pi*unwrap(angle(FRF2_SK)); 
phs_2_SK = 360*ceil(-phs_2_SK(1)/360) + phs_2_SK;

% plot results
figure, hold on
sgtitle("LLS fo simple model with SK filter applied to the input and output data")
subplot(2,1,1),semilogx(f, mag_2_SK)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [m]')
legend('estimated','Location','SouthWest')
axis tight
subplot(2,1,2)
semilogx(f, phs_2_SK)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ]')
legend('estimated','Location','SouthWest')
print -depsc FRF_SK_filter_simpel.eps

%% 2.d) Difference between response of the simulated model and the real system: SIMPLE MODEL, WITH SK FILTER
%empirical
va_est_SK2 = lsim(sys2_SK, voltageA_step(106:505), t_step(106:505));
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
pzmap(sys2_SK)

errorSK2= abs(va_step(106:505) - va_est_SK2)-abs(va_step(106:505) - va_estA);


%% 2.c) Filtering Sanathan Koerner procedure (complex model)

% Denumerator of simple model without filter: Den2
sys1_SK = sys_d1;
error = [0 0 0];
Den1_SK2 = Den1;
Den1_SK1 = 0;
va_SK = va;
voltageA_SK = voltageA;
 
while abs(error) > [eps eps eps]
    Den1_SK1 = Den1_SK2;
    va_SK = filter(1, Den1_SK2, va_SK);
    voltageA_SK = filter(1, Den1_SK2, voltageA_SK);
    b1_SK = va_SK(4:end); 
    phi1_SK = [-va_SK(3:end-1),-va_SK(2:end-2), voltageA_SK(2:end-2), voltageA_SK(1:end-3)]; 
    theta1_SK = phi1_SK\b1_SK;
    Num1_SK = [0 theta1_SK(2)];
    Den1_SK2 = [1, -theta1_SK(1), 0];
    sys1_SK= tf(Num1_SK, Den1_SK2, Ts);
    error = Den1_SK2-Den1_SK1;
end


    
FRF1_SK = squeeze(freqresp(sys1_SK,2*pi*f));
mag_1_SK = 20*log10(abs(FRF1_SK));
phs_1_SK = 180/pi*unwrap(angle(FRF1_SK)); 
phs_1_SK = 360*ceil(-phs_1_SK(1)/360) + phs_1_SK;

% plot results
figure, hold on
sgtitle("LLS fo complex model with SK filter applied to the input and output data")
subplot(2,1,1),semilogx(f, mag_1_SK)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [m]')
legend('estimated','Location','SouthWest')
axis tight
subplot(2,1,2)
semilogx(f, phs_1_SK)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ]')
legend('estimated','Location','SouthWest')
print -depsc FRF_SK_filter_complex.eps

%% 2.d) Difference between response of the simulated model and the real system: COMPLEX MODEL, WITH SK FILTER
%empirical
va_est_SK1 = lsim(sys1_SK, voltageA_step(106:505), t_step(106:505));
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
pzmap(sys2_SK)

errorSK1= abs(va_step(106:505) - va_est_SK1)-abs(va_step(106:505) - va_estA);



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

% Data cart on ground
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

sgtitle('Motor velocity for wave input with cart on the ground')
print -depsc motor_velocity_groundwave.eps

% Motor A
figure
va_est = lsim(sys_d2,voltageA_gs(149:548),t_gs(149:548));
subplot(2,1,1),plot(t(1:400),[va_gs(149:548) va_est]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_gs(149:548) - va_est))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('Step response motor a old model cart on ground')
print -depsc step_response_ground_simple_a.eps

% Motor B
figure
va_est = lsim(sys_d2,voltageA_gs(149:548),t_gs(149:548));
subplot(2,1,1),plot(t(1:400),[vb_gs(149:548) va_est]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(vb_gs(149:548) - va_est))
legend('error')
xlabel('time [s]')
ylabel('\omega_b(empirical) - \omega_b(estimated) [rad/s]')
axis tight
sgtitle('Step response motor b old model cart on ground')
print -depsc step_response_ground_simple_b.eps

% Emperical reacts slower -> logic, inertia cart


%%% Identify new model%%% 


% suppose J = 0
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
va_est = lsim(sys_gs,voltageA_gs(149:548),t_gs(149:548));
subplot(2,1,1),plot(t(1:400),[va_gs(149:548) va_est]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_gs(149:548) - va_est))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('Step response new model motor a for cart on ground without filter')
print -depsc step_response_ground__newmodel_nofilter_a.eps

error_ground_NF = sum(abs(va_gs(149:548) - va_est));

% Denumerator of simple model cart on ground without filter: Den_gs
sys_gs_SK = sys_gs;
error = [0 0 0];
Den_gs_SK2 = Den_gs;
Den_gs_SK1 = 0;
va_gs_SK = va_gs;
voltageA_gs_SK = voltageA_gs;
 
while abs(error) > [eps eps eps]
    Den_gs_SK1 = Den_gs_SK2;
    va_gs_SK = filter(1, Den_gs_SK2, va_gs_SK);
    voltageA_gs_SK = filter(1, Den_gs_SK2, voltageA_gs_SK);
    b_gs_SK = va_gs_SK(3:end); 
    phi_gs_SK = [va_gs_SK(2:end-1), voltageA_gs_SK(1:end-2)]; 
    theta_gs_SK = phi_gs_SK\b_gs_SK;
    Num_gs_SK = [0 theta_gs_SK(2)];
    Den_gs_SK2 = [1, -theta_gs_SK(1), 0];
    sys_gs_SK= tf(Num_gs_SK, Den_gs_SK2, Ts);
    error = Den_gs_SK2-Den_gs_SK1;
end


    
FRF_gs_SK = squeeze(freqresp(sys_gs_SK,2*pi*f));
mag_gs_SK = 20*log10(abs(FRF_gs_SK));
phs_gs_SK = 180/pi*unwrap(angle(FRF_gs_SK)); 
phs_gs_SK = 360*ceil(-phs_gs_SK(1)/360) + phs_gs_SK;

% plot results
figure, hold on
sgtitle("LLS fo simple model with SK filter applied to the input and output data")
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
print -depsc FRF_SK_filter.eps

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

% Filtering
%cutoff = bandwidth(sys_gs)/(2*pi);
cutoff = 35;
[B_filtc,A_filtc] = butter(6, cutoff*(2/fs));
h = fvtool(B_filtc, A_filtc);

% apply the filter to both input and output
va_gs = filter(B_filtc, A_filtc, va_gs); 
voltageA_gs = filter(B_filtc, A_filtc, voltageA_gs);

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
ylabel('\phi(FRF)  [^\circ]')
legend('estimated','Location','SouthWest')
sgtitle('FRF new estimated model for cart on ground with filter')
print -depsc FRF_ground_newmodel_filter.eps

% Motor A
figure
va_est = lsim(sys_gs,voltageA_gs(149:548),t_gs(149:548));
subplot(2,1,1),plot(t(1:400),[va_gs(149:548) va_est]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_gs(149:548) - va_est))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('Step response new estimated model motor a for cart on ground with filter')
print -depsc step_response_ground__newmodel_filter_a.eps

figure, hold on
pzmap(sys_gs)

error_ground_BW = sum(abs(va_gs(149:548) - va_est));