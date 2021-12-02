%% Explanation naming
% 2 different models are used in this script: 1 refers to the complex model, 2 refers to the simple model. 
% Different filters are used: BW stands for Butterworth, SK stands for Sanathanan Koerner

%% Loading data
close all
clear

% csvfile = '../Data/data.csv';
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

% csvfile = '../Data/ground_squarewave.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% ground_squarewave = dlmread(csvfile, ',', 2, 0); % Data follows the labels
% 
% save ground_squarewave

load ground_squarewave.mat

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
f = [0:N-1]'*(fs/2/N); % arrays of frequencies, 0 to f_s Hz

% Data step input 6V
%voltageA_step = air_step(:,2);
%voltageB_step = air_step(:,3);
va_step  = air_step(:, 6);
vb_step = air_step(:, 7);
t_step = air_step(:,10);

% Data cart on ground (same input as before)
va_gs  = ground_step(:, 6);
vb_gs = ground_step(:, 7);
%voltageA_gs = ground_step(:,2);
%voltageB_gs = ground_step(:,3);
va_gb  = ground_squarewave(:, 6);
vb_gb = ground_squarewave(:, 7);
t_gs = ground_step(:,10);


% CONSTRUEREN VOLTAGES
% voltageA = zeros(4800,1);
% voltageB = zeros(4800,1);
voltageA_step = zeros(600,1);
voltageB_step = zeros(600,1);
voltageA_gs = zeros(600,1);
voltageB_gs = zeros(600,1);

voltageA_step(107:600) = 6;
voltageB_step(107:600) = 6;
voltageA_gs(150:600) = 6;
voltageB_gs(150:600) = 6;


% sign = -1;
% volt = 6;
% grens = 301;
% waarde = 6;
% for i = 2:4800
%     if i > grens
%         volt = volt + sign*waarde;
%         if volt ~= 0
%             sign = -sign;
%         end
%                
%         grens = i+299;
%     end
%     voltageA(i) = volt;
%     voltageB(i) = volt;
% end


%% Plotting data
% motor velocity plots
figure
subplot(121)
plot(t, va)
set(gca, 'FontSize', 11)
ylabel('\omega_a [rad/s]')
xlabel('t [s]')

subplot(122)
plot(t, vb)
set(gca, 'FontSize', 11)
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
set(gca, 'FontSize', 11)
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
set(gca, 'FontSize', 11)
grid on
axis tight
xlabel('t  [s]')
ylabel('\omega_a  [rad/s]')
subplot(2,1,2),plot(t(1:points_per_period), dva_matrix, 'LineWidth', 1)
set(gca, 'FontSize', 11)
grid on
axis tight
xlabel('t  [s]')
ylabel('\Delta \omega_a  [rad/s]')
hold off
sgtitle('Motor A')
print -depsc omegaA_deltaomegaA.eps

figure,hold on
subplot(2,1,1),plot(t(1:points_per_period), vb_matrix, 'LineWidth', 1) 
set(gca, 'FontSize', 11)
grid on
axis tight
xlabel('t  [s]')
ylabel('\omega_b  [rad/s]')
subplot(2,1,2),plot(t(1:points_per_period), dvb_matrix, 'LineWidth', 1)
set(gca, 'FontSize', 11)
grid on
axis tight
xlabel('t  [s]')
ylabel('\Delta \omega_b  [rad/s]')
hold off
sgtitle('Motor B')
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

% H(z) = (b0*z+b1)/(z(z^2+a0z+a1))

%%% MOTOR A 

% collect the signals appearing in the difference equation
b_1A = va(4:end); 
phi_1A = [-va(3:end-1), -va(2:end-2), voltageA(2:end-2), voltageA(1:end-3)]; 

% perform the fit to get the desired parameters
theta_1A = phi_1A\b_1A;

% build the identified model
Num_1A = [0, theta_1A(3), theta_1A(4)];
Den_1A = [1, theta_1A(1), theta_1A(2), 0];
sys_1A = tf(Num_1A, Den_1A, Ts)

%%% MOTOR B 

% collect the signals appearing in the difference equation
b_1B = vb(4:end); 
phi_1B = [-vb(3:end-1), -vb(2:end-2), voltageB(2:end-2), voltageB(1:end-3)]; 

% perform the fit to get the desired parameters
theta_1B = phi_1B\b_1B;

% build the identified model
Num_1B = [0, theta_1B(3), theta_1B(4)];
Den_1B = [1, theta_1B(1), theta_1B(2), 0];
sys_1B = tf(Num_1B, Den_1B, Ts)

% Bode plots
figure, hold on
bode(sys_1A)
bode(sys_1B)
legend('Motor A', 'Motor B')
hold off

% Pole-Zero map
figure, hold on
pzmap(sys_1A, sys_1B)
set(gca, 'FontSize', 11)
legend('Motor A', 'Motor B')
print -depsc p&z_complex.eps

%% 2.b) Least square method on data motor A, no filtering: SIMPLE MODEL

% suppose L = 0
% H(z) = b0/(z(z+a0))

%%% Motor A

% collect the signals appearing in the difference equation
b_2A = va(3:end); 
phi_2A = [-va(2:end-1), voltageA(1:end-2)]; 

% perform the fit to get the desired parameters
theta_2A = phi_2A\b_2A;

% build the identified model
Num_2A = [theta_2A(2)];
Den_2A = [1, theta_2A(1), 0];
sys_2A = tf(Num_2A, Den_2A, Ts)

%%% Motor B

% collect the signals appearing in the difference equation
b_2B = vb(3:end); 
phi_2B = [-vb(2:end-1), voltageB(1:end-2)]; 

% perform the fit to get the desired parameters
theta_2B = phi_2B\b_2B;

% build the identified model
Num_2B = [theta_2B(2)];
Den_2B = [1, theta_2B(1), 0];
sys_2B = tf(Num_2B, Den_2B, Ts)

% Bode plots
figure, hold on
bode(sys_2A)
bode(sys_2B)
legend('Motor A', 'Motor B')
hold off

% Pole-Zero map
figure, hold on
pzmap(sys_2A, sys_2B)
set(gca, 'FontSize', 11)
legend('Motor A', 'Motor B')
print -depsc p&z_simple.eps


%% 2.c) Filtering with Butterworth filter: COMPLEX MODEL 

%%% Motor A 

% define a low(band)-pass filter
cutoff = bandwidth(sys_1A)/(2*pi);
[B_BW_1A,A_BW_1A] = butter(6, cutoff*(2/fs));
%h = fvtool(B_BW_1A, A_BW_1A);

% apply the filter to both input and output
va_BW_1A = filter(B_BW_1A, A_BW_1A, va); 
voltageA_BW_1A = filter(B_BW_1A, A_BW_1A, voltageA);

%repeat the identification
b_BW_1A = va_BW_1A(4:end); 
phi_BW_1A = [-va_BW_1A(3:end-1),-va_BW_1A(2:end-2), voltageA_BW_1A(2:end-2), voltageA_BW_1A(1:end-3)]; 
theta_BW_1A = phi_BW_1A\b_BW_1A;
Num_BW_1A = [0,theta_BW_1A(3),theta_BW_1A(4)];
Den_BW_1A = [1, theta_BW_1A(1),theta_BW_1A(2), 0];
sys_BW_1A = tf(Num_BW_1A, Den_BW_1A, Ts)

%%% Motor B

% define a low(band)-pass filter
cutoff = bandwidth(sys_1B)/(2*pi);
[B_BW_1B,A_BW_1B] = butter(6, cutoff*(2/fs));
%h = fvtool(B_BW_1B, A_BW_1B);

% apply the filter to both input and output
vb_BW_1B = filter(B_BW_1B, A_BW_1B, vb); 
voltageB_BW_1B = filter(B_BW_1B, A_BW_1B, voltageB);

%repeat the identification
b_BW_1B = vb_BW_1B(4:end); 
phi_BW_1B = [-vb_BW_1B(3:end-1),-vb_BW_1B(2:end-2), voltageB_BW_1B(2:end-2), voltageB_BW_1B(1:end-3)]; 
theta_BW_1B = phi_BW_1B\b_BW_1B;
Num_BW_1B = [0,theta_BW_1B(3),theta_BW_1B(4)];
Den_BW_1B = [1, theta_BW_1B(1),theta_BW_1B(2), 0];
sys_BW_1B = tf(Num_BW_1B, Den_BW_1B, Ts)

% Bode plots
figure, hold on
bode(sys_BW_1A)
bode(sys_BW_1B)
legend('Motor A', 'Motor B')
sgtitle('Complex model, filtered')
hold off

% Pole-Zero map
figure, hold on
pzmap(sys_BW_1A, sys_BW_1B)
set(gca, 'FontSize', 11)
legend('Motor A', 'Motor B')
print -depsc p&z_complex_BW.eps

figure, hold on
bode(sys_1A)
bode(sys_2A)
bode(sys_BW_1A)
legend('Complex','Simple','Complex BW')
sgtitle('Comparison for motor A')
hold off

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
ylabel('|FRF|  [dB]')
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
ylabel('|FRF|  [dB]')
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
ylabel('|FRF|  [dB]')
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

figure, hold on
bode(sys_1)
bode(sys_2)
bode(sys_SK1)
bode(sys_SK2)
legend('complex model', 'simple model', 'complex model with SK', 'simple model with SK', 'Location','SouthWest')
hold off




%% 2.d) Difference between response of the simulated model and the emperical values: NO FILTER, COMPLEX MODEL 

% Motor A
va_est_1A = lsim(sys_1A,voltageA_step(106:505),t_step(106:505));               % start at 106, moment of step
figure
subplot(2,1,1),plot(t(1:400),[va_step(106:505) va_est_1A]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(106:505) - va_est_1A))
legend('error')
xlabel('time [s]')
ylabel('omega_{a empirical} - \omega_{a estimated} [rad/s]')
axis tight
sgtitle('Step response complex model motor a')
print -depsc step_response_complex_a.eps

error_1A_tot= sum(abs(va_step(106:505) - va_est_1A))

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




%% 2.d) Difference between response of the simulated model and the real system: SIMPLE MODEL, NO FILTER

% Motor A stepinput
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
error_2_tot= sum(abs(va_step(106:505) - va_estA));

figure
plot(t(1:400),abs(va_step(106:505) - va_estA)-abs(va_step(106:505) - va_est1))
xlabel('time [s]')
ylabel('error_{simple} - error_{complex}[rad/s]')


% Motor B stepinput
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

% Motor A square wave input
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

% Motor B square wave input
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
va_est_BW_1A = lsim(sys_BW_1A, voltageA_step(106:505), t_step(106:505));
figure
subplot(2,1,1),plot(t(1:400),[va_step(106:505) va_est_BW_1A]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(106:505) - va_est_BW_1A))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('Step response of complex model with Butterworth filter')
print -depsc step_response_complex_BW_a.eps

error_BW1_tot= sum(abs(va_step(106:505) - va_est_BW_1A))

figure,hold on
pzmap(sys_BW_1A)
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
error_BW2_tot= sum(abs(va_step(106:505) - va_est_BW2));

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

error_SK1_tot= sum(abs(va_step(106:505) - va_est_SK1));




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

error_SK2_tot= sum(abs(va_step(106:505) - va_est_SK2));




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
% va_gs  = ground_step(:, 6);
% vb_gs = ground_step(:, 7);
% voltageA_gs = ground_step(:,2);
% voltageB_gs = ground_step(:,3);
% va_gb  = ground_squarewave(:, 6);
% vb_gb = ground_squarewave(:, 7);
% t_gs = ground_step(:,10);

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
b_gb = va_gb(3:end); 
phi_gb = [va_gb(2:end-1), voltageA(1:end-2)]; 
% perform the fit to get the desired parameters

theta_gb = phi_gb\b_gb;

% build the identified model
Num_gb = [theta_gb(2)];
Den_gb = [1, -theta_gb(1), 0];
sys_g = tf(Num_gb, Den_gb, Ts);

% compute the frequency response of the identified model
FRF_gb = squeeze(freqresp(sys_g,2*pi*f));
mag_gb = 20*log10(abs(FRF_gb));
phs_gb = 180/pi*unwrap(angle(FRF_gb)); 
phs_gb = 360*ceil(-phs_gb(1)/360) + phs_gb;

% plot the results
figure,hold on,
subplot(2,1,1),semilogx(f, mag_gb)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [dB]')
legend('estimated','Location','SouthWest')
subplot(2,1,2),semilogx(f, phs_gb)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ] ')
legend('estimated','Location','SouthWest')
sgtitle('FRF new estimated model for cart on ground without filter')
print -depsc FRF_ground_newmodel_nofilter.eps

% Motor A
figure
va_est_gs = lsim(sys_g,voltageA_gs(149:548),t_gs(149:548));
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


% Denumerator of simple model cart on ground without filter: Den_gb
sys_g_SK = sys_g;
error_gb_SK = [100 100 100];
Den_gb_SK_2 = Den_gb;
Den_gb_SK_1 = 0;
va_gb_SK = va_gb;
voltageA_gb_SK = voltageA;
 
while abs(error_gb_SK) > [eps eps eps]
    Den_gb_SK_1 = Den_gb_SK_2;
    va_gb_SK = filter(1, Den_gb_SK_2, va_gb_SK);
    voltageA_gb_SK = filter(1, Den_gb_SK_2, voltageA_gb_SK);
    b_gb_SK = va_gb_SK(3:end); 
    phi_gb_SK = [va_gb_SK(2:end-1), voltageA_gb_SK(1:end-2)]; 
    theta_gb_SK = phi_gb_SK\b_gb_SK;
    Num_gb_SK = [0 theta_gb_SK(2)];
    Den_gb_SK_2 = [1, -theta_gb_SK(1), 0];
    sys_g_SK= tf(Num_gb_SK, Den_gb_SK_2, Ts);
    error_gb_SK = Den_gb_SK_2-Den_gb_SK_1;
end


    
FRF_gb_SK = squeeze(freqresp(sys_g_SK,2*pi*f));
mag_gb_SK = 20*log10(abs(FRF_gb_SK));
phs_gb_SK = 180/pi*unwrap(angle(FRF_gb_SK)); 
phs_gb_SK = 360*ceil(-phs_gb_SK(1)/360) + phs_gb_SK;

% plot results
figure, hold on
sgtitle("FRF new estimated model for cart on ground with SK filter")
subplot(2,1,1),semilogx(f, mag_gb_SK)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [dB]')
legend('estimated','Location','SouthWest')
axis tight
subplot(2,1,2)
semilogx(f, phs_gb_SK)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ]')
legend('estimated','Location','SouthWest')
print -depsc FRF_SK_filter_ground.eps

% Motor A
figure
va_est_gs_SK = lsim(sys_g_SK,voltageA_gs(149:548),t_gs(149:548));
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

%cutoff = bandwidth(sys_g_BW)/(2*pi);
cutoff = 35;
[B_gb_BW,A_gb_BW] = butter(6, cutoff*(2/fs));
h = fvtool(B_gb_BW, A_gb_BW);

% apply the filter to both input and output
va_gb_BW = filter(B_gb_BW, A_gb_BW, va_gb); 
voltageA_gb_BW = filter(B_gb_BW, A_gb_BW, voltageA);

%repeat the identification
b_gb_BW = va_gb_BW(3:end); 
phi_gb_BW = [va_gb_BW(2:end-1), voltageA_gb_BW(1:end-2)]; 
theta1_gb_BW = phi_gb_BW\b_gb_BW;
Num_gb_BW = [theta1_gb_BW(2)];
Den_gb_BW = [1, -theta1_gb_BW(1), 0];
sys_g_BW = tf(Num_gb_BW, Den_gb_BW, Ts);

% compute the frequency response of the identified model
FRF_gb_BW = squeeze(freqresp(sys_g_BW,2*pi*f));
mag_gb_BW = 20*log10(abs(FRF_gb_BW));
phs_gb_BW = 180/pi*unwrap(angle(FRF_gb_BW)); 
phs_gb_BW = 360*ceil(-phs_gb_BW(1)/360) + phs_gb_BW;

% plot the results
figure,hold on,
subplot(2,1,1),semilogx(f, mag_gb_BW)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [dB]')
legend('estimated','Location','SouthWest')
subplot(2,1,2),semilogx(f, phs_gb_BW)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ]')
legend('estimated','Location','SouthWest')
sgtitle('FRF new estimated model for cart on ground with Butterworth filter')
print -depsc FRF_ground_newmodel_filter.eps

% Motor A
figure
va_est_gs_BW = lsim(sys_g_BW,voltageA_gs(149:548),t_gs(149:548));
subplot(2,1,1),plot(t(1:400),[va_gs(149:548) va_est_gs_BW]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_gs(149:548) - va_est_gs_BW))
legend('error')
xlabel('time [s]')
ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
axis tight
sgtitle('Step response new estimated model motor a for cart on ground with Butterworth filter')
print -depsc step_response_ground__newmodel_filter_a.eps

figure, hold on
pzmap(sys_g_BW)

error_ground_BW = sum(abs(va_gs(149:548) - va_est_gs_BW));


%% Saving variables

save('assignment1_def.mat');