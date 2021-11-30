%% Explanation naming
% 2 different models are used in this script: 1 refers to the complex model, 2 refers to the simple model. 
% Different filters are used: BW stands for Butterworth, SK stands for Sanathanan Koerner

%% Loading data
close all
clear
% csvfile = '../Data/recording9.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% data = dlmread(csvfile, ',', 2, 0); % Data follows the labels
% 
% save data

%load data.mat

% csvfile = '../Data/Excitation_BlockWave.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% Excitation_BlockWave = dlmread(csvfile, ',', 2, 0); % Data follows the labels
 
%save Excitation_BlockWave.mat

load Excitation_BlockWave.mat


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

load Excitation_BlockWave.mat

%% Defining variables

% Data alternating +6V, 0V, -6V, 0V, input
voltageA = Excitation_BlockWave(:,3);
voltageB = Excitation_BlockWave(:,3);
%positionA = Excitation_BlockWave(:,4);
%positionB = Excitation_BlockWave(:,5);
va  = Excitation_BlockWave(:, 1); % velocity motor a
vb = Excitation_BlockWave(:, 2); % velocity motor b
t = (0:0.01:0.01*(length(va)-1));
% t = Excitation_BlockWave(:,10);
% N = length(t);
% num_periods = 4;
% points_per_period = N/num_periods;
% Ts = 0.01;
% fs = 1/Ts;
% f = [0:N-1]'*(fs/2/N); % arrays of frequencies, 0 to f_s Hz

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
va_gb  = ground_blokfuncties(:, 6);
vb_gb = ground_blokfuncties(:, 7);
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

% %% overlap the plot for different periods to appreciate the noise (less important)
% 
% % separate the different periods (one per column)
% va_matrix = reshape(va,points_per_period,num_periods); %points_per_period= #rows, num_periods = #columns
% vb_matrix = reshape(vb,points_per_period,num_periods);
% voltageA_matrix = reshape(voltageA,points_per_period,num_periods);
% 
% % lets compute the mean of the signals across the periods to have a point of comparison to assess the noise
% va_mean = mean(va_matrix,2); %average over dimension 2 (horizontal = over columns)
% vb_mean = mean(vb_matrix,2);
% voltageA_mean = mean(voltageA_matrix,2);
% 
% %repmat creates a large matrix containing of 1 by num_periods copies of
% %va_mean, so 4 times next to each othes, always 1 vector high
% dva_matrix = va_matrix - repmat(va_mean,1,num_periods); 
% dvb_matrix = vb_matrix - repmat(vb_mean,1,num_periods); 
% dvoltageA_matrix = voltageA_matrix - repmat(voltageA_mean,1,num_periods);
% 
% % plotting some interesting comparisons
% figure,hold on
% subplot(2,1,1),plot(t(1:points_per_period), va_matrix, 'LineWidth', 1) 
% set(gca, 'FontSize', 11)
% grid on
% axis tight
% xlabel('t  [s]')
% ylabel('\omega_a  [rad/s]')
% subplot(2,1,2),plot(t(1:points_per_period), dva_matrix, 'LineWidth', 1)
% set(gca, 'FontSize', 11)
% grid on
% axis tight
% xlabel('t  [s]')
% ylabel('\Delta \omega_a  [rad/s]')
% hold off
% sgtitle('Motor A')
% print -depsc omegaA_deltaomegaA.eps
% 
% figure,hold on
% subplot(2,1,1),plot(t(1:points_per_period), vb_matrix, 'LineWidth', 1) 
% set(gca, 'FontSize', 11)
% grid on
% axis tight
% xlabel('t  [s]')
% ylabel('\omega_b  [rad/s]')
% subplot(2,1,2),plot(t(1:points_per_period), dvb_matrix, 'LineWidth', 1)
% set(gca, 'FontSize', 11)
% grid on
% axis tight
% xlabel('t  [s]')
% ylabel('\Delta \omega_b  [rad/s]')
% hold off
% sgtitle('Motor B')
% print -depsc omegaB_deltaomegaB.eps
% 
% 
% figure,hold on
% subplot(2,1,1),plot(t(1:points_per_period), voltageA_matrix, 'LineWidth', 1) 
% grid on
% axis tight
% xlabel('t  [s]')
% ylabel('voltageA  [V]')
% subplot(2,1,2),plot(t(1:points_per_period), dvoltageA_matrix, 'LineWidth', 1)
% grid on
% axis tight
% xlabel('t  [s]')
% ylabel('\Delta voltageA  [V]')
% hold off
% print -depsc V_deltaV.eps

%% 2.b) Least square method on data motor A, no filtering: COMPLEX MODEL

% H(z) = (b0*z+b1)/(z(z^2+a0z+a1))
% 
% collect the signals appearing in the difference equation
b1 = va(4:end); 
phi1 = [-va(3:end-1), -va(2:end-2), voltageA(2:end-2), voltageA(1:end-3)]; 
% perform the fit to get the desired parameters
theta1 = phi1\b1;

% build the identified model
Num1 = [0, theta1(3), theta1(4)];
Den1 = [1, theta1(1), theta1(2), 0];
sys_1 = tf(Num1, Den1, Ts)

% compute the frequency response of the identified model
% FRF1 = squeeze(freqresp(sys_1,2*pi*f));
% mag_1 = 20*log10(abs(FRF1));
% phs_1 = 180/pi*unwrap(angle(FRF1)); 
% phs_1 = 360*ceil(-phs_1(1)/360) + phs_1;
% 
% % plot the results
% figure,hold on,
% subplot(2,1,1),semilogx(f, mag_1)
% grid on
% xlim([f(1) f(end)])
% xlabel('f  [Hz]')
% ylabel('|FRF|  [rad/s]')
% legend('estimated','Location','SouthWest')
% subplot(2,1,2),semilogx(f, phs_1)
% grid on
% xlim([f(1) f(end)])
% xlabel('f  [Hz]')
% ylabel('\phi(FRF)  [^\circ]')
% legend('estimated','Location','SouthWest')
% sgtitle('FRF complex model')
% print -depsc FRF_complex_model.eps

figure
bode(sys_1)