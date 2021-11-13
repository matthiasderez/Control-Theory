% csvfile = 'recording9.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% data = dlmread(csvfile, ',', 2, 0); % Data follows the labels
% 
% save data
% 
% load data.mat

%% Defining variables
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



%% Plotting data
% motor velocity plots
figure
subplot(121)
plot(t, va)
ylabel('Velocity motor A [rad/s]')
xlabel('t [s]')

subplot(122)
plot(t, vb)
ylabel('Velocity motor B [rad/s]')
xlabel('t [s]')

sgtitle('Motor velocity')
print -depsc motor_velocity.eps

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
figure(3),hold on
subplot(2,1,1),plot(t(1:points_per_period), va_matrix, 'LineWidth', 1) 
grid on
axis tight
xlabel('t  [s]')
ylabel('va  [m/s]')
subplot(2,1,2),plot(t(1:points_per_period), dva_matrix, 'LineWidth', 1)
grid on
axis tight
xlabel('t  [s]')
ylabel('\Delta va  [m/s]')
hold off

figure(4),hold on
subplot(2,1,1),plot(t(1:points_per_period), vb_matrix, 'LineWidth', 1) 
grid on
axis tight
xlabel('t  [s]')
ylabel('vb  [m/s]')
subplot(2,1,2),plot(t(1:points_per_period), dvb_matrix, 'LineWidth', 1)
grid on
axis tight
xlabel('t  [s]')
ylabel('\Delta vb  [m/s]')
hold off


figure(2),hold on
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

%% Least square method, no filtering

% H(z) = (b1*z+b2)/(z(z^2-a1z-a2))
% 
% collect the signals appearing in the difference equation
b1 = va(4:end); 
phi1 = [va(2:end-2), va(1:end-3), voltageA(2:end-2), voltageA(1:end-3)]; 
% perform the fit to get the desired parameters
theta1 = phi1\b1;

% build the identified model
Num1 = [0, theta(3), theta(4)];
Den1 = [1, -theta(1), -theta(2), 0];
sys_d1 = tf(Num1, Den1, Ts);

% compute the frequency response of the identified model
FRF1 = squeeze(freqresp(sys_d1,2*pi*f));
mag_1 = 20*log10(abs(FRF1));
phs_1 = 180/pi*unwrap(angle(FRF1)); 
phs_1 = 360*ceil(-phs_1(1)/360) + phs_1;

% plot the results
figure(4),hold on,
subplot(2,2,1),semilogx(f,mag_m, f, mag_e, f, mag_1)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('|FRF|  [m]')
legend('actual system','empirical', 'estimated','Location','SouthWest')
subplot(2,2,3),semilogx(f, phs_m, f, phs_e, f, phs_1)
grid on
xlim([f(1) f(end)])
xlabel('f  [Hz]')
ylabel('\phi(FRF)  [^\circ]')
legend('actual system','empirical', 'estimated','Location','SouthWest')
x1 = lsim(sys_d1,F_t,t);

subplot(2,2,2),plot(t,[x_t x1]);
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('displacement [m]')
axis tight
subplot(2,2,4),plot(t,abs(x_t - x1))
legend('error')
xlabel('time [s]')
ylabel('displacement [m]')
axis tight

figure(5), hold on
pzmap(sys_d1)





