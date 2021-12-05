%% Explanation naming
% 2 different models are used in this script: 1 refers to the complex model, 2 refers to the simple model. 
% Different filters are used: BW stands for Butterworth, SK stands for Sanathanan Koerner
% A refers to motor A, B to motor B

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
voltageA_step = air_step(:,2);
voltageB_step = air_step(:,3);
va_step  = air_step(:, 6);
vb_step = air_step(:, 7);
t_step = air_step(:,10);

% Data cart on ground (same input as before)
ground_squarewave = ground_squarewave(1:4800,:);
va_gs  = ground_step(:, 6);
vb_gs = ground_step(:, 7);
voltageA_gs = ground_step(:,2);
voltageB_gs = ground_step(:,3);
va_gb  = ground_squarewave(:, 6);
vb_gb = ground_squarewave(:, 7);
t_gb = ground_squarewave(:, 10);
t_gs = ground_step(:,10);



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
cutoff_1A = 0.017*bandwidth(sys_1A)/(2*pi);
[B_BW_1A,A_BW_1A] = butter(6, cutoff_1A*(2/fs));
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
cutoff_1B = bandwidth(sys_1B)/(2*pi);
[B_BW_1B,A_BW_1B] = butter(6, cutoff_1B*(2/fs));
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

%% 2.c) Filtering with Butterworth filter: SIMPLE MODEL

%%% Motor A

% define a low(band)-pass filter
cutoff_2A = bandwidth(sys_2A)/(2*pi);
[B_BW_2A,A_BW_2A] = butter(6, cutoff_2A*(2/fs));
%h = fvtool(B_BW_2A, A_BW_2A);

% apply the filter to both input and output
va_BW_2A = filter(B_BW_2A, A_BW_2A, va); 
voltageA_BW_2A = filter(B_BW_2A, A_BW_2A, voltageA);

%repeat the identification
b_BW_2A = va_BW_2A(3:end); 
phi_BW_2A = [-va_BW_2A(2:end-1), voltageA_BW_2A(1:end-2)]; 
theta_BW_2A = phi_BW_2A\b_BW_2A;
Num_BW_2A = [theta_BW_2A(2)];
Den_BW_2A = [1, theta_BW_2A(1), 0];
sys_BW_2A = tf(Num_BW_2A, Den_BW_2A, Ts)

%%% Motor B

% define a low(band)-pass filter
cutoff_2B = bandwidth(sys_2B)/(2*pi);
[B_BW_2B,A_BW_2B] = butter(6, cutoff_2B*(2/fs));
%h = fvtool(B_BW_2B, A_BW_2B);

% apply the filter to both input and output
vb_BW_2B = filter(B_BW_2B, A_BW_2B, vb); 
voltageB_BW_2B = filter(B_BW_2B, A_BW_2B, voltageB);

%repeat the identification
b_BW_2B = vb_BW_2B(3:end); 
phi_BW_2B = [-vb_BW_2B(2:end-1), voltageB_BW_2B(1:end-2)]; 
theta_BW_2B = phi_BW_2B\b_BW_2B;
Num_BW_2B = [theta_BW_2B(2)];
Den_BW_2B = [1, theta_BW_2B(1), 0];
sys_BW_2B = tf(Num_BW_2B, Den_BW_2B, Ts)

% Bode plots
figure, hold on
bode(sys_BW_2A)
bode(sys_BW_2B)
legend('Motor A', 'Motor B')
sgtitle('Simple model, filtered')
hold off

% Pole-Zero map
figure, hold on
pzmap(sys_BW_2A, sys_BW_2B)
set(gca, 'FontSize', 11)
legend('Motor A', 'Motor B')
print -depsc p&z_simple_BW.eps


%% 2.c) Filtering Sanathanan Koerner procedure: COMPLEX MODEL

%%% Motor A

sys_SK_1A = sys_1A;
error_SK_1A = [100 100 100 100];
Den_SK_1A_j = Den_1A;
Den_SK_1A_i = 0;
va_SK_1A = va;
voltageA_SK_1A = voltageA;
 
while abs(error_SK_1A) > [eps eps eps eps]
    % apply the filter to both input and output
    Den_SK_1A_i = Den_SK_1A_j;
    va_SK_1A = filter(1, Den_SK_1A_j, va_SK_1A);
    voltageA_SK_1A = filter(1, Den_SK_1A_j, voltageA_SK_1A);
    
    % repeat the identification
    b_SK_1A = va_SK_1A(4:end); 
    phi_SK_1A = [-va_SK_1A(3:end-1),-va_SK_1A(2:end-2), voltageA_SK_1A(2:end-2), voltageA_SK_1A(1:end-3)]; 
    theta_SK_1A = phi_SK_1A\b_SK_1A;
    Num_SK_1A = [0, theta_SK_1A(3), theta_SK_1A(4) ];
    Den_SK_1A_j = [1, theta_SK_1A(1),  theta_SK_1A(2), 0];
    sys_SK_1A = tf(Num_SK_1A, Den_SK_1A_j, Ts);
    
    % error criterion
    error_SK_1A = Den_SK_1A_j-Den_SK_1A_i;
end


%%% Motor B

sys_SK_1B = sys_1B;
error_SK_1B = [100 100 100 100];
Den_SK_1B_j = Den_1B;
Den_SK_1B_i = 0;
vb_SK_1B = vb;
voltageB_SK_1B = voltageB;
 
while abs(error_SK_1B) > [eps eps eps eps]
    % apply the filter to both input and output
    Den_SK_1B_i = Den_SK_1B_j;
    vb_SK_1B = filter(1, Den_SK_1B_j, vb_SK_1B);
    voltageB_SK_1B = filter(1, Den_SK_1B_j, voltageB_SK_1B);
    
    % repeat the identification
    b_SK_1B = vb_SK_1B(4:end); 
    phi_SK_1B = [-vb_SK_1B(3:end-1),-vb_SK_1B(2:end-2), voltageB_SK_1B(2:end-2), voltageB_SK_1B(1:end-3)]; 
    theta_SK_1B = phi_SK_1B\b_SK_1B;
    Num_SK_1B = [0, theta_SK_1B(3), theta_SK_1B(4) ];
    Den_SK_1B_j = [1, theta_SK_1B(1),  theta_SK_1B(2), 0];
    sys_SK_1B = tf(Num_SK_1B, Den_SK_1B_j, Ts);
    
    % error criterion
    error_SK_1B = Den_SK_1B_j-Den_SK_1B_i;
end


% Bode plots
figure, hold on
bode(sys_SK_1A)
bode(sys_SK_1B)
legend('Motor A', 'Motor B')
sgtitle('Complex model, SK filter')
hold off

% Pole-Zero map
figure, hold on
pzmap(sys_SK_1A, sys_SK_1B)
set(gca, 'FontSize', 11)
legend('Motor A', 'Motor B')
print -depsc p&z_complex_SK.eps

figure, hold on
pzplot(sys_1A, 'b', sys_BW_1A, 'r', sys_SK_1A, 'g')
set(gca, 'FontSize', 11)
legend('No filter', 'Butterworth', 'Sanathanan Koerner')
hold off
print -depsc p&z_complex_all_A.eps

figure, hold on
pzplot(sys_1B, 'b', sys_BW_1B, 'r', sys_SK_1B, 'g')
set(gca, 'FontSize', 11)
legend('No filter', 'Butterworth', 'Sanathanan Koerner')
hold off
print -depsc p&z_complex_all_B.eps

%% 2.c) Filtering Sanathanan Koerner procedure: SIMPLE MODEL

%%% Motor A

sys_SK_2A = sys_2A;
error_SK_2A = [100 100 100];
Den_SK_2A_j = Den_2A;
Den_SK_2A_i = 0;
va_SK_2A = va;
voltageA_SK_2A = voltageA;
 
while abs(error_SK_2A) > [eps eps eps]
    % apply the filter to both input and output
    Den_SK_2A_i = Den_SK_2A_j;
    va_SK_2A = filter(1, Den_SK_2A_j, va_SK_2A);
    voltageA_SK_2A = filter(1, Den_SK_2A_j, voltageA_SK_2A);
    
    % repeat the identification
    b_SK_2A = va_SK_2A(3:end); 
    phi_SK_2A = [-va_SK_2A(2:end-1), voltageA_SK_2A(1:end-2)]; 
    theta_SK_2A = phi_SK_2A\b_SK_2A;
    Num_SK_2A = [theta_SK_2A(2)];
    Den_SK_2A_j = [1, theta_SK_2A(1), 0];
    sys_SK_2A = tf(Num_SK_2A, Den_SK_2A_j, Ts);
    
    % error criterion
    error_SK_2A = Den_SK_2A_j-Den_SK_2A_i;
end


%%% Motor B

sys_SK_2B = sys_2B;
error_SK_2B = [100 100 100];
Den_SK_2B_j = Den_2B;
Den_SK_2B_i = 0;
vb_SK_2B = vb;
voltageB_SK_2B = voltageB;
 
while abs(error_SK_2B) > [eps eps eps]
    % apply the filter to both input and output
    Den_SK_2B_i = Den_SK_2B_j;
    vb_SK_2B = filter(1, Den_SK_2B_j, vb_SK_2B);
    voltageB_SK_2B = filter(1, Den_SK_2B_j, voltageB_SK_2B);
    
    % repeat the identification
    b_SK_2B = vb_SK_2B(3:end); 
    phi_SK_2B = [-vb_SK_2B(2:end-1), voltageB_SK_2B(1:end-2)]; 
    theta_SK_2B = phi_SK_2B\b_SK_2B;
    Num_SK_2B = [theta_SK_2B(2)];
    Den_SK_2B_j = [1, theta_SK_2B(1), 0];
    sys_SK_2B = tf(Num_SK_2B, Den_SK_2B_j, Ts);
    
    % error criterion
    error_SK_2B = Den_SK_2B_j-Den_SK_2B_i;
end


% Bode plots
figure, hold on
bode(sys_SK_2A)
bode(sys_SK_2B)
legend('Motor A', 'Motor B')
sgtitle('Simple model, SK filter')
hold off

% Pole-Zero map
figure, hold on
pzmap(sys_SK_2A, sys_SK_2B)
set(gca, 'FontSize', 11)
legend('Motor A', 'Motor B')
print -depsc p&z_simple_SK.eps

figure, hold on
pzplot(sys_2A, 'b', sys_BW_2A, 'r', sys_SK_2A, 'g')
set(gca, 'FontSize', 11)
legend('No filter', 'Butterworth', 'Sanathanan Koerner')
hold off
print -depsc p&z_simple_all_A.eps

figure, hold on
pzplot(sys_2B, 'b', sys_BW_2B, 'r', sys_SK_2B, 'g')
set(gca, 'FontSize', 11)
legend('No filter', 'Butterworth', 'Sanathanan Koerner')
hold off
print -depsc p&z_simple_all_B.eps

%% 2.d) Difference between response of the simulated model and the emperical values: COMPLEX MODEL, NO FILTER

% Motor A
va_est_1A = lsim(sys_1A,voltageA_step(1:400),t_step(1:400));              
figure
subplot(2,1,1),plot(t(1:400),[va_step(1:400) va_est_1A]);
set(gca, 'FontSize', 11)
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(1:400) - va_est_1A))
set(gca, 'FontSize', 11)
legend('absolute error')
xlabel('time [s]')
ylabel('|\omega_{a emp} - \omega_{a est}| [rad/s]')
axis tight
sgtitle('Complex model - Motor A', 'fontweight', 'bold')
print -depsc step_response_complex_a.eps

error_1A_tot = sum(abs(va_step(1:400) - va_est_1A))

% Motor B
vb_est_1B = lsim(sys_1B,voltageB_step(1:400),t_step(1:400));
figure
subplot(2,1,1),plot(t(1:400),[vb_step(1:400) vb_est_1B]);
set(gca, 'FontSize', 11)
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(vb_step(1:400) - vb_est_1B))
set(gca, 'FontSize', 11)
legend('absolute error')
xlabel('time [s]')
ylabel('|\omega_{b emp} - \omega_{b est}| [rad/s]')
axis tight
sgtitle('Complex model - Motor B', 'fontweight','bold')
print -depsc step_response_complex_b.eps

error_1B_tot = sum(abs(vb_step(1:400) - vb_est_1B))

%% 2.d) Difference between response of the simulated model and the real system: SIMPLE MODEL, NO FILTER

% Motor A
va_est_2A = lsim(sys_2A,voltageA_step(1:400),t_step(1:400));
figure
subplot(2,1,1),plot(t(1:400),[va_step(1:400) va_est_2A]);
set(gca, 'FontSize', 11)
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(1:400) - va_est_2A))
set(gca, 'FontSize', 11)
legend('absolute error')
xlabel('time [s]')
ylabel('|\omega_{a emp} - \omega_{a est}| [rad/s]')
axis tight
sgtitle('Simple model - Motor A', 'fontweight', 'bold')
print -depsc step_response_simple_a.eps

error_2A_tot = sum(abs(va_step(1:400) - va_est_2A))

figure
plot(t(1:400),abs(va_step(1:400) - va_est_2A) - abs(va_step(1:400) - va_est_1A))
set(gca, 'FontSize', 11)
xlabel('time [s]')
ylabel('error_{simple} - error_{complex}[rad/s]')

% Motor B 
vb_est_2B = lsim(sys_2B,voltageB_step(1:400),t_step(1:400));
figure
subplot(2,1,1),plot(t(1:400),[vb_step(1:400) vb_est_2B]);
set(gca, 'FontSize', 11)
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(vb_step(1:400) - vb_est_2B))
set(gca, 'FontSize', 11)
legend('absolute error')
xlabel('time [s]')
ylabel('|\omega_{b emp} - \omega_{b est}| [rad/s]')
axis tight
sgtitle('Simple model - Motor B', 'fontweight', 'bold')
print -depsc step_response_simple_b.eps

error_2B_tot = sum(abs(vb_step(1:400) - vb_est_2B))

%% 2.d) Difference between response of the simulated model and the real system: COMPLEX MODEL, BUTTERWORTH FILTER

% Motor A 
va_est_BW_1A = lsim(sys_BW_1A,voltageA_step(1:400),t_step(1:400));              
figure
subplot(2,1,1),plot(t(1:400),[va_step(1:400) va_est_BW_1A]);
set(gca, 'FontSize', 11)
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(1:400) - va_est_BW_1A))
set(gca, 'FontSize', 11)
legend('absolute error')
xlabel('time [s]')
ylabel('|\omega_{a emp} - \omega_{a est}| [rad/s]')
axis tight
sgtitle('Complex model - Butterworth filter - Motor A', 'fontweight', 'bold')
print -depsc step_response_complex_BW_a.eps

error_BW_1A_tot = sum(abs(va_step(1:400) - va_est_BW_1A))

% Motor B 
vb_est_BW_1B = lsim(sys_BW_1B,voltageB_step(1:400),t_step(1:400));              
figure
subplot(2,1,1),plot(t(1:400),[vb_step(1:400) vb_est_BW_1B]);
set(gca, 'FontSize', 11)
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(vb_step(1:400) - vb_est_BW_1B))
set(gca, 'FontSize', 11)
legend('absolute error')
xlabel('time [s]')
ylabel('|\omega_{b emp} - \omega_{b est}| [rad/s]')
axis tight
sgtitle('Complex model - Butterworth filter - Motor B', 'fontweight', 'bold')
print -depsc step_response_complex_BW_b.eps

error_BW_1B_tot = sum(abs(vb_step(1:400) - vb_est_BW_1B))

%% 2.d) Difference between response of the simulated model and the real system: SIMPLE MODEL, BUTTERWORTH FILTER

% Motor A 
va_est_BW_2A = lsim(sys_BW_2A,voltageA_step(1:400),t_step(1:400));              
figure
subplot(2,1,1),plot(t(1:400),[va_step(1:400) va_est_BW_2A]);
set(gca, 'FontSize', 11)
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(1:400) - va_est_BW_2A))
set(gca, 'FontSize', 11)
legend('absolute error')
xlabel('time [s]')
ylabel('|\omega_{a emp} - \omega_{a est}| [rad/s]')
axis tight
sgtitle('Simple model - Butterworth filter - Motor A', 'fontweight', 'bold')
print -depsc step_response_simple_BW_a.eps

error_BW_2A_tot = sum(abs(va_step(1:400) - va_est_BW_2A))

% Motor B 
vb_est_BW_2B = lsim(sys_BW_2B,voltageB_step(1:400),t_step(1:400));              
figure
subplot(2,1,1),plot(t(1:400),[vb_step(1:400) vb_est_BW_2B]);
set(gca, 'FontSize', 11)
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(vb_step(1:400) - vb_est_BW_2B))
set(gca, 'FontSize', 11)
legend('absolute error')
xlabel('time [s]')
ylabel('|\omega_{b emp} - \omega_{b est}| [rad/s]')
axis tight
sgtitle('Simple model - Butterworth filter - Motor B', 'fontweight', 'bold')
print -depsc step_response_simple_BW_b.eps

error_BW_2B_tot = sum(abs(vb_step(1:400) - vb_est_BW_2B))

%% 2.d) Difference between response of the simulated model and the real system: COMPLEX MODEL, SK FILTER

% Motor A 
va_est_SK_1A = lsim(sys_SK_1A,voltageA_step(1:400),t_step(1:400));              
figure
subplot(2,1,1),plot(t(1:400),[va_step(1:400) va_est_SK_1A]);
set(gca, 'FontSize', 11)
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(1:400) - va_est_SK_1A))
set(gca, 'FontSize', 11)
legend('absolute error')
xlabel('time [s]')
ylabel('|\omega_{a emp} - \omega_{a est}| [rad/s]')
axis tight
sgtitle('Complex model - Sanathanan Koerner filter - Motor A', 'fontweight', 'bold')
print -depsc step_response_complex_SK_a.eps

error_SK_1A_tot = sum(abs(va_step(1:400) - va_est_SK_1A))

% Motor B 
vb_est_SK_1B = lsim(sys_SK_1B,voltageB_step(1:400),t_step(1:400));              
figure
subplot(2,1,1),plot(t(1:400),[vb_step(1:400) vb_est_SK_1B]);
set(gca, 'FontSize', 11)
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(vb_step(1:400) - vb_est_SK_1B))
set(gca, 'FontSize', 11)
legend('absolute error')
xlabel('time [s]')
ylabel('|\omega_{b emp} - \omega_{b est}| [rad/s]')
axis tight
sgtitle('Complex model - Sanathanan Koerner filter - Motor B', 'fontweight', 'bold')
print -depsc step_response_complex_SK_b.eps

error_SK_1B_tot = sum(abs(vb_step(1:400) - vb_est_SK_1B))

%% 2.d) Difference between response of the simulated model and the real system: SIMPLE MODEL, SK FILTER

% Motor A 
va_est_SK_2A = lsim(sys_SK_2A,voltageA_step(1:400),t_step(1:400));              
figure
subplot(2,1,1),plot(t(1:400),[va_step(1:400) va_est_SK_2A]);
set(gca, 'FontSize', 11)
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(va_step(1:400) - va_est_SK_2A))
set(gca, 'FontSize', 11)
legend('absolute error')
xlabel('time [s]')
ylabel('|\omega_{a emp} - \omega_{a est}| [rad/s]')
axis tight
sgtitle('Simple model - Sanathanan Koerner filter - Motor A', 'fontweight', 'bold')
print -depsc step_response_simple_SK_a.eps

error_SK_2A_tot = sum(abs(va_step(1:400) - va_est_SK_2A))

% Motor B 
vb_est_SK_2B = lsim(sys_SK_2B,voltageB_step(1:400),t_step(1:400));              
figure
subplot(2,1,1),plot(t(1:400),[vb_step(1:400) vb_est_SK_2B]);
set(gca, 'FontSize', 11)
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t(1:400),abs(vb_step(1:400) - vb_est_SK_2B))
set(gca, 'FontSize', 11)
legend('absolute error')
xlabel('time [s]')
ylabel('|\omega_{b emp} - \omega_{b est}| [rad/s]')
axis tight
sgtitle('Simple model - Sanathanan Koerner filter - Motor B', 'fontweight', 'bold')
print -depsc step_response_simple_SK_b.eps

error_SK_2B_tot = sum(abs(vb_step(1:400) - vb_est_SK_2B))




%% 2.d) Superposition principle

va10  = data10(:, 6); % data10 contains data resulting from a square wave input with amplitude 4 V
vb10 = data10(:, 7);
va11  = data11(:, 6); % data11 contains data resulting from a square wave input with amplitude 10 V
vb11 = data11(:, 7);

figure
subplot(121)
plot(t(1:300), [va(1:300) + va10(1:300), va11(1:300)])
set(gca, 'FontSize', 11)
ylabel('\omega_a [rad/s]')
xlabel('t [s]')
legend('\omega_{a\_6V} + \omega_{a\_4V}','\omega_{a\_10V}', 'Location', 'South', 'FontSize', 11)
subplot(122)
plot(t(1:300), abs(va(1:300) + va10(1:300) - va11(1:300)))
set(gca, 'FontSize', 11)
ylabel('|\omega_{a\_6V} + \omega_{a\_4V} -  \omega_{a\_10V}| [rad/s]')
xlabel('t [s]')
legend('absolute error')
print -depsc superposition_a.eps


figure
subplot(121)
plot(t(1:300), [vb(1:300) + vb10(1:300), vb11(1:300)])
set(gca, 'FontSize', 11)
ylabel('\omega_b [rad/s]')
xlabel('t [s]')
legend('\omega_{b\_6V} + \omega_{b\_4V}','\omega_{b\_10V}', 'Location', 'South', 'FontSize', 11)
subplot(122)
plot(t(1:300), abs(vb(1:300) + vb10(1:300) - vb11(1:300)))
set(gca, 'FontSize', 11)
ylabel('|\omega_{b\_6V} + \omega_{b\_4V} -  \omega_{b\_10V}| [rad/s]')
xlabel('t [s]')
legend('absolute error')
print -depsc superposition_b.eps


%% 3.a) Step input to cart on ground

% Remark: only worked with simple model, no 2 used in naming anymore

% _gs --> Ground Stepinput
% _gb --> Ground Blokfunctie input (square wave)

% Motor A
va_est_gs_A = lsim(sys_2A,voltageA_gs(1:400),t_gs(1:400));
figure
subplot(2,1,1),plot(t_gs(1:400),[va_gs(1:400) va_est_gs_A]);
set(gca, 'FontSize', 11)
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t_gs(1:400),abs(va_gs(1:400) - va_est_gs_A))
set(gca, 'FontSize', 11)
legend('absolute error')
xlabel('time [s]')
ylabel('|\omega_{a emp} - \omega_{a est}| [rad/s]')
axis tight
sgtitle('Cart on ground - Old model - Motor A', 'fontweight', 'bold')
print -depsc step_response_g_a.eps

sum(abs(va_gs(1:400) - va_est_gs_A))

% Motor B 
vb_est_gs_B = lsim(sys_2B,voltageB_gs(1:400),t_gs(1:400));
figure
subplot(2,1,1),plot(t_gs(1:400),[vb_gs(1:400) vb_est_gs_B]);
set(gca, 'FontSize', 11)
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t_gs(1:400),abs(vb_gs(1:400) - vb_est_gs_B))
set(gca, 'FontSize', 11)
legend('absolute error')
xlabel('time [s]')
ylabel('|\omega_{b emp} - \omega_{b est}| [rad/s]')
axis tight
sgtitle('Cart on ground - Old model - Motor B', 'fontweight', 'bold')
print -depsc step_response_g_b.eps

sum(abs(vb_gs(1:400) - vb_est_gs_B))

%% 3.b) Identify new model

% 'g' stands for 'ground'

%%%%%%%%%%%%%%%%%%%% No filter

%%%%%%%%%%% Identification

%%% Motor A

% collect the signals appearing in the difference equation
b_gA = va_gb(3:end); 
phi_gA = [-va_gb(2:end-1), voltageA(1:end-2)]; % voltage_gb_A would be the same as voltageA

% perform the fit to get the desired parameters
theta_gA = phi_gA\b_gA;

% build the identified model
Num_gA = [theta_gA(2)];
Den_gA = [1, theta_gA(1), 0];
sys_gA = tf(Num_gA, Den_gA, Ts)

%%% Motor B

% collect the signals appearing in the difference equation
b_gB = vb_gb(3:end); 
phi_gB = [-vb_gb(2:end-1), voltageB(1:end-2)]; 

% perform the fit to get the desired parameters
theta_gB = phi_gB\b_gB;

% build the identified model
Num_gB = [theta_gB(2)];
Den_gB = [1, theta_gB(1), 0];
sys_gB = tf(Num_gB, Den_gB, Ts)

% Bode plots
figure, hold on
bode(sys_gA)
bode(sys_gB)
legend('Motor A', 'Motor B')
hold off

% Pole-Zero map
figure, hold on
pzmap(sys_gA, sys_gB)
set(gca, 'FontSize', 11)
legend('Motor A', 'Motor B')
print -depsc p&z_g.eps

%%%%%%%%%%% Validation
% NF stands for 'no filter'

% Motor A
va_est_gs_NF_A = lsim(sys_gA,voltageA_gs(1:400),t_gs(1:400));
figure
subplot(2,1,1),plot(t_gs(1:400),[va_gs(1:400) va_est_gs_NF_A]);
set(gca, 'FontSize', 11)
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_a [rad/s]')
axis tight
subplot(2,1,2),plot(t_gs(1:400),abs(va_gs(1:400) - va_est_gs_NF_A))
set(gca, 'FontSize', 11)
legend('absolute error')
xlabel('time [s]')
ylabel('|\omega_{a emp} - \omega_{a est}| [rad/s]')
axis tight
sgtitle('Cart on ground - New model - Motor A', 'fontweight', 'bold')
print -depsc step_response_g_NF_a.eps

sum(abs(va_gs(1:400) - va_est_gs_NF_A))

% Motor B 
vb_est_gs_NF_B = lsim(sys_gA,voltageB_gs(1:400),t_gs(1:400));
figure
subplot(2,1,1),plot(t_gs(1:400),[vb_gs(1:400) vb_est_gs_NF_B]);
set(gca, 'FontSize', 11)
legend('empirical','estimated','Location','SouthWest')
xlabel('time [s]')
ylabel('\omega_b [rad/s]')
axis tight
subplot(2,1,2),plot(t_gs(1:400),abs(vb_gs(1:400) - vb_est_gs_NF_B))
set(gca, 'FontSize', 11)
legend('absolute error')
xlabel('time [s]')
ylabel('|\omega_{b emp} - \omega_{b est}| [rad/s]')
axis tight
sgtitle('Cart on ground - New model - Motor B', 'fontweight', 'bold')
print -depsc step_response_g_NF_b.eps

sum(abs(vb_gs(1:400) - vb_est_gs_NF_B))



%% overig
% 
% %%%%%%%%%%%%%%%%%%%% SK filter
% 
% %%%%%%%%%%% Identification
% 
% %%% Motor A
% 
% sys_SK_gA = sys_gA;
% error_SK_gA = [100 100 100];
% Den_SK_gA_j = Den_gA;
% Den_SK_gA_i = 0;
% va_SK_gA = va_gb;
% voltageA_SK_gA = voltageA;
%  
% while abs(error_SK_gA) > [eps eps eps]
%     % apply the filter to both input and output
%     Den_SK_gA_i = Den_SK_gA_j;
%     va_SK_gA = filter(1, Den_SK_gA_j, va_SK_gA);
%     voltageA_SK_gA = filter(1, Den_SK_gA_j, voltageA_SK_gA);
%     
%     % repeat the identification
%     b_SK_gA = va_SK_gA(3:end); 
%     phi_SK_gA = [-va_SK_gA(2:end-1), voltageA_SK_gA(1:end-2)]; 
%     theta_SK_gA = phi_SK_gA\b_SK_gA;
%     Num_SK_gA = [theta_SK_gA(2)];
%     Den_SK_gA_j = [1, theta_SK_gA(1), 0];
%     sys_SK_gA = tf(Num_SK_gA, Den_SK_gA_j, Ts);
%     
%     % error criterion
%     error_SK_gA = Den_SK_gA_j-Den_SK_gA_i;
% end
% 
% 
% %%% Motor B
% 
% sys_SK_gB = sys_gB;
% error_SK_gB = [100 100 100];
% Den_SK_gB_j = Den_gB;
% Den_SK_gB_i = 0;
% vb_SK_gB = vb_gb;
% voltageB_SK_gB = voltageB;
%  
% while abs(error_SK_gB) > [eps eps eps]
%     % apply the filter to both input and output
%     Den_SK_gB_i = Den_SK_gB_j;
%     vb_SK_gB = filter(1, Den_SK_gB_j, vb_SK_gB);
%     voltageB_SK_gB = filter(1, Den_SK_gB_j, voltageB_SK_gB);
%     
%     % repeat the identification
%     b_SK_gB = vb_SK_gB(3:end); 
%     phi_SK_gB = [-vb_SK_gB(2:end-1), voltageB_SK_gB(1:end-2)]; 
%     theta_SK_gB = phi_SK_gB\b_SK_gB;
%     Num_SK_gB = [theta_SK_gB(2)];
%     Den_SK_gB_j = [1, theta_SK_gB(1), 0];
%     sys_SK_gB = tf(Num_SK_gB, Den_SK_gB_j, Ts);
%     
%     % error criterion
%     error_SK_gB = Den_SK_gB_j-Den_SK_gB_i;
% end
% 
% 
% % Bode plots
% figure, hold on
% bode(sys_SK_gA)
% bode(sys_SK_gB)
% legend('Motor A', 'Motor B')
% hold off
% 
% % Pole-Zero map
% figure, hold on
% pzmap(sys_SK_gA, sys_SK_gB)
% set(gca, 'FontSize', 11)
% legend('Motor A', 'Motor B')
% print -depsc p&z_SK_g.eps
% 
% %%%%%%%%%%% Validation
% 
% % Motor A
% va_est_gs_SK_A = lsim(sys_SK_gA,voltageA_gs(1:400),t_gs(1:400));
% figure
% subplot(2,1,1),plot(t_gs(1:400),[va_gs(1:400) va_est_gs_SK_A]);
% set(gca, 'FontSize', 11)
% legend('empirical','estimated','Location','SouthWest')
% xlabel('time [s]')
% ylabel('\omega_a [rad/s]')
% axis tight
% subplot(2,1,2),plot(t_gs(1:400),abs(va_gs(1:400) - va_est_gs_SK_A))
% set(gca, 'FontSize', 11)
% legend('absolute error')
% xlabel('time [s]')
% ylabel('|\omega_{a emp} - \omega_{a est}| [rad/s]')
% axis tight
% sgtitle('Cart on ground - Sanathanan Koerner - Motor A', 'fontweight', 'bold')
% print -depsc step_response_g_SK_a.eps
% 
% sum(abs(va_gs(1:400) - va_est_gs_SK_A))
% 
% % Motor B 
% vb_est_gs_SK_B = lsim(sys_SK_gB,voltageB_gs(1:400),t_gs(1:400));
% figure
% subplot(2,1,1),plot(t_gs(1:400),[vb_gs(1:400) vb_est_gs_SK_B]);
% set(gca, 'FontSize', 11)
% legend('empirical','estimated','Location','SouthWest')
% xlabel('time [s]')
% ylabel('\omega_b [rad/s]')
% axis tight
% subplot(2,1,2),plot(t_gs(1:400),abs(vb_gs(1:400) - vb_est_gs_SK_B))
% set(gca, 'FontSize', 11)
% legend('absolute error')
% xlabel('time [s]')
% ylabel('|\omega_{b emp} - \omega_{b est}| [rad/s]')
% axis tight
% sgtitle('Cart on ground - Sanathanan Koerner - Motor B', 'fontweight', 'bold')
% print -depsc step_response_g_SK_b.eps
% 
% sum(abs(vb_gs(1:400) - vb_est_gs_SK_B))
% 
% 
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%% BW filter
% 
% %%%%%%%%%%% Identification
% 
% %cutoff = bandwidth(sys_g_BW)/(2*pi);
% cutoff = 35;
% [B_gb_BW,A_gb_BW] = butter(6, cutoff*(2/fs));
% h = fvtool(B_gb_BW, A_gb_BW);
% 
% % apply the filter to both input and output
% va_gb_BW = filter(B_gb_BW, A_gb_BW, va_gb); 
% voltageA_gb_BW = filter(B_gb_BW, A_gb_BW, voltageA);
% 
% %repeat the identification
% b_gb_BW = va_gb_BW(3:end); 
% phi_gb_BW = [va_gb_BW(2:end-1), voltageA_gb_BW(1:end-2)]; 
% theta1_gb_BW = phi_gb_BW\b_gb_BW;
% Num_gb_BW = [theta1_gb_BW(2)];
% Den_gb_BW = [1, -theta1_gb_BW(1), 0];
% sys_g_BW = tf(Num_gb_BW, Den_gb_BW, Ts);
% 
% % compute the frequency response of the identified model
% FRF_gb_BW = squeeze(freqresp(sys_g_BW,2*pi*f));
% mag_gb_BW = 20*log10(abs(FRF_gb_BW));
% phs_gb_BW = 180/pi*unwrap(angle(FRF_gb_BW)); 
% phs_gb_BW = 360*ceil(-phs_gb_BW(1)/360) + phs_gb_BW;
% 
% % plot the results
% figure,hold on,
% subplot(2,1,1),semilogx(f, mag_gb_BW)
% grid on
% xlim([f(1) f(end)])
% xlabel('f  [Hz]')
% ylabel('|FRF|  [dB]')
% legend('estimated','Location','SouthWest')
% subplot(2,1,2),semilogx(f, phs_gb_BW)
% grid on
% xlim([f(1) f(end)])
% xlabel('f  [Hz]')
% ylabel('\phi(FRF)  [^\circ]')
% legend('estimated','Location','SouthWest')
% sgtitle('FRF new estimated model for cart on ground with Butterworth filter')
% print -depsc FRF_ground_newmodel_filter.eps
% 
% %%%%%%%%%%% Validation
% 
% % Motor A
% figure
% va_est_gs_BW = lsim(sys_g_BW,voltageA_gs(1:400),t_gs(1:400));
% subplot(2,1,1),plot(t(1:400),[va_gs(1:400) va_est_gs_BW]);
% legend('empirical','estimated','Location','SouthWest')
% xlabel('time [s]')
% ylabel('\omega_a [rad/s]')
% axis tight
% subplot(2,1,2),plot(t(1:400),abs(va_gs(1:400) - va_est_gs_BW))
% legend('absolute error')
% xlabel('time [s]')
% ylabel('\omega_a(empirical) - \omega_a(estimated) [rad/s]')
% axis tight
% sgtitle('Step response new estimated model motor a for cart on ground with Butterworth filter')
% print -depsc step_response_ground__newmodel_filter_a.eps
% 
% figure, hold on
% pzmap(sys_g_BW)
% 
% error_ground_BW = sum(abs(va_gs(1:400) - va_est_gs_BW));

%% Saving variables

save('assignment1_def.mat');