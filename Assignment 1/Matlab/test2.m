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


%% Defining vbriables

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

%% Complex, motor A

%%% Complex
b_1A = va(4:end); 
phi_1A = [-va(3:end-1), -va(2:end-2), voltageA(2:end-2), voltageA(1:end-3)]; 

% perform the fit to get the desired parameters
theta_1A = phi_1A\b_1A;

% build the identified model
Num_1A = [0, theta_1A(3), theta_1A(4)];
Den_1A = [1, theta_1A(1), theta_1A(2), 0];
sys_1A = tf(Num_1A, Den_1A, Ts);


%%% Complex, BW
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
sys_BW_1A = tf(Num_BW_1A, Den_BW_1A, Ts);

va_est_BW_1A = lsim(sys_BW_1A,voltageA_step(1:400),t(1:400)); 
error_BW_1A_tot = sum(abs(va_step(1:400) - va_est_BW_1A))

va_est_BW_1A_sq = lsim(sys_BW_1A,voltageA(1:400),t(1:400));  
error_BW_1A_tot_sq = sum(abs(va(1:400) - va_est_BW_1A_sq))

%% Complex, motor B

%%% Complex
b_1B = vb(4:end); 
phi_1B = [-vb(3:end-1), -vb(2:end-2), voltageB(2:end-2), voltageB(1:end-3)]; 

% perform the fit to get the desired parameters
theta_1B = phi_1B\b_1B;

% build the identified model
Num_1B = [0, theta_1B(3), theta_1B(4)];
Den_1B = [1, theta_1B(1), theta_1B(2), 0];
sys_1B = tf(Num_1B, Den_1B, Ts);


%%% Complex, BW
% define a low(band)-pass filter
cutoff_1B = 3.5*bandwidth(sys_1B)/(2*pi);
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
sys_BW_1B = tf(Num_BW_1B, Den_BW_1B, Ts);

vb_est_BW_1B = lsim(sys_BW_1B,voltageB_step(1:400),t(1:400)); 
error_BW_1B_tot = sum(abs(vb_step(1:400) - vb_est_BW_1B))

vb_est_BW_1B_sq = lsim(sys_BW_1B,voltageB(1:400),t(1:400));  
error_BW_1A_tot_sq = sum(abs(vb(1:400) - vb_est_BW_1B_sq))

%% Simple, motor A

%%% Simple
% collect the signals appearing in the difference equation
b_2A = va(3:end); 
phi_2A = [-va(2:end-1), voltageA(1:end-2)]; 

% perform the fit to get the desired parameters
theta_2A = phi_2A\b_2A;

% build the identified model
Num_2A = [theta_2A(2)];
Den_2A = [1, theta_2A(1), 0];
sys_2A = tf(Num_2A, Den_2A, Ts);


%%% Simple, BW
% define a low(band)-pass filter
cutoff_2A = 0.009*bandwidth(sys_2A)/(2*pi);
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
sys_BW_2A = tf(Num_BW_2A, Den_BW_2A, Ts);

va_est_BW_2A = lsim(sys_BW_2A,voltageA_step(1:400),t(1:400)); 
error_BW_2A_tot = sum(abs(va_step(1:400) - va_est_BW_2A))

va_est_BW_2A_sq = lsim(sys_BW_2A,voltageA(1:400),t(1:400)); 
error_BW_2A_tot_sq = sum(abs(va(1:400) - va_est_BW_2A_sq))

%% Simple, motor B

%%% Simple
% collect the signals appearing in the difference equation
b_2B = vb(3:end); 
phi_2B = [-vb(2:end-1), voltageB(1:end-2)]; 

% perform the fit to get the desired parameters
theta_2B = phi_2B\b_2B;

% build the identified model
Num_2B = [theta_2B(2)];
Den_2B = [1, theta_2B(1), 0];
sys_2B = tf(Num_2B, Den_2B, Ts);


%%% Simple, BW
% define a low(band)-pass filter
cutoff_2B = 0.0098*bandwidth(sys_2B)/(2*pi);
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
sys_BW_2B = tf(Num_BW_2B, Den_BW_2B, Ts);

vb_est_BW_2B = lsim(sys_BW_2B,voltageB_step(1:400),t(1:400)); 
error_BW_2B_tot = sum(abs(vb_step(1:400) - vb_est_BW_2B))

vb_est_BW_2B_sq = lsim(sys_BW_2B,voltageB(1:400),t(1:400)); 
error_BW_2B_tot_sq = sum(abs(vb_step(1:400) - vb_est_BW_2B_sq))
