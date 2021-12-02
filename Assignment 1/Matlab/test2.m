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

%% Yeet

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
cutoff = 0.23*bandwidth(sys_1A)/(2*pi);
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
sys_BW_1A = tf(Num_BW_1A, Den_BW_1A, Ts);




%%% Complex, SK
% Denumerator of complex model without filter: Den1
sys_SK1 = sys_1A;
error_SK1 = [100 100 100 100];
Den_SK1_2 = Den_1A;
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




%%% Difference between response of the simulated model and the emperical values: NO FILTER, COMPLEX MODEL 
va_est_1A = lsim(sys_1A,voltageA_step(106:505),t_step(106:505));               % start at 106, moment of step
error_1A_tot= sum(abs(va_step(106:505) - va_est_1A))

%%%Difference between response of the simulated model and the real system: COMPLEX MODEL, BUTTERWORTH FILTER
va_est_BW_1A = lsim(sys_BW_1A, voltageA_step(106:505), t_step(106:505));
error_BW1_tot= sum(abs(va_step(106:505) - va_est_BW_1A))

%%%Difference between response of the simulated model and the real system: COMPLEX MODEL, SK FILTER
va_est_SK1 = lsim(sys_SK1, voltageA_step(106:505), t_step(106:505));
error_SK1_tot= sum(abs(va_step(106:505) - va_est_SK1))



