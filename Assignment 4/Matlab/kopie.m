%% Data
% csvfile = '../Data/pendulumangle.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% pendulumangle = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save pendulumangle.mat
load pendulumangle.mat

time = pendulumangle(:,5)-pendulumangle(1,5);
angle = pendulumangle(:,9); % in degrees radians

figure
plot(time, angle*180/pi)
xlabel('Time [s]')
ylabel('Pendulum angle [�]')
A = zeros(12,1); % Amplitudes
indices = zeros(12); % Indices of maxima
timeMax = zeros(12,1); % Time instance of maxima
That = 71;
sum = 0;
for i = 1:12
    A(i) = max(angle(40+(i-1)*That:40+i*That));
    x =  find(angle(40+(i-1)*That:40+i*That) == A(i));
    indices(i,1:length(x)) = x+39+(i-1)*That;
    timeMax(i) = time(indices(i,1));
end

Adegrees = A*180/pi;

% Calculating the period of the oscilator by calculating the mean of the
% timedifferences between the maxima
for i = 2:length(timeMax)
    sum = sum + (timeMax(i) - timeMax(i-1));
    T = sum/(length(timeMax)-1);
end

% Calculating zeta
wd = 2*pi/T; % damped frequency
delta = log(A(1)/A(2)); % logarithmic increment
zeta = delta/(sqrt((2*pi)^2+delta^2));
wn = wd/(sqrt(1-zeta^2));
L = 9.81/wn^2;
L2 = 9.81/wd^2;






% %% 2
% 
% % with wheel encoder
% R_x = 2.7878e-9;
% R_theta = 9.4124e-6;
% 
% R = [R_x, 0; 0, R_theta];
% obj = KalmanExperiment.createfromQRC45();
% 
% %%% state 1
% figure()
% hold on 
% plotstates(obj, 1, 0.95)
% plotmeasurements(obj, 1, 0.95)
% hold off
% 
% %%% state 2
% figure()
% hold on 
% plotstates(obj, 2, 0.95)
% plotmeasurements(obj, 2, 0.95)
% ylim([-0.6 0.6])
% hold off
% 
% %%% state 3
% figure()
% hold on 
% plotstates(obj, 3, 0.95)
% hold off
% 
% 
% 
% 
% % without wheel encoder




%% 4 Feedback and feedforward controller
Cz = [1 L2 0];
rho = 1e1;
Qlqr = rho*Cz'*Cz; 
Rlqr = 1;
Ts = 0.01;


A = [1 0 0;
     0 1 Ts/L2;
     0 -9.81*Ts 1];
 B = [Ts -Ts/L2 0]';
Kdlqr = dlqr(A,B,Qlqr,Rlqr)

C = [1 0 0;
    0 1 0];
z = 1;
H = C*inv((z*eye(3)-A+B*Kdlqr))*B;


Np = 1/(H(1)+L*H(2))

% csvfile = '../Data/FBFFrho100.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% FBFFrho100 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save FBFFrho100.mat
load FBFFrho100.mat
% csvfile = '../Data/FBFFrho10.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% FBFFrho10 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save FBFFrho10.mat
load FBFFrho10.mat
% csvfile = '../Data/FBFFrho1.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% FBFFrho1 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save FBFFrho1.mat
load FBFFrho1.mat

t = 0:0.01:6.72;
x1 = FBFFrho1(1:673,12);
theta1 = FBFFrho1(1:673,13);
v1 = FBFFrho1(1:673,3);
x10 = FBFFrho10(1:673,12)-x10(1);
theta10 = FBFFrho10(1:673,13)-theta10(1);
v10 = FBFFrho10(1:673,3);
x100 = FBFFrho100(1:673,12);
theta100 = FBFFrho100(1:673,13);
v100 = FBFFrho100(1:673,3);

figure
hold on
plot(t, [x1+L2*theta1 x10+L2*theta10 x100+L2*theta100])
legend('rho = 1','rho = 10','rho = 100')
ylabel('Pendulum mass position [m]')
xlabel('Time [s]')
hold off

figure
hold on
plot(t, [v1 v10 v100])
legend('rho = 1','rho = 10','rho = 100')
ylabel('Voltage applied on motor A [V]')
xlabel('Time [s]')
hold off

    