%% Data
% csvfile = '../Data/pendulumangle.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% pendulumangle = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save pendulumangle.mat
load pendulumangle.mat
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
% csvfile = '../Data/FBFFrho8.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% FBFFrho8 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save FBFFrho8.mat
load FBFFrho8.mat
% csvfile = '../Data/FBFFrho7.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% FBFFrho7 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save FBFFrho7.mat
load FBFFrho7.mat
% csvfile = '../Data/FBFFrho6.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% FBFFrho6 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save FBFFrho6.mat
load FBFFrho6.mat
% csvfile = '../Data/FBFFrho5.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% FBFFrho5 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save FBFFrho5.mat
load FBFFrho5.mat
% csvfile = '../Data/FBFFrho4.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% FBFFrho4 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save FBFFrho4.mat
load FBFFrho4.mat
% csvfile = '../Data/FBFFrho3.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% FBFFrho3 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save FBFFrho3.mat
load FBFFrho3.mat
% csvfile = '../Data/FBFFrho2.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% FBFFrho2 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save FBFFrho2.mat
load FBFFrho2.mat

time = pendulumangle(:,5)-pendulumangle(1,5);
angle = pendulumangle(:,9); % in degrees radians

figure
plot(time, angle*180/pi)
xlabel('Time [s]')
ylabel('Pendulum angle [?]')
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
rho =[1 2 3 4 5 6 7 8 10];
Kdlqr = zeros(length(rho),3);
Ts = 0.01;
A = [1 0 0;
     0 1 Ts/L2;
     0 -9.81*Ts 1];
 B = [Ts -Ts/L2 0]';
 C = [1 0 0;
    0 1 0];
z = 1;
Np = zeros(length(rho),1);
for i = 1:length(rho)
    Qlqr = rho(i)*Cz'*Cz; 
    Rlqr = 1;
    Kdlqr(i,:) = dlqr(A,B,Qlqr,Rlqr);
    H = C*inv((z*eye(3)-A+B*Kdlqr(i)))*B;
    Np(i) = 1/(H(1)+L2*H(2));
end











load FBFFrho10.mat

t = 0:0.01:6.38;
x1 = FBFFrho1(1:639,12);
theta1 = FBFFrho1(1:639,13);
v1 = FBFFrho1(1:639,3);
desiredVel1 = FBFFrho1(1:639,2);
uA1 = FBFFrho1(1:639,4);

x10 = FBFFrho10(1:639,12);
theta10 = FBFFrho10(1:639,13);
v10 = FBFFrho10(1:639,3);
desiredVel10 = FBFFrho10(1:639,2);
uA10 = FBFFrho10(1:639,4);

x100 = FBFFrho100(1:639,12);
theta100 = FBFFrho100(1:639,13);
v100 = FBFFrho100(1:639,3);
desiredVel100 = FBFFrho100(1:639,2);
uA100 = FBFFrho100(1:639,4);

x8 = FBFFrho8(1:639,12);
theta8 = FBFFrho8(1:639,13);
v8 = FBFFrho8(1:639,3);
desiredVel8 = FBFFrho8(1:639,2);
uA8 = FBFFrho8(1:639,4);

x7 = FBFFrho7(1:639,12);
theta7 = FBFFrho7(1:639,13);
v7 = FBFFrho7(1:639,3);
desiredVel7 = FBFFrho7(1:639,2);
uA7 = FBFFrho7(1:639,4);

x6 = FBFFrho6(1:639,12);
theta6 = FBFFrho6(1:639,13);
v6= FBFFrho6(1:639,3);
desiredVel6 = FBFFrho6(1:639,2);
uA6 = FBFFrho6(1:639,4);

x5 = FBFFrho5(1:639,12);
theta5 = FBFFrho5(1:639,13);
v5= FBFFrho5(1:639,3);
desiredVel5 = FBFFrho5(1:639,2);
uA5 = FBFFrho5(1:639,4);

x4 = FBFFrho4(1:639,12);
theta4 = FBFFrho4(1:639,13);
v4= FBFFrho4(1:639,3);
desiredVel4 = FBFFrho4(1:639,2);
uA4 = FBFFrho4(1:639,4);

x3 = FBFFrho3(1:639,12);
theta3 = FBFFrho3(1:639,13);
v3= FBFFrho3(1:639,3);
desiredVel3 = FBFFrho3(1:639,2);
uA3 = FBFFrho3(1:639,4);

x2 = FBFFrho2(1:639,12);
theta2 = FBFFrho2(1:639,13);
v2= FBFFrho2(1:639,3);
desiredVel2 = FBFFrho2(1:639,2);
uA2 = FBFFrho2(1:639,4);




figure
hold on
plot(t, [x1+L2*theta1 x8+L2*theta8 x10+L2*theta10 x100+L2*theta100])
legend('rho = 1','rho = 8','rho = 10','rho = 100', 'Location', 'SouthEast')
ylabel('Pendulum mass position [m]')
xlabel('Time [s]')
hold off

figure
hold on
plot(t, [x5+L2*theta5 x6+L2*theta6 x7+L2*theta7 x8+L2*theta8])
legend('rho = 5','rho = 6','rho = 7','rho = 8', 'Location', 'SouthEast')
ylabel('Pendulum mass position [m]')
xlabel('Time [s]')
hold off

figure
hold on
plot(t, [x3+L2*theta3 x4+L2*theta4 x5+L2*theta5 x7+L2*theta7])
legend('rho = 3','rho = 4','rho = 5','rho 7', 'Location', 'SouthEast')
ylabel('Pendulum mass position [m]')
xlabel('Time [s]')
hold off

figure
hold on
plot(t, [x1+L2*theta1 x2+L2*theta2 x3+L2*theta3 x7+L2*theta7])
legend('rho = 1','rho = 2','rho = 3','rho 7', 'Location', 'SouthEast')
ylabel('Pendulum mass position [m]')
xlabel('Time [s]')
hold off


% Voltage plots
figure
hold on
plot(t, [v1 v10 v100])
legend('rho = 1','rho = 10','rho = 100', 'Location', 'NorthEast')
ylabel('Voltage applied on motor A [V]')
xlabel('Time [s]')
hold off



figure
hold on
plot(t, [v5 v6 v7 v8])
legend('rho = 5','rho = 6', 'rho = 7', 'rho = 8', 'Location', 'NorthEast')
ylabel('Voltage applied on motor A [V]')
xlabel('Time [s]')
hold off

figure
hold on
plot(t, [v3 v4 v5 v10])
legend('rho = 3', 'rho = 4', 'rho = 5','rho 10', 'Location', 'NorthEast')
ylabel('Voltage applied on motor A [V]')
xlabel('Time [s]')
hold off

% Plots voltage controller wants to apply

figure
hold on
plot(t, [uA5 uA6 uA7 uA8])
legend('rho = 5','rho = 6', 'rho = 7', 'rho = 8', 'Location', 'NorthEast')
ylabel('Voltage controller wants to apply on motor A [V]')
xlabel('Time [s]')
hold off

figure
hold on
plot(t, [uA1 uA2 uA3 uA4])
legend('rho = 1','rho = 2', 'rho = 3', 'rho = 4', 'Location', 'NorthEast')
ylabel('Voltage controller wants to apply on motor A [V]')
xlabel('Time [s]')
hold off

% Plots desiredvelocity
figure
hold on
plot(t, [desiredVel5 desiredVel6 desiredVel7 desiredVel8])
legend('rho = 5','rho = 6', 'rho = 7', 'rho = 8', 'Location', 'NorthEast')
ylabel('Desired velocity [m/s]')
xlabel('Time [s]')
hold off

figure
hold on
plot(t, [desiredVel1 desiredVel2 desiredVel3 desiredVel4])
legend('rho = 1','rho = 2', 'rho = 3', 'rho = 4', 'Location', 'NorthEast')
ylabel('Desired velocity [m/s]')
xlabel('Time [s]')
hold off
    