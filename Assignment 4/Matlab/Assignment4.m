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
ylabel('Pendulum angle [°]')
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






%% 2

% with wheel encoder
R_x = 2.7878e-9;
R_theta = 9.4124e-6;

R = [R_x, 0; 0, R_theta];
obj = KalmanExperiment.createfromQRC45();

%%% state 1
figure()
hold on 
plotstates(obj, 1, 0.95)
plotmeasurements(obj, 1, 0.95)
xlabel('time [s]')
ylabel('x [m]')
hold off
print -depsc state1.eps

figure()
hold on 
plotstates(obj, 1, 0.95)
plotmeasurements(obj, 1, 0.95)
xlabel('time [s]')
ylabel('x [m]')
xlim([31.82 31.85])
ylim([0.1947 0.19485])
hold off
print -depsc state1_closeup.eps

%%% state 2
figure()
hold on 
plotstates(obj, 2, 0.95)
plotmeasurements(obj, 2, 0.95)
xlabel('time [s]')
ylabel('\theta [rad]')
ylim([-0.6 0.6])
hold off
print -depsc state2.eps

figure()
hold on 
plotstates(obj, 2, 0.95)
plotmeasurements(obj, 2, 0.95)
xlabel('time [s]')
ylabel('\theta [rad]')
xlim([35.5 37])
ylim([-0.12 0.12])
hold off
print -depsc state2_closeup.eps

%%% state 3
figure()
hold on 
plotstates(obj, 3, 0.95)
xlabel('time [s]')
ylabel('v_{tan} [m/s]')
hold off
print -depsc state3.eps




% without wheel encoder
R_theta = 9.4124e-6;

R = [R_theta, 0; 0, 0];
obj = KalmanExperiment.createfromQRC45();

%%% state 1
figure()
hold on 
plotstates(obj, 2, 0.95)
legend('state 1 (95% interval)')
xlabel('time [s]')
ylabel('x [m]')
hold off
print -depsc state1_w.eps

figure()
hold on 
plotstates(obj, 2, 0.95)
legend('state 1 (95% interval)')
xlabel('time [s]')
ylabel('x [m]')
xlim([131 138])
ylim([-0.03 0.03])
hold off
print -depsc state1_w_closeup.eps

%%% state 2
figure()
hold on 
plotstates(obj, 1, 0.95)
plotmeasurements(obj, 1, 0.95)
legend('state 2 (95% interval)','measurement 2 (95% interval)')
xlabel('time [s]')
ylabel('\theta [rad]')
ylim([-0.6 0.6])
hold off
print -depsc state2_w.eps

figure()
hold on 
plotstates(obj, 1, 0.95)
plotmeasurements(obj, 1, 0.95)
legend('state 2 (95% interval)','measurement 2 (95% interval)')
xlabel('time [s]')
ylabel('\theta [rad]')
xlim([133.7 135.1])
ylim([-0.15 0.15])
hold off
print -depsc state2_w_closeup.eps

%%% state 3
figure()
hold on 
plotstates(obj, 3, 0.95)
xlabel('time [s]')
ylabel('v_{tan} [m/s]')
hold off
print -depsc state3_w.eps












    