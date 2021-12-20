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



    