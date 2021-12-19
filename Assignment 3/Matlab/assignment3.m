%% 1
close all
clear

% Discrete-time state space model
Ts = 0.01;

A = 1;
B = Ts;
C = -1;
D = 0;

sys = ss(A,B,C,D,Ts);

% Closed-loop state space model
% Pole-zero map for varying k

figure 
hold on 
% poles
K = 0:10:200;
poles = 1-Ts*K;

color = jet(21);
for i = 1:length(poles)
    scatter(poles(i), 0, 50, color(i,:), 'x')
end
h = legend({'0','10','20','30','40','50','60','70','80','90','100','110','120','130','140','150','160','170','180','190','200'},'AutoUpdate','off', 'Fontsize', 8.5);
% unit circle
theta = linspace(0,2*pi,300); 
plot(cos(theta), sin(theta), 'k', 'LineWidth', 0.5)
% y axis
xline(0);
% markup
xlim([-1 1.7])
ylim([-1.35 1.35])
axis square
sgtitle('Poles of the closed loop system', 'fontweight', 'bold')
xlabel('Real Axis')
ylabel('Imaginary Axis')
print -depsc poles.eps

poles_c = log(poles)/Ts;
poles_d2 = exp(poles_c*Ts);
figure
hold on
for i = 1:length(poles)
    scatter(real(poles_c(i)), imag(poles_c(i)), 50, color(i,:), 'x')
end
h = legend({'0','10','20','30','40','50','60','70','80','90','100','110','120','130','140','150','160','170','180','190','200'},'Location', 'East','AutoUpdate','off', 'Fontsize', 6.5);
sgtitle('Poles of the closed loop system in continuous time', 'fontweight', 'bold')
xlabel('Real Axis')
ylabel('Imaginary Axis')
xlim([-300 50])
ylim([-50 350])
print -depsc poles_cont.eps

% response speed depending on K
figure
hold on
for K = 10:10:50
    sys = ss(A-B*K,B,C-D*K,D,Ts);
    impulse(sys)
end
set(gca, 'Fontsize', 12)
xlabel('Time', 'FontSize', 12)
ylabel('Amplitude', 'FontSize', 12)
ylim([-1.2 0.2])
legend({'10','20','30','40','50'}, 'Location','SouthEast')
hold off
print -depsc impulse_response1.eps

figure
hold on
for K = [0, 100]
    sys = ss(A-B*K,B,C-D*K,D,Ts);
    impulse(sys)
end
set(gca, 'Fontsize', 12)
xlabel('Time', 'FontSize', 12)
ylabel('Amplitude', 'FontSize', 12)
ylim([-1.2 0.2])
legend({'0', '100'}, 'Location','SouthEast')
print -depsc impulse_response3.eps

%% 2
% LQE estimator gain
Q = 0.1;
R = 0.01;
L1 = dlqr(A', A'*C', Q, R)';

% steady state Kalmain gain using derived formula
P = (1/2)*(-Q+ sqrt(Q^2+4*R*Q));
L2 = -(P+Q)/(P+Q+R);

% Closed loop poles of the LQE
figure 
hold on 
% poles
ratio = 0:0.5:10;
poles = 2./(ratio + 2 + sqrt(ratio.^2 + 4*ratio));
color = jet(21);
for i = 1:length(poles)
    scatter(poles(i), 0, 50, color(i,:), 'x')
end
h = legend({'0','0.5','1','1.5','2.5','3','3.5','4','4.5','5','5.5','6','6.5','7','7.5','8','8.5','9','9.5','10'},'AutoUpdate','off', 'Fontsize', 8.5);
% unit circle
theta = linspace(0,2*pi,300); 
plot(cos(theta), sin(theta), 'k', 'LineWidth', 0.5)
% y axis
xline(0);
% markup
xlim([-1 1.7])
ylim([-1.35 1.35])
axis square
sgtitle('Closed loop poles of the LQE', 'fontweight', 'bold')
xlabel('Real Axis')
ylabel('Imaginary Axis')
print -depsc poles_LQE.eps



%% 3

%%% 3a) determine the measurement noise covariance 

% csvfile = '../Data/FrontDistance.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% FrontDistance = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save FrontDistance

t = 0:0.01:10.63;
load FrontDistance.mat
x = FrontDistance(:,2);
figure
plot(t,x)
xlabel('time [s]')
ylabel('Distance [m]')

R = cov(x);
Q = 10*R;
P00 = (0.015/3)^2;

%%% 3b)

% loeading measurements
load K1rho10.mat
load K2rho10.mat
load K3rho10.mat
load K4rho10.mat

% plots
t = K1rho10(:, 5);
x1 = -K1rho10(:, 9);
x2 = -K2rho10(:, 9);
x3 = -K3rho10(:, 9);
x4 = -K4rho10(:, 9);

figure
hold on
plot(t, [x1, x2, x3, x4]);
xlabel('time [s]')
ylabel('measured distance [m]')
legend('K = 1','K = 2','K = 3','K = 4', 'Location', 'NorthEast')
sgtitle('Measured response for variable K')

voltageA1 = K1rho10(:, 2);
voltageA2 = K2rho10(:, 2);
voltageA3 = K3rho10(:, 3);
voltageA4 = K4rho10(:, 2);

figure
hold on
plot(t, [voltageA1, voltageA2, voltageA3, voltageA4]);
yline(0);
xlabel('time [s]')
ylabel('voltage [V]')
legend('K = 1','K = 2','K = 3','K = 4', 'Location', 'NorthEast')
sgtitle('Control signal for variable K')

%%% 3c)

load K2_4rho1000.mat
load K2_4rho100.mat
load K2_4rho10.mat
load K2_4rho1.mat
load K2_4rho0_1.mat
load K2_4rho0_01.mat
load K2_4rho0_001.mat

t = K2_4rho1000(:, 5);
x1 = -K2_4rho1000(:, 9);
x2 = -K2_4rho100(:, 9);
x3 = -K2_4rho10(:, 9);
x4 = -K2_4rho1(:, 9);
x5 = -K2_4rho0_1(:, 9);
x6 = -K2_4rho0_01(:, 9);
x7 = -K2_4rho0_001(:, 9);

figure
hold on
plot(t, [x1, x2, x3, x4, x5, x6, x7]);
xlabel('time [s]')
ylabel('measured distance [m]')
legend('\rho = 1000', '\rho = 100','\rho = 10','\rho = 1','\rho = 0.1','\rho = 0.01','\rho = 0.001','Location', 'NorthEast')
sgtitle('Measured response for variable \rho')
%%%%%%%%%%%%%%%%%%%%%
figure
hold on
plot(t, [x1, x2]);
xlabel('time [s]')
ylabel('measured distance [m]')
legend('\rho = 1000', '\rho = 100', 'Location', 'NorthEast')
sgtitle('Measured response for variable \rho')

figure
hold on
plot(t, [x3, x4]);
xlabel('time [s]')
ylabel('measured distance [m]')
legend('\rho = 10', '\rho = 1', 'Location', 'NorthEast')
sgtitle('Measured response for variable \rho')

figure
hold on
plot(t, [x5, x6]);
xlabel('time [s]')
ylabel('measured distance [m]')
legend('\rho = 0.1', '\rho = 0.01', 'Location', 'NorthEast')
sgtitle('Measured response for variable \rho')

figure
hold on
plot(t, x7);
xlabel('time [s]')
ylabel('measured distance [m]')
legend('\rho = 0.001', 'Location', 'NorthEast')
sgtitle('Measured response for variable \rho')
%%%%%%%%%%%%%%%%%%%%
% Voor de foute xhat(0) zijn volgende waarden gekozen:
    % xhat(0) = -0.05 terwijl het wagentje rijdt van -0.25 naar -0.15
% wagentje zou dus eerst even naar achter moeten rijden.
    
%% State estimator using pole placement
K = 73;

pd = 1-0.033*Ts*K;
pc = log(pd)/Ts;

% Choose estimater poles 10 times slower in continuous time
pce = pc/10;
pde = exp(pce*Ts);

%estimator gain
Lplace = place(A',C',pde)

% Weer zelfde stap met verkeerde xhat(0) = -0.05, zeer duidelijk sichtbaar
% dat door trage estimator echt de verkeerde kant wordt opgegaan in het begin, zichtbaar
% tijdens metingen!!!
% Bestand met naam pole_placement
% duurt zeer lang, dus mss grafiek met transiente respons en volledige
% respons?

 sgt
