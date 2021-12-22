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
hold off

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
hold off

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
hold off

%% 2

%%% 2d)
% error between LQE estimator gain and steady state Kalmain gain using derived formula
P = @(Q,R) (1/2)*(-Q+ sqrt(Q^2+4*R*Q));

Q = linspace(10^(-4), 1, 100);
R = linspace(10^(-4), 1, 100);
error = zeros(length(Q), length(R));
for i = 1:length(Q)
    for j = 1:length(R)
    P_ij = P(Q(i),R(j));
    Q_i = Q(i);
    R_j = R(j);
    error(i,j) = dlqr(A', A'*C', Q_i, R_j)' - (-(P_ij+Q_i)/(P_ij+Q_i+R_j));
    end
end
figure
surf(Q,R,error);
xlabel('Q [m^2]')
ylabel('R [m^2]')
zlabel('error [-]')
print -depsc gain_error.eps

%%% 2e)
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
hold off


%% 3

%%% 3a) determine the measurement noise covariance 
% 

% csvfile = '../Data2/K2.4rho0.001.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% K2_4rho0_001 = dlmread(csvfile, ',', 2, 0); % Data follows the labels
%  
% save K2_4rho0_001.mat


t = 0:0.01:10.62;
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
t = K1rho10(1:351, 5);
reference = zeros(size(t));
reference(t<=0) = -0.25;
reference(t>0) = -0.1;
x1 = -K1rho10(1:351, 9);
x2 = -K2rho10(1:351, 9);
x3 = -K3rho10(1:351, 9);
x4 = -K4rho10(1:351, 9);


figure
hold on
plot(t, [x1, x2, x3, x4]);
plot(t, reference, 'LineWidth',2)
set(gca, 'FontSize', 11)
xlabel('time [s]')
ylabel('measured distance [m]')
legend({'K = 1','K = 2','K = 3','K = 4', 'reference'}, 'Location', 'SouthEast')
sgtitle('Measured response for variable K')
print -depsc response_variable_K.eps
hold off

t = K1rho10(1:201, 5);
voltageA1 = K1rho10(1:201, 2);
voltageA2 = K2rho10(1:201, 2);
voltageA3 = K3rho10(1:201, 2);
voltageA4 = K4rho10(1:201, 2);

figure
hold on
plot(t, [voltageA1, voltageA2, voltageA3, voltageA4]);
plot(t, zeros(size(t)), 'LineWidth',2)
set(gca, 'FontSize', 11)
xlabel('time [s]')
ylabel('voltage [V]')
legend({'K = 1','K = 2','K = 3', 'K = 4'}, 'Location', 'NorthEast')
sgtitle('Control signal for variable K')
print -depsc voltage_variable_K.eps
hold off


%%% 3c)

% loading data
load K2_4rho1000.mat
load K2_4rho100.mat
load K2_4rho10.mat
load K2_4rho1.mat
load K2_4rho0_1.mat
load K2_4rho0_01.mat
load K2_4rho0_001.mat

% plots for phat
t = K2_4rho1000(:, 5);
phat1 = K2_4rho1000(:, 11);
phat2 = K2_4rho100(:, 11);
phat3 = K2_4rho10(:, 11);
phat4 = K2_4rho1(:, 11);
phat5 = K2_4rho0_1(:, 11);
phat6 = K2_4rho0_01(:, 11);
phat7 = K2_4rho0_001(:, 11);

phatk = [phat1 phat2 phat3 phat4 phat5 phat6 phat7];

figure
hold on
plot(t, phatk, 'LineWidth', 1.5);
set(gca, 'FontSize', 11)
xlabel('$time \: [s]$','Interpreter','latex')
ylabel('$\hat{P}_{k|k} \: [m^2]$','Interpreter','latex')
legend({'\rho = 1000', '\rho = 100','\rho = 10','\rho = 1','\rho = 0.1','\rho = 0.01','\rho = 0.001'},'Location', 'east')
title('$\bf{State \: estimate \: covariance \: \hat{P}_{k|k} \: for \: variable \: \rho}$','Interpreter','latex', 'FontSize', 12)
print -depsc phat.eps
hold off

% error on P
P = @(Q,R) (1/2)*(-Q+ sqrt(Q^2+4*R*Q));
rho = [1000 100 10 1 0.1 0.01 0.001];
error_P = zeros(length(rho), 1);
for i = 1:length(rho)
    phatk_i = phatk(:,i);
    error_P(i) = phatk_i(100) - P(rho(i)*8.4588e-06, 8.4588e-06);
end

figure
hold on
scatter(rho, error_P, 70, lines(7),'x')
yline(0)
set(gca, 'xscale','log', 'fontsize', 11)
xlabel('$\rho \: [-]$', 'interpreter','latex')
ylabel('$error \: [-]$', 'interpreter','latex')
title('$\bf{Error \: on \: \hat{P}_{\infty} \: for \: variable \: \rho}$','Interpreter','latex', 'FontSize', 12)
hold off
print -depsc error_P.eps

% plots for L
L = @(phat, rho) -(phat + 8.4588e-06*rho)/(phat + 8.4588e-06*rho + 8.4588e-06);

L1 = L(phat1, 1000);
L2 = L(phat2, 100);
L3 = L(phat3, 10);
L4 = L(phat4, 1);
L5 = L(phat5, 0.1);
L6 = L(phat6, 0.01);
L7 = L(phat7, 0.001);

Lk = [L1(:,1), L2(:,1), L3(:,1), L4(:,1), L5(:,1), L6(:,1), L7(:,1)];

figure
hold on
plot(t, Lk, 'LineWidth', 1.5);
set(gca, 'FontSize', 11)
ylim([-1.1 0.1])
xlabel('$time \: [s]$','Interpreter','latex')
ylabel('$L_k \: [-]$','Interpreter','latex')
legend({'\rho = 1000', '\rho = 100','\rho = 10','\rho = 1','\rho = 0.1','\rho = 0.01','\rho = 0.001'},'Location', 'east')
title('$\bf{Kalman \: gain \: L_k \: for \: variable \: \rho}$','Interpreter','latex', 'FontSize', 12)
print -depsc L.eps
hold off

% error on L
rho = [1000 100 10 1 0.1 0.01 0.001];
error_L = zeros(length(rho), 1);
for i = 1:length(rho)
    Lk_i = Lk(:,i);
    error_L(i) = Lk_i(100) - dlqr(1, -1, rho(i)*8.4588e-06, 8.4588e-06);
end

figure
hold on
scatter(rho, error_L, 70, lines(7),'x')
yline(0)
set(gca, 'xscale','log', 'fontsize', 11)
xlabel('$\rho \: [-]$', 'interpreter','latex')
ylabel('$error \: [-]$', 'interpreter','latex')
title('$\bf{Error \: on \: L_{\infty} \: for \: variable \: \rho}$','Interpreter','latex', 'FontSize', 12)
hold off
print -depsc error_L.eps

%%% 3d)
R = 8.4588e-06;
ke_rho1000 = KalmanExperiment.createfromQRC3();
ke_rho100 = KalmanExperiment.createfromQRC3();
ke_rho10 = KalmanExperiment.createfromQRC3();
ke_rho1 = KalmanExperiment.createfromQRC3();
ke_rho0_1 = KalmanExperiment.createfromQRC3();
ke_rho0_01 = KalmanExperiment.createfromQRC3();
ke_rho0_001 = KalmanExperiment.createfromQRC3();

ke_rho1000.analyzeconsistency();
sgtitle('$\bf{Q = 8.4588 \cdot 10^{-3} m^2 \: || \: R = 8.4588 \cdot 10^{-6} m^2}$','Interpreter','latex', 'FontSize', 15)
print -depsc SNIS_1000.eps

ke_rho100.analyzeconsistency();
sgtitle('$\bf{Q = 8.4588 \cdot 10^{-4} m^2 \: || \: R = 8.4588 \cdot 10^{-6} m^2}$','Interpreter','latex', 'FontSize', 15)
print -depsc SNIS_100.eps

ke_rho10.analyzeconsistency();
sgtitle('$\bf{Q = 8.4588 \cdot 10^{-5} m^2 \: || \: R = 8.4588 \cdot 10^{-6} m^2}$','Interpreter','latex', 'FontSize', 15)
print -depsc SNIS_10.eps

ke_rho1.analyzeconsistency();
sgtitle('$\bf{Q = 8.4588 \cdot 10^{-6} m^2 \: || \: R = 8.4588 \cdot 10^{-6} m^2}$','Interpreter','latex', 'FontSize', 15)
print -depsc SNIS_1.eps

ke_rho0_1.analyzeconsistency();
sgtitle('$\bf{Q = 8.4588 \cdot 10^{-7} m^2 \: || \: R = 8.4588 \cdot 10^{-6} m^2}$','Interpreter','latex', 'FontSize', 15)
print -depsc SNIS_01.eps

ke_rho0_01.analyzeconsistency();
sgtitle('$\bf{Q = 8.4588 \cdot 10^{-8} m^2 \: || \: R = 8.4588 \cdot 10^{-6} m^2}$','Interpreter','latex', 'FontSize', 15)
print -depsc SNIS_001.eps

ke_rho0_001.analyzeconsistency();
sgtitle('$\bf{Q = 8.4588 \cdot 10^{-9} m^2 \: || \: R = 8.4588 \cdot 10^{-6} m^2}$','Interpreter','latex', 'FontSize', 15)
print -depsc SNIS_0001.eps


%%% 3e)

% loading data
load K2_4x0_05rho1000.mat
load K2_4x0_05rho100.mat
load K2_4x0_05rho10.mat
load K2_4x0_05rho1.mat
load K2_4x0_05rho0_1.mat
load K2_4x0_05rho0_01.mat
load K2_4x0_05rho0_001.mat

t = K2_4x0_05rho1000(1:351, 5);
reference = zeros(size(t));
reference(t<=0) = -0.25;
reference(t>0) = -0.1;

x1 = -K2_4x0_05rho1000(1:351, 9);
x2 = -K2_4x0_05rho100(1:351, 9);
x3 = -K2_4x0_05rho10(1:351, 9);
x4 = -K2_4x0_05rho1(1:351, 9);
x5 = -K2_4x0_05rho0_1(1:351, 9);
x6 = -K2_4x0_05rho0_01(1:351, 9);
x7 = -K2_4x0_05rho0_001(1:351, 9);

xhat1 = K2_4x0_05rho1000(1:351, 10);
xhat2 = K2_4x0_05rho100(1:351, 10);
xhat3 = K2_4x0_05rho10(1:351, 10);
xhat4 = K2_4x0_05rho1(1:351, 10);
xhat5 = K2_4x0_05rho0_1(1:351, 10);
xhat6 = K2_4x0_05rho0_01(1:351, 10);
xhat7 = K2_4x0_05rho0_001(1:351, 10);

% plots
figure
hold on
plot(t, [x1, xhat1]);
plot(t, reference, 'LineWidth',2)
set(gca, 'FontSize', 11)
xlabel('time [s]')
ylabel('distance [m]')
legend({'measured', 'estimated','reference'},'Location', 'east')
title('\rho = 1000', 'Fontsize', 15)
print -depsc wrong_pos_1000.eps
hold off

figure
hold on
plot(t, [x2, xhat2]);
plot(t, reference, 'LineWidth',2)
set(gca, 'FontSize', 11)
xlabel('time [s]')
ylabel('distance [m]')
legend({'measured', 'estimated','reference'},'Location', 'east', 'FontSize', 11)
title('\rho = 100', 'Fontsize', 15)
print -depsc wrong_pos_100.eps
hold off

figure
hold on
plot(t, [x3, xhat3]);
plot(t, reference, 'LineWidth',2)
set(gca, 'FontSize', 11)
xlabel('time [s]')
ylabel('distance [m]')
legend({'measured', 'estimated','reference'},'Location', 'east', 'FontSize', 11)
title('\rho = 10', 'Fontsize', 15)
print -depsc wrong_pos_10.eps
hold off

figure
hold on
plot(t, [x4, xhat4]);
plot(t, reference, 'LineWidth',2)
set(gca, 'FontSize', 11)
xlabel('time [s]')
ylabel('distance [m]')
legend({'measured', 'estimated','reference'},'Location', 'east', 'FontSize', 11)
title('\rho = 1', 'Fontsize', 15)
print -depsc wrong_pos_1.eps
hold off

figure
hold on
plot(t, [x5, xhat5]);
plot(t, reference, 'LineWidth',2)
set(gca, 'FontSize', 11)
xlabel('time [s]')
ylabel('distance [m]')
legend({'measured', 'estimated','reference'},'Location', 'east', 'FontSize', 11)
title('\rho = 0.1', 'Fontsize', 15)
print -depsc wrong_pos_01.eps
hold off

figure
hold on
plot(t, [x6, xhat6]);
plot(t, reference, 'LineWidth',2)
set(gca, 'FontSize', 11)
xlabel('time [s]')
ylabel('distance [m]')
legend({'measured', 'estimated','reference'},'Location', 'east', 'FontSize', 11)
title('\rho = 0.01', 'Fontsize', 15)
print -depsc wrong_pos_001.eps
hold off

figure
hold on
plot(t, [x7, xhat7]);
plot(t, reference, 'LineWidth',2)
set(gca, 'FontSize', 11)
xlabel('time [s]')
ylabel('distance [m]')
legend({'measured', 'estimated','reference'},'Location', 'east', 'FontSize', 11)
title('\rho = 0.001', 'Fontsize', 15)
print -depsc wrong_pos_0001.eps
hold off


%%% 3f) 

% State estimator using pole placement
K = 2.4;
pd = 1-Ts*K;
pc = log(pd)/Ts;

% Choose estimater poles 10 times slower in continuous time
pce = pc/10;
pde = exp(pce*Ts);

% estimator gain
Lplace = acker(A',C',pde);

% load data
load K2_4_poleplacement.mat

t = K2_4_poleplacement(:, 5);
reference = zeros(size(t));
reference(t<=0) = -0.25;
reference(t>0) = -0.2;

x = -K2_4_poleplacement(:, 9);
xhat = K2_4_poleplacement(:, 10);

% plot
figure
hold on
plot(t, [x, xhat]);
plot(t, reference, 'LineWidth',2)
set(gca, 'FontSize', 11)
xlabel('time [s]')
ylabel('distance [m]')
legend({'measured', 'estimated','reference'},'Location', 'southeast')
title('Pole placement', 'Fontsize', 15)
print -depsc pole_placement.eps
hold off