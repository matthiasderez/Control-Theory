%% 1

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

% response speed depending on K
figure
hold on
for K = 10:10:50
    sys = ss(A-B*K,B,C-D*K,D,Ts);
    impulse(sys)
end
set(gca, 'Fontsize', 11)
ylim([-1.2 0.2])
legend({'0','10','20','30','40','50'})
hold off
print -depsc impulse_response1.eps

figure
hold on
for K = 150:10:190
    sys = ss(A-B*K,B,C-D*K,D,Ts);
    impulse(sys)
end
set(gca, 'Fontsize', 11)
legend({'150','160','170','180','190'})
hold off
print -depsc impulse_response2.eps

figure
hold on
for K = [0, 100]
    sys = ss(A-B*K,B,C-D*K,D,Ts);
    impulse(sys)
end
set(gca, 'Fontsize', 11)
ylim([-1.2 0.2])
legend({'0', '100'})
print -depsc impulse_response3.eps

%% 2
% LQE estimator gain
Q = 0.1;
R = 0.01;
L = dlqr(A', A'*C', Q, R)'

% steady state Kalmain gain using derived formula
P = (1/2)*(-Q+ sqrt(Q^2+4*R*Q));
L = -(P+Q)/(P+Q+R)

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

