% Controller and open loop system A with correct gain
sysA_PI_wogain = sysA_PI;
contrA_PI_wogain = contrA_PI_d;
num_loopgainA = num_loopgainA*gainA;
num_PI_A = num_PI_A * gainA;
contrA_PI = tf(num_PI_A, den_PI_A, Ts)
sysA_PI = tf(num_loopgainA, den_loopgainA, Ts)

[mag, phase] = bode(sysA_PI,w);
phase_PI_g_A = zeros(length(w),1);
mag_PI_g_A = zeros(length(w),1);
phase_PI_g_A(:) = phase(:,:,:);
mag_PI_g_A(:) = mag(:,:,:);

sysB_PI_wogain = sysB_PI;
contrB_PI_wogain = contrB_PI_d;
num_loopgainB = num_loopgainB*gainB;
num_PI_B = num_PI_B * gainB;
contrB_PI = tf(num_PI_B, den_PI_B, Ts)
sysB_PI = tf(num_loopgainB, den_loopgainB, Ts)

[mag, phase] = bode(sysB_PI,w);
phase_PI_g_B = zeros(length(w),1);
mag_PI_g_B = zeros(length(w),1);
phase_PI_g_B(:) = phase(:,:,:);
mag_PI_g_B(:) = mag(:,:,:);


% Bodeplot controller
figure
bode(contrA_PI,w)
grid on
title('Frequency respons PI controller motor A')
print -depsc bodeplot_controllerA_method2.eps

figure
bode(contrB_PI,w)
grid on
title('Frequency respons PI controller motor B')
print -depsc bodeplot_controllerB_method2.eps

% Bodeplot open loop system As
figure
subplot(211)
semilogx(w, [20*log10(mag_oriA), 20*log10(mag_PI_A),20*log10(mag_PI_g_A)])
grid on, axis tight
ylabel('|P(j\omega_A)| [dB]')
title('Bodeplot open loop systems motor A')
legend('Original system motor A', 'system motor A with PI controller', 'system motor A with PI with correct gain', 'location', 'NorthEast')
subplot(212)
semilogx(w, [phase_oriA, phase_PI_A, phase_PI_g_A])
grid on, axis tight
ylabel('\phi(P(j\omega_A)) [^o]')
xlabel('\omega_A [rad/s]')
print -depsc bodeplot_openloop_method2.eps

figure
subplot(211)
semilogx(w, [20*log10(mag_oriB), 20*log10(mag_PI_B),20*log10(mag_PI_g_B)])
grid on, axis tight
ylabel('|P(j\omega_B)| [dB]')
title('Bodeplot open loop systems motor B')
legend('Original system motor B', 'system motor B with PI controller', 'system motor B with PI with correct gain', 'location', 'NorthEast')
subplot(212)
semilogx(w, [phase_oriB, phase_PI_B, phase_PI_g_B])
grid on, axis tight
ylabel('\phi(P(j\omega_B)) [^o]')
xlabel('\omega_B [rad/s]')
print -depsc bodeplot_openloop_method2.eps

% Bodeplot of only serial connection of controller and system A
% figure
% subplot(211)
% semilogx(w, 20*log10(mag_PI_g_A))
% grid on, axis tight
% ylabel('|P(j\omega_A)| [dB]')
% title('Bodeplot serial connection of PI controller and system motor A')
% subplot(212)
% semilogx(w, phase_PI_g_A)
% grid on, axis tight
% ylabel('\phi(P(j\omega_A)) [^o]')
% xlabel('\omega_A [rad/s]')
% print -depsc bodeplot_contr_sysA_openloop_method2.eps
% 
% figure
% subplot(211)
% semilogx(w, 20*log10(mag_PI_g_B))
% grid on, axis tight
% ylabel('|P(j\omega_B)| [dB]')
% title('Bodeplot serial connection of PI controller and system motor B')
% subplot(212)
% semilogx(w, phase_PI_g_B)
% grid on, axis tight
% ylabel('\phi(P(j\omega_B)) [^o]')
% xlabel('\omega_B [rad/s]')
% print -depsc bodeplot_contr_sysB_openloop_method2.eps

figure
margin(sysA_PI)
print -depsc bodeplot_contr_sysA_openloop_method2.eps

figure
margin(sysB_PI)
print -depsc bodeplot_contr_sysB_openloop_method2.eps

% Parameters open loop system A with PI controller 
[GM_PI_g_A,PM_PI_g_A,Wpi_PI_g_A,Wc_PI_g_A] = margin(sysA_PI);

[GM_PI_g_B,PM_PI_g_B,Wpi_PI_g_B,Wc_PI_g_B] = margin(sysB_PI);


% Bode plot closed loop system A
sysA_cl = feedback(sysA_PI,1)

[mag, phase] = bode(sysA_cl,w);
phase_clA = zeros(length(w),1);
mag_clA = zeros(length(w),1);
phase_clA(:) = phase(:,:,:);
mag_clA(:) = mag(:,:,:);

figure
subplot(211)
semilogx(w, 20*log10(mag_clA))
grid on, axis tight
ylabel('|P(j\omega_A)| [dB]')
title('Bodeplot closed loop system A with PI controller')
subplot(212)
semilogx(w,phase_clA)
grid on, axis tight
ylabel('\phi(P(j\omega_A)) [^o]')
xlabel('\omega_A [rad/s]')
print -depsc bodeplotA_cl_method2.eps

% Bode plot closed loop system B
sysB_cl = feedback(sysB_PI,1)

[mag, phase] = bode(sysB_cl,w);
phase_clB = zeros(length(w),1);
mag_clB = zeros(length(w),1);
phase_clB(:) = phase(:,:,:);
mag_clB(:) = mag(:,:,:);

figure
subplot(211)
semilogx(w, 20*log10(mag_clB))
grid on, axis tight
ylabel('|P(j\omega_B)| [dB]')
title('Bodeplot closed loop system motor B with PI controller')
subplot(212)
semilogx(w,phase_clB)
grid on, axis tight
ylabel('\phi(P(j\omega_B)) [^o]')
xlabel('\omega_B [rad/s]')
print -depsc bodeplotB_cl_method2.eps


%% Step response closed loop system 

t = [0:0.01:10];
[sim_stepA] = step(sysA_cl,t);
figure
plot(t, sim_stepA)
title('Respons motor A for a step input as reference velocity')
print -depsc stepresponsA_cl_method2.eps


error11 = sum(abs(1-sim_stepA))
gainA
GM_PI_g_A
PM_PI_g_A
Ti_A
1/Ti_A
Wc_PI_g_A
bandwidth(sysA_cl)

[sim_stepB] = step(sysB_cl,t);
figure
plot(t, sim_stepB)
title('Respons motor A for a step input as reference velocity')
print -depsc stepresponsB_cl_method2.eps


error11 = sum(abs(1-sim_stepB))
gainB
GM_PI_g_B
PM_PI_g_B
Ti_B
1/Ti_B
Wc_PI_g_B
bandwidth(sysB_cl)