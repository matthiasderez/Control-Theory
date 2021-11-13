% csvfile = 'recording3.csv';
% labels = strsplit(fileread(csvfile), '\n'); % Split file in lines
% labels = strsplit(labels{:, 2}, ', '); % Split and fetch the labels (they are in line 2 of every record)
% data = dlmread(csvfile, ',', 2, 0); % Data follows the labels

% save data

load data.mat

time = data(:,1);
va  = data(:, 5);
vb = data(:, 6);
inputVoltage = data(:, 2);

% motor velocity plots
figure
subplot(121)
plot(time, va)
ylabel('Velocity motor A [rad/s]')
xlabel('t [s]')

subplot(122)
plot(time, vb)
ylabel('Velocity motor B [rad/s]')
xlabel('t [s]')

sgtitle('Motor velocity')
print -depsc motor_velocity.eps

% motor voltage plot
figure
plot(time, inputVoltage);
ylabel('Input voltage [V]')
xlabel('t [s]')
sgtitle('Input voltage')
print -depsc input_voltage.eps






