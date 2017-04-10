% 1-D example of Kalman filtering/smoothing estimation

clear
close all

% 1-dim system model
A = 1;
B = 1;
Sigma_s = 0.2;
C = 1;
Sigma_m = 1.0;


% control input
L = 100;
U = normrnd(0, 1, 1, L-1);

% init state
mu_init = 0;
Sigma_init = 1;
% simulation
[Z, X] = simulation(mu_init, Sigma_init, U, A, B, Sigma_s, C, Sigma_m);

% forward: Kalman filtering
[Xf, Sigma_xf] = kalman_forward(Z, U, mu_init, Sigma_init, A, B, Sigma_s, ...
    C, Sigma_m);


% plot
figure, grid, hold on
gtruth = plot(1:L, X, 'b');
meas = plot(1:L, Z, 'r.');
fest = plot(1:L, Xf, 'm-.');
legend([gtruth meas], 'Ground truth', 'Measurements')
hold off
