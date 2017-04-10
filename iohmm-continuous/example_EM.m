% 1-D example of EM param learning

clear
close all

% 1-dim system model
A = 1;
B = 1;
Sigma_s = 0.2;
C = 1;
Sigma_m = 1.0;


% control input
L = 500;
U = normrnd(0, 1, 1, L-1);

% init state
mu_init = 0;
Sigma_init = 1;
% simulation
[Z, X] = simulation(mu_init, Sigma_init, U, A, B, Sigma_s, C, Sigma_m);


% ground truth param init
A_init = A;
B_init = B;
Sigma_s_init = Sigma_s;
C_init = C;
Sigma_m_init = Sigma_m;

% random param init

% EM param learning
nr_iter = 20;

[A, B, Sigma_s, C, Sigma_m, Xest] = EM_param_learning(Z, U, mu_init, Sigma_init, ...
    A_init, B_init, Sigma_s_init, C_init, Sigma_m_init, nr_iter);

A, B, Sigma_s, C, Sigma_m


% plot
figure, grid, hold on
gtruth = plot(1:L, X, 'b');
meas = plot(1:L, Z, 'r.');
xest = plot(1:L, Xest, 'm-.');
legend([gtruth meas xest], 'Ground truth', 'Measurements', ....
    'EM estimated')
hold off


