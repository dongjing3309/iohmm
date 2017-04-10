% example of HMM viterbi

clear
close all

% system model
N = 2;
M = 2;
A = [0.8 0.2;0.3 0.7];
B = [0.9 0.1;0.1 0.9];
pinit = [0.5; 0.5];

% simulation data
L = 100;
[X, Z] = simulation(pinit, A, B, L);

fprintf('states / measurements:\n\n')
print_states(X)
disp('-----')
print_states(Z)

% sequence likelihood
Pf = forward_pass(Z, pinit, A, B);
lseq = sum(Pf(:,L));
fprintf('\nsequence log2 likelihood = %f\n\n', log2(lseq))

% init param
A_init = rand(2,2);
B_init = rand(2,2);
pinit_init = rand(2,1);

nr_iter = 20;

[pinit_lrn, A_lrn, B_lrn] = EM_param_learning(Z, pinit_init, A_init, B_init, nr_iter)

