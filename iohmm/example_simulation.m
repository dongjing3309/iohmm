% example of HMM simulation

clear
close all

% system model
A = zeros(2,2,2);
A(:,:,1) = [0.9 0.1;0.1 0.9];   % action: non-trans
A(:,:,2) = [0.2 0.8;0.8 0.2];   % action: trans
B = [0.9 0.1;0.1 0.9];
pinit = [0.5; 0.5];

% control seq
L = 100;
p_trans = 0.2;     % prob to chose trans action
U = (rand(L-1, 1) < p_trans) + 1;

fprintf('control:\n\n')
print_states(U)

[X, Z] = simulation(pinit, A, B, U);

fprintf('states / measurements:\n\n')
print_states(X)
print_states(Z)