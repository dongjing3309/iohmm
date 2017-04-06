% example of HMM simulation

clear
close all

% system model
A = [0.7 0.3;0.4 0.6];
B = [0.8 0.2;0.2 0.8];
pinit = [0.5; 0.5];

[X, Z] = simulation(pinit, A, B, 20);

fprintf('states / measurements:\n\n')
print_states(X)
disp('-----')
print_states(Z)