% example of HMM forward-backward

clear
close all

% system model
N = 2;
M = 2;
A = [0.7 0.3;0.4 0.6];
B = [0.8 0.2;0.2 0.8];
pinit = [0.5; 0.5];

% simulation data
L = 10;
[X, Z] = simulation(pinit, A, B, L);

fprintf('states / measurements:\n\n')
print_states(X)
disp('-----')
print_states(Z)

% forward-backward
Pf = forward_pass(Z, pinit, A, B);
Pb = backward_pass(Z, A, B);

% sequence likelihood
lseq = sum(Pf(:,L));
fprintf('\nsequence likelihood = %f\n\n', lseq)
    
% state probabilities given measured sequence
P = zeros(N,L);
for i=1:L
    for j=1:N
        P(j,i) = Pf(j,i) * Pb(j,i) / lseq;
    end
end

% best possible states
Pmax = max(P);
Xest = [];
for i=1:L
    Xest = [Xest;, find(P(:,i)==Pmax(i))];
end

disp('estimated states')
print_states(Xest)
