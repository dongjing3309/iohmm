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
L = 50;
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
fprintf('\nsequence log2 likelihood = %f\n\n', log2(lseq))
    
% state probabilities given measured sequence
P = zeros(N,L);
for i=1:L
    for j=1:N
        P(j,i) = Pf(j,i) * Pb(j,i) / lseq;
    end
end

% best possible states: estimate seperately
Pmax = max(P);
Xest = [];
for i=1:L
    Xest = [Xest;, find(P(:,i)==Pmax(i))];
end

disp('estimated states: seperately')
print_states(Xest)

% best possible states: viterbi algorithm
V = forward_viterbi(Z, pinit, A, B);
Xv = backward_viterbi(V, A);

disp('estimated states: viterbi algorithm')
print_states(Xv)

