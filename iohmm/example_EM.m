% example of IO-HMM EM learning
% 2-states trans-and-no-trans example

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

% sequence likelihood
Pf = forward_pass(Z, pinit, A, B, U);
lseq = sum(Pf(:,L));
fprintf('\nsequence log2 likelihood = %f\n\n', log2(lseq))

% init param
A_init = rand(2,2,2);
B_init = rand(2,2);
pinit_init = rand(2,1);

% normalize init param
for k=1:size(A_init,3)
    for i=1:size(A_init,1)
        A_init(i,:,k) = A_init(i,:,k) / sum(A_init(i,:,k));
    end
end
for i=1:size(B_init,1)
    B_init(i,:) = B_init(i,:) / sum(B_init(i,:));
end
pinit_init = pinit_init / sum(pinit_init);

nr_iter = 20;

[pinit_lrn, A_lrn, B_lrn] = EM_param_learning(Z, U, pinit_init, A_init, B_init, nr_iter)

