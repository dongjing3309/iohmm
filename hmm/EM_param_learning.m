function [pinit, A, B] = EM_param_learning(Z, pinit_init, A_init, B_init, nr_iter)
%EM_PARAM_LEARNING EM HMM parameter learning

% check dimensions
N = size(A_init,1);
M = size(B_init,2);
L = numel(Z);

if numel(pinit_init) ~= N || size(A_init,2) ~= N || size(B_init,1) ~=N || max(Z) > M
    error('dimension error')
end

% init
pinit = pinit_init;
A = A_init;
B = B_init;

for iter=1:nr_iter
    
    % E-step
    % ---------------------------------------------------------------------
    % forward backward pass
    Pf = forward_pass(Z, pinit, A, B);
    Pb = backward_pass(Z, A, B);
    
    % sequence likelihood
    lseq = sum(Pf(:,L));
    fprintf('[iter=%d] learned seq log2 likelihood = %f\n', iter, log2(lseq))
    
    % state probabilities given measured sequence p(x_t=i | Z)
    P = zeros(N,L);
    for i=1:L
        for j=1:N
            P(j,i) = Pf(j,i) * Pb(j,i) / lseq;
        end
    end
    
    % M-step
    % ---------------------------------------------------------------------
    % state transition probabilities given measured sequence p(x_t=i, x_{t+1}=j | Z)
    phi = zeros(N,N,L-1);
    for i=1:L-1
        for j=1:N
            for k=1:N
                phi(j,k,i) = Pf(j,i) * A(j,k) * B(k,Z(i+1)) * Pb(k,i+1) / lseq;
            end
        end
    end
    
    % update params
    for i=1:N
        pinit(i) = P(i,1);
    end
    
    for i=1:N
        for j=1:N
            A(i,j) = sum(phi(i,j,1:L-1)) / sum(P(i,1:L-1));
        end
    end
    
    for i=1:N
        for k=1:M
            zk_idx = find(Z==k);
            B(i,k) = sum(P(i, zk_idx)) / sum(P(i,1:L));
        end
    end
end


end

