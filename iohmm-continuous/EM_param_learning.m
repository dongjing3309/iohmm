function [A, B, Sigma_s, C, Sigma_m, X] = EM_param_learning(Z, U, mu_init, Sigma_init, ...
    A_init, B_init, Sigma_s_init, C_init, Sigma_m_init, nr_iter)
%EM_PARAM_LEARNING Summary of this function goes here
%   Detailed explanation goes here

% check dimensions
N = size(A_init,1);
K = size(B_init,2);
M = size(C_init,1);
L = size(U,2) + 1;

if sum( size(mu_init) ~= [N 1] | size(Sigma_init) ~= [N N] | size(U) ~= [K L-1] ...
        | size(A_init) ~= [N N] | size(B_init) ~= [N K] | size(Sigma_s_init) ~= [N N] ...
        | size(C_init) ~= [M N] | size(Sigma_m_init) ~= [M M])
    error('dimension error')
end


% init
A = A_init;
B = B_init;
Sigma_s = Sigma_s_init;
C = C_init;
Sigma_m = Sigma_m_init;

for iter=1:nr_iter

    fprintf('[iter %d] ...\n', iter)
    
    % E-step
    % ---------------------------------------------------------------------
    
    [X, Sigma_x] = kalman_forwardbackward(Z, U, mu_init, Sigma_init, A, B, Sigma_s, ...
    	C, Sigma_m);
    
    % M-step
    % ---------------------------------------------------------------------
    
    % A, B, Sigma_s
    % cache vars 
    xt1_xt1t = zeros(N,N);
    xt1_xtt = zeros(N,N);
    xt1_utt = zeros(N,N);
    xt_xt1t = zeros(N,N);
    xt_xtt = zeros(N,N);
    xt_utt = zeros(N,N);
    ut_xt1t = zeros(N,N);
    ut_xtt = zeros(N,N);
    ut_utt = zeros(N,N);
    
    for i=1:L-1
        xt1_xt1t = xt1_xt1t + X(:,i+1) * X(:,i+1)';
        xt1_xtt = xt1_xtt + X(:,i+1) * X(:,i)';
        xt1_utt = xt1_utt + X(:,i+1) * U(:,i)';
        xt_xt1t = xt_xt1t + X(:,i) * X(:,i+1)';
        xt_xtt = xt_xtt + X(:,i) * X(:,i)';
        xt_utt = xt_utt + X(:,i) * U(:,i)';
        ut_xt1t = ut_xt1t + U(:,i) * X(:,i+1)';
        ut_xtt = ut_xtt + U(:,i) * X(:,i)';
        ut_utt = ut_utt + U(:,i) * U(:,i)';
    end
    
    A_tmp = (xt1_xtt - B * ut_xtt) * inv(xt_xtt);
    B_tmp = (xt1_utt - A * xt_utt) * inv(ut_utt);
    Sigma_s_tmp = 1/L * (xt1_xt1t - xt1_xtt * A' - xt1_utt * B' - A * xt_xt1t ...
        + A * xt_xtt * A' + A * xt_utt * B' - B * ut_xt1t + B * ut_xtt * A' ...
        + B * ut_utt * B');
    
    % C, Sigma_m
    xt_xtt = zeros(M,M);
    xt_ytt = zeros(M,M);
    yt_xtt = zeros(M,M);
    yt_ytt = zeros(M,M);
    
    for i=1:L
        xt_xtt = xt_xtt + X(:,i) * X(:,i)';
        xt_ytt = xt_ytt + X(:,i) * Z(:,i)';
        yt_xtt = yt_xtt + Z(:,i) * X(:,i)';
        yt_ytt = yt_ytt + Z(:,i) * Z(:,i)';
    end
    
    C_tmp = yt_xtt * inv(xt_xtt);
    Sigma_m_tmp = 1/(L+1) * (yt_ytt - yt_xtt * C' - C * xt_ytt + C * xt_xtt * C');
    
    % update params
    A = A_tmp;
    B = B_tmp;
    Sigma_s = Sigma_s_tmp;
    C = C_tmp;
    Sigma_m = Sigma_m_tmp;
end

end

