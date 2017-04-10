function [X, Sigma_x] = kalman_forwardbackward(Z, U, mu_init, Sigma_init, A, B, Sigma_s, ...
    C, Sigma_m)
%KALMAN_FORWARDBACKWARD Kalman smoothing (forward-backward passing)
%   @param mu_init Initial probabilities mean Nx1
%   @param Sigma_init Initial probabilities covariance NxN
%   @param U Input control sequence Kx(L-1)
%   @param A Transitional model NxN
%   @param B Control model NxK
%   @param Sigma_s State covariance matrix NxN
%   @param C Measurement model MxN
%   @param Sigma_m Measurement covariance matrix MxM
%   @return X Output estimated mean NxL
%   @return Sigma_x Output estimated covariance NxNxL


% check dimensions
N = size(A,1);
K = size(B,2);
M = size(C,1);
L = size(U,2) + 1;

if sum( size(mu_init) ~= [N 1] | size(Sigma_init) ~= [N N] | size(U) ~= [K L-1] ...
        | size(A) ~= [N N] | size(B) ~= [N K] | size(Sigma_s) ~= [N N] ...
        | size(C) ~= [M N] | size(Sigma_m) ~= [M M])
    error('dimension error')
end

% forward passing
% -------------------------------------------------------------------------

% init state
X = mu_init;
Sigma_x = zeros(N,N,L);
Sigma_x(:,:,1) = Sigma_init;

% cache for backward passing
X_bar = [];
Sigma_x_bar = zeros(N,N,L-1);

for i=2:L
    % prediction
    mu_t_bar = A * X(:,i-1) + B * U(:,i-1);
    Sigma_t_bar = A * Sigma_x(:,:,i-1) * A' + Sigma_s;
    
    % measurement update
    Kt = Sigma_t_bar * C' * inv(C * Sigma_t_bar *C' + Sigma_m);
    mu_t = mu_t_bar + Kt * (Z(i) - C * mu_t_bar);
    Sigma_t = (eye(N) - Kt * C) * Sigma_t_bar;
    
    X = [X, mu_t];
    Sigma_x(:,:,i) = Sigma_t;
    
    X_bar = [X_bar, mu_t_bar];
    Sigma_x_bar(:,:,i-1) = Sigma_t_bar;
end

% backward passing
% -------------------------------------------------------------------------

for i=L-1:-1:1
    Li = Sigma_x(:,:,i) * A' * Sigma_x_bar(:,:,i);
    X(:,i) = X(:,i) + Li * (X(:,i+1) - X_bar(i));
    Sigma_x(:,:,i) = Sigma_x(:,:,i) + Li * (Sigma_x(:,:,i+1) - Sigma_x_bar(:,:,i)) * Li';
end


end

