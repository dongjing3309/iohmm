function [Z, X] = simulation(mu_init, Sigma_init, U, A, B, Sigma_s, C, Sigma_m)
%SIMULATION Simulate a IO-HMM given a system model
%   @param mu_init Initial probabilities mean Nx1
%   @param Sigma_init Initial probabilities covariance NxN
%   @param U Input control sequence Kx(L-1)
%   @param A Transitional model NxN
%   @param B Control model NxK
%   @param Sigma_s State covariance matrix NxN
%   @param C Measurement model MxN
%   @param Sigma_m Measurement covariance matrix MxM
%   @return Z Output measurements MxL
%   @return X Output ground truth states NxL


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

% init
X = mvnrnd(mu_init', Sigma_init);
Z = C * X + mvnrnd(zeros(1,M), Sigma_m)';

% system
for i=2:L
    xi = A * X(:,i-1) + B * U(:,i-1) + mvnrnd(zeros(1,N), Sigma_m)';
    zi = C * xi + mvnrnd(zeros(1,M), Sigma_m)';
    X = [X, xi];
    Z = [Z, zi];
end

end