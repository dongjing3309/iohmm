function [Z, X] = simulation(pinit, A, B, U)
%SIMULATION Simulate a IO-HMM given a system model
%   @param pinit Initial probabilities Nx1
%   @param A Transitional model NxNxK
%   @param B Measureemtn model NxM
%   @param U Input control sequence (L-1)x1
%   @return Z Output measurements Lx1
%   @return X Output ground truth states Lx1


% check dimensions
N = size(A,1);
M = size(B,2);
L = numel(U) + 1;
K = size(A,3);

if numel(pinit) ~= N || size(A,2) ~= N || size(B,1) ~=N || max(U) > K
    error('dimension error')
end

% initial states
X = state_sampler(pinit);
Z = state_sampler(B(X(1),:));

for i=2:L
    X = [X; state_sampler(A(X(i-1),:,U(i-1)))];
    Z = [Z; state_sampler(B(X(i),:))];
end

end


% sampler function
function x = state_sampler(xprob)

if abs(sum(xprob) - 1.0) > 1e-6
    error('sum prob should be one')
end

accum_prob = cumsum(xprob);
sample = rand;
for i=1:numel(xprob)
    if sample <= accum_prob(i)
        x = i;
        break;
    end
end

end
