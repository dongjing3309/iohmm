function [Z, X] = simulation(pinit, A, B, L)
%SIMULATION Simulate a HMM given a system model
%   @param pinit Initial probabilities Nx1
%   @param A Transitional model NxN
%   @param B Measureemtn model NxM
%   @param L Sequence length
%   @return Z Output measurements Lx1
%   @return X Output ground truth states Lx1


% check dimensions
N = size(A,1);
M = size(B,2);

if numel(pinit) ~= N || size(A,2) ~= N || size(B,1) ~=N
    error('dimension error')
end

% initial states
X = state_sampler(pinit);
Z = state_sampler(B(X(1),:));

if L > 1
    for i=2:L
        X = [X; state_sampler(A(X(i-1),:))];
        Z = [Z; state_sampler(B(X(i),:))];
    end
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
