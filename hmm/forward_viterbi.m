function V = forward_viterbi(Z, pinit, A, B)
%FORWARD_VITERBI Forward pass of Viterbi algorithm
%   @param Z Input measured sequence 1xL or Lx1
%   @param pinit Initial probabilities Nx1
%   @param A Transitional model NxN
%   @param B Measureemtn model NxM
%   @return V Output probability V(j,i) of most likely sequence of states 
%       ending at state S_i = j NxL

% check dimensions
N = size(A,1);
M = size(B,2);
L = numel(Z);

if numel(pinit) ~= N || size(A,2) ~= N || size(B,1) ~=N || max(Z) > M
    error('dimension error')
end

V = zeros(N,L);

% init
for j=1:N
    V(j,1) = pinit(j) * B(j,Z(1));
end

% forward pass
if L > 1
    for i=2:L
        for j=1:N
            % p(x_i = j | x_{i-1} = k) V^k_{i-1}, max in all k
            P = zeros(N,1);
            for k=1:N
                P(k) = V(k,i-1) * A(k,j);
            end
            V(j,i) = B(j,Z(i)) * max(P);
        end
    end
end

end

