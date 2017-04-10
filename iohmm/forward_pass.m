function Pf = forward_pass(Z, pinit, A, B, U)
%FORWARD_PASS Forward pass algorithm
%   @param Z Input measured sequence 1xL or Lx1
%   @param pinit Initial probabilities Nx1
%   @param A Transitional model NxNxK
%   @param B Measureemtn model NxM
%   @param U Input control sequence (L-1)x1
%   @return Pf Output forward pass results NxL

% check dimensions
N = size(A,1);
M = size(B,2);
L = numel(Z);

if numel(pinit) ~= N || size(A,2) ~= N || size(B,1) ~=N || max(Z) > M || numel(U) ~= L-1
    error('dimension error')
end

Pf = zeros(N,L);

% init
for j=1:N
    Pf(j,1) = pinit(j) * B(j,Z(1));
end

% forward pass
for i=2:L
    for j=1:N
        % Pf(j,i) = 0
        for k=1:N
            Pf(j,i) = Pf(j,i) + Pf(k,i-1) * A(k,j,U(i-1)) * B(j,Z(i));
        end
    end
end

end
