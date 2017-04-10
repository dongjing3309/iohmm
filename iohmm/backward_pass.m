function Pb = backward_pass(Z, A, B, U)
%BACKWARD_PASS Backward pass algorithm
%   @param Z Input measured sequence 1xL or Lx1
%   @param A Transitional model NxNxK
%   @param B Measureemtn model NxM
%   @param U Input control sequence (L-1)x1
%   @return Pb Output backward pass results NxL

% check dimensions
N = size(A,1);
M = size(B,2);
L = numel(Z);

if size(A,2) ~= N || size(B,1) ~=N || max(Z) > M || numel(U) ~= L-1
    error('dimension error')
end

Pb = zeros(N,L);

% init
Pb(:,L) = ones(N,1);

% backward pass
for i=L-1:-1:1
    for j=1:N
        % Pb(j,i) = 0
        for k=1:N
            Pb(j,i) = Pb(j,i) + A(j,k,U(i)) * B(k,Z(i+1)) * Pb(k,i+1);
        end
    end
end

end

