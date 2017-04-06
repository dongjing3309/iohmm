function S = backward_viterbi(V, A)
%BACKWARD_VITERBI Backward pass of Viterbi algorithm
%   @param V Forward pass probability V(j,i) of most likely sequence of states 
%       ending at state S_i = j NxL
%   @param A Transitional model NxN
%   @return S Max possible sequence

% check dimensions
N = size(A,1);
L = size(V,2);

if size(A,2) ~= N || size(V,1) ~=N
    error('dimension error')
end

% init
S = zeros(N,1);
S(L) = find(V(:,L) == max(V(:,L)));

% backward pass
if L > 1
    for i=L-1:-1:1
        % argmax_k p(S(i+1) | S(i)=k) V(k, i) 
        P = zeros(N,1);
        for k=1:N
            P(k) = A(k, S(i+1)) * V(k, i);
        end
        S(i) = find(P == max(P));
    end
end

end

