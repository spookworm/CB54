%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                                     %%%
%%%               Function to calculate X*V with 2D FFT,                %%%
%%%                   Size of space grid is N by M,                     %%%
%%%   X = Row of Blocked Mirrored Toeplitz function in N orientatien    %%%
%%%                   So both V and X has length N*M                    %%%
%%%                                                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Output] = BMT_FFT(X, V, N)
M = length(X) / N;
vec2matSimulationX = zeros(M, N);
vec2matSimulationV = zeros(M, N);
for i = 1:M
    vec2matSimulationX(i, :) = X((i - 1)*N+1:i*N);
    vec2matSimulationV(i, :) = V((i - 1)*N+1:i*N);
end
X = vec2matSimulationX;
X(:, N+1:2*N) = [zeros(M, 1), fliplr(X(:, 2:N))];
X(M+1:2*M, :) = [zeros(1, 2*N); flipud(X(2:M, :))];
V = [vec2matSimulationV, zeros(M, N); zeros(M, 2*N)];
full_solution = ifft2(fft2(X).*fft2(V));
SolutionPartWeNeed(1:N*M) = full_solution(1:M, 1:N).';
Output = SolutionPartWeNeed.';