function [Y] = BMT_FFT(X, V) % calculate X*V with FFT
% X = Nˆ2 BMT row
% V = Nˆ2 vector colom
N = sqrt(length(X));
for i = 1:N
    vec2matSimulationX(i, :) = X((i - 1)*N+1:i*N);
    vec2matSimulationV(i, :) = V((i - 1)*N+1:i*N);
end
X = vec2matSimulationX;
X(:, N+1:2*N) = [zeros(N, 1), fliplr(X(:, 2:N))]; % Extend with mirrored @ right (left to right)
X(N+1:2*N, :) = [zeros(1, 2*N); flipud(X(2:N, :))]; % Extend with mirrored @ bottom(up to down)
V = [vec2matSimulationV, zeros(N); zeros(N, 2*N)];
full_solution = ifft2(fft2(X).*fft2(V)); % 2N by 2N solution
Solution2D(1:N*N) = full_solution(1:N, 1:N).'; %takes te part we need
Y = Solution2D.';