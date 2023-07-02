function [w] = ITERBiCGSTABw(u_inc, input)
% BiCGSTAB_FFT scheme for contrast source integral equation Aw = b
itmax = 1000;
Errcri = input.Errcri;
b = input.CHI(:) .* u_inc(:); % known 1D vector

x0 = zeros(size(b));
[w, flag, relres, iter, resvec] = bicgstab(@(w) Aw(w, input), b, Errcri, itmax, [], [], x0); % call BICGSTAB

% Print the final residual norm
disp(['INITIAL! Something wrong...Final residual norm: ', num2str(norm(b - Aw(w, input).*w))]);
% Display the residual norm at each iteration
for i = 1:length(resvec)
    disp(['Iteration ', num2str(i), ' - Residual norm: ', num2str(resvec(i))]);
end

w = vector2matrix(w, input); % output matrix

end

function y = Aw(w, input)
w = vector2matrix(w, input); % Convert 1D vector to matrix
y = w - input.CHI .* Kop(w, input.FFTG);
y = y(:); % Convert matrix to 1D vector
end

function w = vector2matrix(w, input)
% Modify vector output from 'bicgstab' to matrix for further computations
w = reshape(w, [input.N1, input.N2]);
end