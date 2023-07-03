function [w_E] = ITERBiCGSTABwE(E_inc, input)
% BiCGSTAB scheme for contrast source integral equation Aw = b
itmax = 1000;
Errcri = input.Errcri;
[N, ~] = size(input.CHI_eps(:));
b(1:N, 1) = input.CHI_eps(:) .* E_inc{1}(:);
b(N+1:2*N, 1) = input.CHI_eps(:) .* E_inc{2}(:);
x0 = complex(zeros(size(b)));

w = bicgstab(@(w) Aw(w, input), b, Errcri, itmax, [], [], x0); % call BICGSTAB
% [w, flag, relres, iter, resvec] = bicgstab(@(w) Aw(w, input), b, Errcri, itmax); % call BICGSTAB
% display(flag)
% display(relres)
% display(iter)
% display(resvec)
[w_E] = vector2matrix(w, input); % output matrices
end %----------------------------------------------------------------------

function y = Aw(w, input)
[N, ~] = size(input.CHI_eps(:));
[w_E] = vector2matrix(w, input);
[Kw_E] = KopE(w_E, input);
y(1:N, 1) = w_E{1}(:) - input.CHI_eps(:) .* Kw_E{1}(:);
y(N+1:2*N, 1) = w_E{2}(:) - input.CHI_eps(:) .* Kw_E{2}(:);
% display(max(y))
end %----------------------------------------------------------------------

function [w_E] = vector2matrix(w, input)
% Modify vector output from 'bicgstab' to matrices for further computation
[N, ~] = size(input.CHI_eps(:));
w_E = cell(1, 2);
DIM = [input.N1, input.N2];
w_E{1} = reshape(w(1:N, 1), DIM);
w_E{2} = reshape(w(N+1:2*N, 1), DIM);
end