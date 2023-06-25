function [data] = Dop(w, input)
gam0 = input.gamma_0;
dx = input.dx;
xR = input.xR;

data = zeros(1, input.NR);

X1 = input.X1;
X2 = input.X2;
delta = (pi)^(-1 / 2) * dx; % radius circle with area of dx^2
factor = 2 * besseli(1, gam0*delta) / (gam0 * delta);
for p = 1:input.NR
    DIS = sqrt((xR(1, p) - X1).^2+(xR(2, p) - X2).^2);
    G = 1 / (2 * pi) .* besselk(0, gam0*DIS);
    data(1, p) = (gam0^2 * dx^2) * factor * sum(G(:).*w(:));
end % p_loop
