function [u_inc] = IncWave(input)
gam0 = input.gamma_0;
xS = input.xS;
dx = input.dx;

% incident wave on two-dimensional grid


X1 = input.X1;
X2 = input.X2;
DIS = sqrt((X1 - xS(1)).^2+(X2 - xS(2)).^2);
DISu = DIS;
G = 1 / (2 * pi) .* besselk(0, gam0*DIS);
Gu = G;
delta = (pi)^(-1 / 2) * dx; % radius circle with area of dx^2
factor = 2 * besseli(1, gam0*delta) / (gam0 * delta);
factoru = factor;

u_inc = factor * G; % factor for weak form if DIS > delta
