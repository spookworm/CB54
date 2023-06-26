clear; clc; close all;
input = init();
c_0 = input.c_0;
c_sct = input.c_sct;
gam0 = input.gamma_0;
gam_sct = input.gamma_0 * c_0 / c_sct;

if exist(fullfile(cd, 'DATA2D.mat'), 'file')
    delete DATA2D.mat;
end

% (1) Compute coefficients of series expansion ----------------------------








arg0 = gam0 * input.a;
args = gam_sct * input.a;
M = 100; % increase M for more accuracy
A = zeros(1, M+1);
for m = 0:M
    Ib0 = besseli(m, arg0);
    dIb0 = besseli(m+1, arg0) + m / arg0 * Ib0;
    Ibs = besseli(m, args);
    dIbs = besseli(m+1, args) + m / args * Ibs;
    Kb0 = besselk(m, arg0);
    dKb0 = -besselk(m+1, arg0) + m / arg0 * Kb0;
	
    A(m+1) = -(gam_sct * dIbs * Ib0 - gam0 * dIb0 * Ibs) ...
        / (gam_sct * dIbs * Kb0 - gam0 * dKb0 * Ibs);
end

% (2) Compute reflected field at receivers (data) -------------------------
xR = input.xR;
xS = input.xS;
rR = sqrt(xR(1, :).^2+xR(2, :).^2);
phiR = atan2(xR(2, :), xR(1, :));
rS = sqrt(xS(1)^2+xS(2)^2);
phiS = atan2(xS(2), xS(1));
data2D = A(1) * besselk(0, gam0*rS) .* besselk(0, gam0*rR);
for m = 1:M
    factor = 2 * besselk(m, gam0*rS) .* cos(m*(phiS - phiR));
    data2D = data2D + A(m+1) * factor .* besselk(m, gam0*rR);
end % m_loop
data2D = 1 / (2 * pi) * data2D;
displayData(data2D, input);
save DATA2D data2D;