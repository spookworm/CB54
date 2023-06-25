function [input] = initEMcontrast(input)
input.a = 40; % radius circle cylinder / radius sphere

R = sqrt(input.X1.^2+input.X2.^2);

% (1) Compute permittivity contrast --------------------------------------
input.CHI_eps = (1 - input.eps_sct) * (R < input.a);

% (2) Compute permeability contrast --------------------------------------
input.CHI_mu = (1 - input.mu_sct) * (R < input.a);