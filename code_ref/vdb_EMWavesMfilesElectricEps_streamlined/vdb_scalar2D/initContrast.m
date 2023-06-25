function [input] = initContrast(input)
input.a = 40; % half width slab / radius circle cylinder / radius sphere
contrast = 1 - input.c_0^2 / input.c_sct^2;
R = sqrt(input.X1.^2+input.X2.^2);


input.CHI = contrast .* (R < input.a);
