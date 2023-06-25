function input = initEMgrid(input)
% Grid in two-dimensional space ------------------------
input.N1 = 120; % number of samples in x_1
input.N2 = 100; % number of samples in x_2
input.dx = 2; % with meshsize dx
x1 = -(input.N1 + 1) * input.dx / 2 + (1:input.N1) * input.dx;
x2 = -(input.N2 + 1) * input.dx / 2 + (1:input.N2) * input.dx;
[input.X1, input.X2] = ndgrid(x1, x2);