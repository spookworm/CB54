function input = initGrid(input)
% Grid in two-dimensional space -----------------------
    input.N1 = 120; % number of samples in x_1
    input.N2 = 100; % number of samples in x_2
    input.dx = 2; % with meshsize dx
    x1 = -(input.N1 + 1) * input.dx / 2 + (1:input.N1) * input.dx;
    x2 = -(input.N2 + 1) * input.dx / 2 + (1:input.N2) * input.dx;
    [input.X1, input.X2] = ndgrid(x1, x2);
    % Now array subscripts are equivalent with Cartesian coordinates
    % x1 axis points downwards and x2 axis is in horizontal direction
    % x1 = X1(:,1) is a column vector in vertical direction
    % x2 = X2(1,:) is a row vector in horizontal direction
