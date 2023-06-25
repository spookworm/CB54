function plotEtotalwavefield(E_inc, E_sct, input)
a = input.a;

E = cell(1, 2);
for n = 1:2
    E{n} = E_inc{n} + E_sct{n};
end

% Plot wave fields in two-dimensional space ----------------

set(figure, 'Units', 'centimeters', 'Position', [5, 5, 18, 12]);
x1 = input.X1(:, 1);
x2 = input.X2(1, :);
N1 = input.N1;
N2 = input.N2;
subplot(1, 2, 1);
IMAGESC(x1, x2, abs(E{1}));
title(['\fontsize{13} 2D Electric field E_1 '])
hold on;
phi = 0:.01:2 * pi;
plot(a*cos(phi), a*sin(phi), 'w');
subplot(1, 2, 2);
IMAGESC(x1, x2, abs(E{2}));
title(['\fontsize{13} 2D Electric field E_2 '])
hold on;
phi = 0:.01:2 * pi;
plot(a*cos(phi), a*sin(phi), 'w');
