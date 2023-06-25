function plotContrastSource(w, input)
CHI = input.CHI;
% Plot 2D contrast/source distribution -------------
x1 = input.X1(:, 1);
x2 = input.X2(1, :);
set(figure, 'Units', 'centimeters', 'Position', [5, 5, 18, 12]);
subplot(1, 2, 1)
IMAGESC(x1, x2, CHI);
title('\fontsize{13} \chi = 1 - c_0^2 / c_{sct}^2');
subplot(1, 2, 2)
IMAGESC(x1, x2, abs(w))
title('\fontsize{13} abs(w)');