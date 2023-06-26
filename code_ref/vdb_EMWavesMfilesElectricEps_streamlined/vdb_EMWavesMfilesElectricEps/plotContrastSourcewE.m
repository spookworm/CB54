function plotContrastSourcewE(w_E, input)

% Plot 2D contrast/source distribution ---------------
x1 = input.X1(:, 1);
x2 = input.X2(1, :);
set(figure, 'Units', 'centimeters', 'Position', [5, 5, 18, 12]);
subplot(1, 2, 1);
IMAGESC(x1, x2, abs(w_E{1}));
title('\fontsize{13} abs(w_1^E)');
subplot(1, 2, 2);
IMAGESC(x1, x2, abs(w_E{2}));
title('\fontsize{13} abs(w_2^E)');