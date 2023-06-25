function displayData(data, input)
if exist(fullfile(cd, 'DATA2D.mat'), 'file')
    load DATA2D data2D;
    error = num2str(norm(data(:)-data2D(:), 1)/norm(data2D(:), 1));
    disp(['error=', error]);
end
set(figure, 'Units', 'centimeters', 'Position', [5, 5, 18, 7]);
angle = input.rcvr_phi * 180 / pi;
if exist(fullfile(cd, 'DATA2D.mat'), 'file')
    plot(angle, abs(data), '--r', angle, abs(data2D), 'b')
    legend('Integral-equation method', ...
        'Bessel-function method', 'Location', 'NorthEast');
    text(50, 0.8*max(abs(data)), ...
        ['Error^{sct} = ', error, '  '], 'EdgeColor', 'red', 'Fontsize', 11);
else plot(angle, abs(data), 'b')
    legend('Bessel-function method', 'Location', 'NorthEast');
end
title('\fontsize{12} scattered wave data in 2D');
axis tight;
xlabel('observation angle in degrees');
ylabel('abs(data) \rightarrow');
