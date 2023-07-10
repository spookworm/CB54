clear all; clc; close all; clear workspace
format long;

input = init();

%  (1) Compute analytically scattered field data --------------------------
ForwardCanonicalObjects

%  (2) Compute incident field ---------------------------------------------
u_inc = IncWave(input);

%  (3) Solve integral equation for contrast source with FFT ---------------
tic;
w = ITERBiCGSTABw(u_inc, input);
toc;

plotContrastSource(w, input);

%  (4) Compute synthetic data and plot fields and data --------------------
data = Dop(w, input);
displayData(data, input);

if exist(fullfile(cd, 'DATA2D.mat'), 'file');
    load DATA2D data2D;
    error = norm(data(:)-data2D(:), 1) / norm(data2D(:), 1);
end
display(vpa(error, 20));
save('.\w.mat', "w");
save('.\data.mat', "data");