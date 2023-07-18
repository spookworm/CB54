%%%     2D Scattering Solver using 2D FFT and reduced CG algorithm     %%%
clear;
close all;
clc;
format compact;

total_time = tic;

% Inputs
path_geo = './Geometry/';
path_lut = './lut/materials.json';
% object_name = 'object_mp_landscape_empty.txt';
object_name = 'placeholder.png';
input_disc_per_lambda = 10; % chosen accuracy
input_carrier_frequency = 60e6; % frequency (Hz)
input_solver_tol = 1e-1; % error tolerance

% Hard-coded Variables
epsilon0 = 8.854187817e-12;
mu0 = 4.0 * pi * 1.0e-7;
angular_frequency = 2.0 * pi * input_carrier_frequency; % angular frequency (rad/s)
% matlab_offset = 1;

% image_object = readmatrix([path_geo, object_name]);
image_object = im2gray(imread([path_geo, object_name]));
unique_integers = unique(image_object, 'sorted');

% Pass list of master materials with associated visualisation colourings to be used in scene to the generator.
% Order of this list should not change unless the numeric identifiers in the imported geometry also reflect such changes.
fid = fopen(path_lut); % Opening the file
raw = fread(fid,inf); % Reading the contents
str = char(raw'); % Transformation
fclose(fid); % Closing the file
materials_master = struct2table(jsondecode(str)); % Using the jsondecode function to parse JSON from string

% Subset on entries in the scene.
materials_master = materials_master(ismember(materials_master.("uint8"), unique_integers), :);

materials_master.('map') = sscanf(char(materials_master.('HEX'))', '#%2x%2x%2x', [3, size(materials_master.('HEX'), 1)]).' / 255;

% Switch the source dictionary so only the required names are printed
markerColor = mat2cell(materials_master.('map'), ones(1, height(materials_master.('name'))), 3);

for k = 1:length(unique_integers)
    [materials_master.('epsilonr')(k), materials_master.('sigma')(k), materials_master.('epsilonr_complex')(k)] = buildingMaterialPermittivity(materials_master.('name'){k}, input_carrier_frequency);
    materials_master.('mur')(k) = 1.0;
    materials_master.('mur_complex')(k) = 1.0 - (0.0 * 1i);
    materials_master.('cr')(k) = 1.0 / sqrt(materials_master.('epsilonr')(k)*epsilon0*materials_master.('mur')(k)*mu0);
    materials_master.('cr_complex')(k) = 1.0 / sqrt(materials_master.('epsilonr_complex')(k)*epsilon0*materials_master.('mur')(k)*mu0);
    materials_master.('kr')(k) = angular_frequency * sqrt(materials_master.('epsilonr')(k)*epsilon0*materials_master.('mur')(k)*mu0);
    materials_master.('kr_complex')(k) = angular_frequency * sqrt(materials_master.('epsilonr_complex')(k)*epsilon0*materials_master.('mur')(k)*mu0);
end

% Assign vacuum for speed-up.
vacuum_kr = materials_master(strcmp(materials_master.('name'), 'vacuum'),:).('kr');

% VISUALISE
image_object_render(image_object, materials_master, markerColor, 'Material Configuration Before Scaling for f')

% This assumes that the scale of the imported geometry is 1m per cell in both directions.
% As a result, the room in the original code will not work properly. Use
% the old code to work with that room.
[length_y_side, length_x_side] = size(image_object); % M~x in meters

% Choose the smallest lambda (wavelength) of all materials in the configuration.
lambda_smallest = min(materials_master.('cr')) / input_carrier_frequency;

% decision to be made here is which one should be calculated first to shortcut the mod loop
% this decision is based on which  length_?_side is BIGGER
if length_x_side > length_y_side
    N = floor(length_x_side/(abs(lambda_smallest) / input_disc_per_lambda)); % force N = multp 4
    fourth_of_N = ceil(N/4);
    while (mod(N, fourth_of_N) ~= 0)
        N = N + 1;
    end
    delta_x = length_x_side / N;

    M = floor(length_y_side/(delta_x)); % force M = multp 4, size dy near dx
    fourth_of_M = ceil(M/4);
    while (mod(M, fourth_of_M) ~= 0)
        M = M + 1;
    end
    delta_y = length_y_side / M;
else
    M = floor(length_y_side/(abs(lambda_smallest) / input_disc_per_lambda)); % force N = multp 4
    fourth_of_M = ceil(M/4);
    while (mod(M, fourth_of_M) ~= 0)
        M = M + 1;
    end
    delta_y = length_y_side / M;

    N = floor(length_x_side/(delta_y)); % force N = multp 4
    fourth_of_N = ceil(N/4);
    while (mod(N, fourth_of_N) ~= 0)
        N = N + 1;
    end
    delta_x = length_x_side / N;
end
equiv_a = sqrt(delta_x*delta_y/pi);

% Need to change geometry resolution here if frequency is lower scale
% Also need to check how upscaling and downscaling perform.
image_resize = imresize(image_object, [M, N], "nearest");

% VISUALISE
% image_object_render(image_object, materials_master, markerColor, 'Material Configuration After Scaling for f')

centre = 0.0 + 0.0 * 1i;
start_pt = centre - 0.5 * length_x_side - 0.5 * length_y_side * 1i;
fprintf('\n \nSize of field of interst is %d meter by %d meter \n', length_x_side, length_y_side)
fprintf('Discretizised space is %d grids by %d grids\n', N, M)
fprintf('with grid size %d meter by %d meter \n', delta_x, delta_y)

% SPECIFY_MATERIALS: POSITION, K AND RHO, FOR EACH BASIS COUNTER
% INTERIOR SPECIFICATION: position, phi, k, and rho for each number of position (basis counter)
basis_counter = 0;
% This is wrong I think unless the vacuum in the scene is indexed as one.
% Would it be better to pre-assign based on most likely material occurence?
basis_wave_number(1:M*N, 1) = vacuum_kr;
position = zeros(M*N, 1);
rho = zeros(M*N, 1);
the_phi = zeros(M*N, 1);
for ct1 = 1:M % runs in y direction
    for ct2 = 1:N % runs in x direction
        basis_counter = (ct1 - 1) * N + ct2; % nr of position
        % ORIENTATION OF BASIS COUNTER WILL BE IN X DIRECTION!

        position(basis_counter, 1) = start_pt + (ct2 - 0.5) * delta_x + 1i * (ct1 - 0.5) * delta_y;
        temp_vec = position(basis_counter, 1) - centre;
        rho(basis_counter, 1) = abs(temp_vec);
        the_phi(basis_counter, 1) = atan2(imag(temp_vec), real(temp_vec));

        % basis_wave_number(basis_counter) = kr(image_resize(ct1, ct2));
        basis_wave_number(basis_counter) = materials_master(materials_master.('uint8') == image_resize(ct1, ct2), :).('kr');
    end
end

% PROCESSING
% Input Data: Error Bound; Max Iteration Steps
% Processes: Mesh Adaption; Numerical Method; Equation Solver
% MAKE_VEFIE_ELEMENTS: CREATES G_VECTOR, D AND V FROM (I+GD)E=V
% CREATION OF ELEMENTS FOR VEFIE: (I+GD)E=V (Volume-equivalent Electric Field Intergral Equation)
V = zeros(basis_counter, 1);
D = zeros(basis_counter, 1); % Diagonal contrast matrix stored as a vector
fprintf('\nStart creation of all %d ellements of V and D\n \n', basis_counter)
% start1 = tic;

% Incident Field
for ct1 = 1:basis_counter
    % V(ct1) = exp(-1i*kr(1)*rho(ct1)*cos(the_phi(ct1)));
    V(ct1) = exp(-1i*vacuum_kr*rho(ct1)*cos(the_phi(ct1)));
    D(ct1, 1) = (basis_wave_number(ct1, 1) * basis_wave_number(ct1, 1) - vacuum_kr * vacuum_kr); % contrast function
end
% Time_creation_all_elements_from_V_and_D = toc(start1);

x = real(position(1:N));
y = imag(position(1:N:N*M));

% x = flip(x);
% x = rot90(x, 2);
% y = rot90(y, 2);

% SOLVE_WITH_FFT_AND_REDUCEDCG
% Inclsuion of matrix pre-conditioners to help control for condition
% number may be required.

%%% Reduced CG algoritm with FFT, solving Z*E=V with Z =I+GD %%%
clear r p solver_error icnt a b;
Rfo = logical(D); % Reduced foreward operator
Vred = Rfo .* V; % create V reduced
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nStart creation of all %d ellements of G\n \n', basis_counter)
G_vector = zeros(basis_counter, 1); % BMT dense matrix, stored as a vector
% start2 = tic;
for ct1 = 1:basis_counter
    if (mod(ct1, 100) == 0)
        fprintf(1, '%dth element \n', ct1);
    end
    R_mn2 = abs(position(ct1)-position(1));
    if ct1 == 1
        % G_vector(ct1,1)=(1i/4.0)*((2.0*pi*equiv_a/kr(1))*(besselj(1,kr(1)*equiv_a)-1i*bessely(1,kr(1)*equiv_a))-4.0*1i/(kr(1)*kr(1)));
        % G_vector(ct1, 1) = (1i / 4.0) * ((2.0 * pi * equiv_a / kr(1)) * besselh(1, 2, kr(1)*equiv_a) - 4.0 * 1i / (kr(1) * kr(1)));
        G_vector(ct1, 1) = (1i / 4.0) * ((2.0 * pi * equiv_a / vacuum_kr) * besselh(1, 2, vacuum_kr*equiv_a) - 4.0 * 1i / (vacuum_kr * vacuum_kr));
    else
        % G_vector(ct1,1)=(1i/4.0)*(2.0*pi*equiv_a/kr(1))*besselj(1,kr(1)*equiv_a)*(besselj(0,kr(1)*R_mn2)-1i*bessely(0,kr(1)*R_mn2));
        % G_vector(ct1, 1) = (1i / 4.0) * (2.0 * pi * equiv_a / kr(1)) * besselj(1, kr(1)*equiv_a) * besselh(0, 2, kr(1)*R_mn2);
        G_vector(ct1, 1) = (1i / 4.0) * (2.0 * pi * equiv_a / vacuum_kr) * besselj(1, vacuum_kr*equiv_a) * besselh(0, 2, vacuum_kr*R_mn2);
    end
end
% Time_creation_all_elements_from_G = toc(start2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solver
Ered = zeros(basis_counter, 1); % guess
% Z*E - V (= error)
r = Rfo .* Ered + Rfo .* BMT_FFT(G_vector.', D.*Ered, N) - Vred;
p = -(Rfo .* r + conj(D) .* (BMT_FFT(conj(G_vector.'), Rfo.*r, N))); % -Z'*r

fprintf('\nStart reduced CG iteration with 2D FFT\n')
solver_error = abs(r'*r); % definition of error

Reduced_iteration_error = zeros(1, 1);
icnt = 0; % iteration counter
% start3 = tic;
while (solver_error > input_solver_tol) && (icnt <= basis_counter)
    icnt = icnt + 1;
    if (mod(icnt, 50) == 0)
        fprintf(1, '%dth iteration Red \n', icnt);
    end
    a = (norm(Rfo.*r+conj(D).*BMT_FFT(conj(G_vector.'), Rfo.*r, N)) / norm(Rfo.*p+Rfo.*(BMT_FFT(G_vector.', D.*p, N))))^2; %(norm(Z'*r)^2)/(norm(Z*p)^2);
    Ered = Ered + a * p;
    r_old = r;
    r = r + a * (Rfo .* p + Rfo .* (BMT_FFT(G_vector.', D.*p, N))); % r = r + a*z*p
    b = (norm(Rfo.*r+conj(D).*BMT_FFT(conj(G_vector.'), Rfo.*r, N)) / norm(Rfo.*r_old+conj(D).*BMT_FFT(conj(G_vector.'), Rfo.*r_old, N)))^2; %b = (norm(Z'*r)^2) /(norm(Z'*r_old)^2);
    p = -(Rfo .* r + conj(D) .* BMT_FFT(conj(G_vector.'), Rfo.*r, N)) + b * p; % p=-Z'*r+b*p
    solver_error = abs(r'*r);
    Reduced_iteration_error(icnt, 1) = abs(r'*r);
end
% time_Red = toc(start3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POST-PROCESSING
% Input Data: Evaluation Locality for diagrams, colour plots etc.
% Processes: Optimisation; Further Modelling (Lumped Parameters); Approximation of local field qualities; Field Coupling)
% PLOT_RESULTS
% Write variable information to Command Window and plot reults
% WRITE INFORMATION TO COMMAND WINDOW
fprintf('\nUsed frequency of incomming wave = %d Hz \n', input_carrier_frequency)
% fprintf('\nRelative Epsilon: \nconcrete = %d \nwood = %d \nglass = %d \n', epsilonrd, epsilonrdw, epsilonrdg)
% fprintf('\nWave Lengths: \nfree space %d meter\nconcrete %d meter\nwood %d meter \nglass %d meter\n', lambda0, lambda_d, lambda_w, lambda_g)
% fprintf('\n \nSize of field of interst is %d meter by %d meter \n', length_x_side, length_y_side)
fprintf('Discretizised space is %d grids by %d grids\n', N, M)
fprintf('with grid size %d meter by %d meter', delta_x, delta_y)
fprintf('So basis counter goes to %d \n', basis_counter)
fprintf('\nNumber of unknowns in Z is than %d\n', N*M*N*M)
fprintf('Number of reduced unknowns is %d\n', sum(Rfo)*N*M)
fprintf('with %.2f%% percent of the is filled by contrast \n', sum(Rfo)/(N * M)*100)
fprintf('\nCG iteration error tollerance = %d \n', input_solver_tol)
% fprintf('Duration of reduced CG iteration = %d seconds \n', time_Red)

% CONVERT SOLUTIONS ON 2D GRID, FOR 3D PLOTS
vec2matSimulationVred = zeros(M, N);
for i = 1:M %without vec2mat:
    vec2matSimulationVred(i, :) = Vred((i - 1)*N+1:i*N);
end
vec2matSimulationEred = zeros(M, N);
for i = 1:M %without vec2mat:
    vec2matSimulationEred(i, :) = Ered((i - 1)*N+1:i*N);
end

Vred_2D = vec2matSimulationVred;
Ered_2D = vec2matSimulationEred;
ScatRed_2D = Ered_2D - Vred_2D;

% CREATE ALL THE PLOTS
figure
set(gcf, 'units', 'normalized', 'outerposition', [0, 0, 1, 1])
subplot(1, 2, 1)
imagesc(ind2rgb(image_resize, materials_master.('map')), 'XData', 1/2, 'YData', 1/2)
axis tight
title('Material Configuration After Simulation')
legend
hold on
L = plot(ones(height(materials_master(:, 2))), 'LineStyle', 'none', 'marker', 's', 'visible', 'on');
set(L, {'MarkerFaceColor'}, markerColor, {'MarkerEdgeColor'}, markerColor);
colormap(materials_master.('map'))
legend(materials_master.('name'))

subplot(1, 2, 2)
semilogy(Reduced_iteration_error, 'r')
title('Iteration Error')
legend('Reduced')
% TBC: The axis for the iteration should be fixed so that improvement can
% be compared easily. This will be implemented when there is more than one
% iterative approach being pursued.
axis tight
ylabel('Number of iteration')
xlabel('Error')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3D plots of real/abs of total/scatterd/incomming
figure
set(gcf, 'units', 'normalized', 'outerposition', [0, 0, 1, 1])

subplot(2, 3, 1)
surf(x, y, real(Vred_2D));
% view(2)
view(0, 270)
shading interp
title('Reduced Incoming Wave Part Real');
xlabel('x (meters)')
ylabel('y (meter)')
axis tight

subplot(2, 3, 2)
surf(x, y, real(ScatRed_2D));
% view(2)
view(0, 270)
shading interp
title('Reduced Scattered Field Part Real');
xlabel('x (meters)')
ylabel('y (meter)')
axis tight

subplot(2, 3, 3)
surf(x, y, real(Ered_2D))
% view(2)
view(0, 270)
shading interp
title('Reduced Total Field Part Real');
xlabel('x (meters)')
ylabel('y (meter)')
axis tight

subplot(2, 3, 4)
surf(x, y, abs(Vred_2D));
% view(2)
view(0, 270)
shading interp
title('Reduced Incoming Wave Part Absolute');
xlabel('x (meters)')
ylabel('y (meter)')
axis tight

subplot(2, 3, 5)
surf(x, y, abs(ScatRed_2D));
% view(2)
view(0, 270)
shading interp
title('Reduced Scattered Field Part Absolute');
xlabel('x (meters)')
ylabel('y (meter)')
axis tight

subplot(2, 3, 6)
surf(x, y, abs(Ered_2D))
% view(2)
view(0, 270)
shading interp
title('Reduced Total Field Part Absolute');
xlabel('x (meters)')
ylabel('y (meter)')
axis tight

total_time_ran = toc(total_time);
