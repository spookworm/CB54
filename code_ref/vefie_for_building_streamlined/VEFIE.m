%%%     2D Scattering Solver using 2D FFT and reduced CG algorithm     %%%
clear;
close all;
clc;
format compact;

% Inputs
disc_per_lambda = 10; % chosen accuracy
directory_geom = './Geometry/';
object_name = 'object_mp_landscape_empty.txt';
f = 60e6; % frequency (Hz)

% Hard-coded Variables
epsilon0 = 8.854e-12;
mu0 = 4.0 * pi * 1.0e-7;
omega = 2.0 * pi * f; % angular frequency (rad/s)

% Pass list of master materials with associated visualisation colourings to be used in scene to the generator.
% Order of this list should not change unless the numeric identifiers in the imported geometry also reflect such changes.
Name = {'vacuum','concrete','wood','glass','brick','plasterboard','ceiling-board','chipboard','floorboard','metal'}';
hex = ['#FDFF00'; '#A7A9AB'; '#D08931'; '#B9E8E9'; '#ED774C'; '#EFEFEE'; '#F4F1DB'; '#C09B53'; '#7A5409'; '#909090'];
Number = 1:1:length(Name);
materials_master = table(Number',Name,hex);
materials = table2array(materials_master(:, 2));
hex = table2array(materials_master(:, 3));
map = sscanf(hex', '#%2x%2x%2x', [3, size(hex, 1)]).' / 255;
markerColor = mat2cell(map, ones(1, height(materials_master(:, 2))), 3);

object = readmatrix([directory_geom, object_name]);
material_id = unique(object, 'sorted');

epsilonr = ones(length(material_id), 1);
sigma = ones(length(material_id), 1);
epsilonr_complex = ones(length(material_id), 1);
mur = ones(length(material_id), 1);
mur_complex = ones(length(material_id), 1);
cr = ones(length(material_id), 1);
kr = ones(length(material_id), 1);
lambdar = ones(length(material_id), 1);
lambda_smallest = realmax;
for k = 1:length(material_id)
    % TBC: Expand possible materials list
    % materials = readtable('./Geometry/pmt-hps-dielectric-constant-table.xlsx');
    % materials(materials.("Name")=="Acetal","epsilonrd_pt")

    % Set epsilon and sigma based on internal MATLAB function values taken
    % from international standard.
    % [vacuum_epsilon, vacuum_sigma, vacuum_epsilonr_complex] = buildingMaterialPermittivity('vacuum', 60e6);
    % [concrete_epsilon, concrete_sigma, concrete_epsilonr_complex] = buildingMaterialPermittivity('concrete', 60e6);
    % [glass_epsilon, glass_sigma, glass_epsilonr_complex] = buildingMaterialPermittivity('glass', 60e6);
    % [wood_epsilon, wood_sigma, wood_epsilonr_complex] = buildingMaterialPermittivity('glass', 60e6);
    [epsilonr(k), sigma(k), epsilonr_complex(k)] = buildingMaterialPermittivity(materials(material_id(k)), f);

    % Set mu.
    mur(k) = 1.0;
    mur_complex(k) = 1.0 - (0.0 * 1i);

    % Set c in material.
    cr(k) = 1.0 / sqrt(epsilonr(k)*epsilon0*mur(k)*mu0);
    % TBC: Expanded to complex case without impact?
    % cr(k)=1.0 / sqrt(epsilonr_complex(k)*epsilon0*mur(k)*mu0);

    % Set k in material.
    % TBC: Expand to complex case.
    kr(k) = omega * sqrt(epsilonr(k)*epsilon0*mur(k)*mu0);
    % kr(k) = omega * sqrt(epsilonr_complex(k)*epsilon0*mur(k)*mu0);

    % Choose the smallest lambda (wavelength) of all materials in the configuration.
    if lambda_smallest > cr(k) / f
        lambda_smallest = cr(k) / f;
    end
end

% This assumes that the scale of the imported geometry is 1m per cell in both directions.
% As a result, the room in the original code will not work properly. Use
% the old code to work with that room.
[M_geo, N_geo] = size(object); % M~x
length_x_side = M_geo; % in meters
length_y_side = N_geo; % in meters

% decision to be made here is which one should be calculated first to shortcut the mod loop
% this decision is based on which  length_?_side is BIGGER
if length_x_side > length_y_side
    N = floor(length_x_side/(abs(lambda_smallest) / disc_per_lambda)); % force N = multp 4
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
    M = floor(length_y_side/(abs(lambda_smallest) / disc_per_lambda)); % force N = multp 4
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

% VISUALISE
figure
imagesc(object, 'XData', 1/2, 'YData', 1/2)
title('Material Configuration Before Scaling for f')
legend
set(gcf, 'units', 'normalized', 'outerposition', [0, 0, 1, 1])
hold on
L = plot(ones(height(materials_master(:, 2))), 'LineStyle', 'none', 'marker', 's', 'visible', 'on');
set(L, {'MarkerFaceColor'}, markerColor, {'MarkerEdgeColor'}, markerColor);
colormap(map)
legend(materials)

% Need to change geometry resolution here if frequency is lower scale
% This doesn't look great, is there a better way to resize?
% Also need to check how upscaling and downscaling perform.
object = imresize(object, [M, N], "nearest");

% Visualise imported object after rescaling for f.
figure
imagesc(object, 'XData', 1/2, 'YData', 1/2)
title('Material Configuration After Scaling for f')
legend
set(gcf, 'units', 'normalized', 'outerposition', [0, 0, 1, 1])
hold on
L = plot(ones(height(materials_master(:, 2))), 'LineStyle', 'none', 'marker', 's', 'visible', 'on');
set(L, {'MarkerFaceColor'}, markerColor, {'MarkerEdgeColor'}, markerColor);
colormap(map)
legend(materials)

centre = 0.0 + 0.0 * 1i;
start_pt = centre - 0.5 * length_x_side - 0.5 * length_y_side * 1i;
fprintf('\n \nSize of field of interst is %d meter by %d meter \n', length_x_side, length_y_side)
fprintf('Discretizised space is %d grids by %d grids\n', N, M)
fprintf('with grid size %d meter by %d meter \n', delta_x, delta_y)

% SPECIFY_MATERIALS: POSITION, K AND RHO, FOR EACH BASIS COUNTER
% INTERIOR SPECIFICATION: position, phi, k, and rho for each number of position (basis counter)
basis_counter = 0;
% This is wrong I think unless there is a vacuum in the scene and that it
% is indexed as one. Would it not be better to assign based on most likely
% material occurence? See the other occurances too. The incident waves etc.
% assume that there is a vacuum. What if there is no vacuum? What are the
% contrasts in contrast with then? Is the inclusion of a single cell of
% vacuum enough to make the thing work?
basis_wave_number(1:M*N, 1) = kr(1);
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

        % basis_wave_number(basis_counter) = kr(object(ct1, ct2));
        basis_wave_number(basis_counter) = kr(material_id==object(ct1, ct2));
    end
end

% PROCESSING
% Input Data: Error Bound; Max Iteration Steps
% Processes: Mesh Adaption; Numerical Method; Equation Solver
% MAKE_VEFIE_ELEMENTS: CREATES G_VECTOR, D AND V FROM (I+GD)E=V
% CREATION OF ELEMENTS FOR VEFIE: (I+GD)E=V (Volume-equivalent Electric Field Intergral Equation)
V = zeros(basis_counter, 1);
D = zeros(basis_counter, 1); % Diagonal contrast matrix stored as a vector
G_vector = zeros(basis_counter, 1); % BMT dense matrix, stored as a vector
fprintf('\nStart creation of all %d ellements of G,V and D \n \n', basis_counter)
start1 = tic;
% Incident Field
for ct1 = 1:basis_counter
    V(ct1) = exp(-1i*kr(1)*rho(ct1)*cos(the_phi(ct1)));
    D(ct1, 1) = (basis_wave_number(ct1, 1) * basis_wave_number(ct1, 1) - kr(1) * kr(1)); % contrast function
end
for ct1 = 1:basis_counter
    if (mod(ct1, 100) == 0)
        fprintf(1, '%dth element \n', ct1);
    end
    R_mn2 = abs(position(ct1)-position(1));
    if ct1 == 1
        %         G_vector(ct1,1)=(1i/4.0)*((2.0*pi*equiv_a/kr(1))*(besselj(1,kr(1)*equiv_a)-1i*bessely(1,kr(1)*equiv_a))-4.0*1i/(kr(1)*kr(1)));
        G_vector(ct1, 1) = (1i / 4.0) * ((2.0 * pi * equiv_a / kr(1)) * besselh(1, 2, kr(1)*equiv_a) - 4.0 * 1i / (kr(1) * kr(1)));
    else
        %         G_vector(ct1,1)=(1i/4.0)*(2.0*pi*equiv_a/kr(1))*besselj(1,kr(1)*equiv_a)*(besselj(0,kr(1)*R_mn2)-1i*bessely(0,kr(1)*R_mn2));
        G_vector(ct1, 1) = (1i / 4.0) * (2.0 * pi * equiv_a / kr(1)) * besselj(1, kr(1)*equiv_a) * besselh(0, 2, kr(1)*R_mn2);
    end
end
Time_creation_all_elements_from_G_V_and_D = toc(start1);

% SOLVE_WITH_FFT_AND_REDUCEDCG
% Inclsuion of matrix pre-conditioners to help control for condition
% number may be required.

%%% Reduced CG algoritm with FFT, solving Z*E=V with Z =I+GD %%%
clear r p error icnt a b;
Rfo = logical(D); % Reduced foreward operator
Vred = Rfo .* V; % create V reduced
Ered = zeros(basis_counter, 1); % guess
r = Rfo .* Ered + Rfo .* BMT_FFT(G_vector.', D.*Ered, N) - Vred; % Z*E - V (= error)
p = -(Rfo .* r + conj(D) .* (BMT_FFT(conj(G_vector.'), Rfo.*r, N))); % -Z'*r
tol = 1e-3; % error tolerance
icnt = 0; % iteration counter
error = abs(r'*r); % definition of error
Reduced_iteration_error = zeros(1, 1);
fprintf('\nStart reduced CG iteration with 2D FFT\n')
start3 = tic;
while (error > tol) && (icnt <= basis_counter)
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
    error = abs(r'*r);
    Reduced_iteration_error(icnt, 1) = abs(r'*r);
end
time_Red = toc(start3);

% POST-PROCESSING
% Input Data: Evaluation Locality for diagrams, colour plots etc.
% Processes: Optimisation; Further Modelling (Lumped Parameters); Approximation of local field qualities; Field Coupling)
% PLOT_RESULTS
% Write variable information to Command Window and plot reults
% WRITE INFORMATION TO COMMAND WINDOW
fprintf('\nUsed frequency of incomming wave = %d Hz \n', f)
% fprintf('\nRelative Epsilon: \nconcrete = %d \nwood = %d \nglass = %d \n', epsilonrd, epsilonrdw, epsilonrdg)
% fprintf('\nWave Lengths: \nfree space %d meter\nconcrete %d meter\nwood %d meter \nglass %d meter\n', lambda0, lambda_d, lambda_w, lambda_g)
% fprintf('\n \nSize of field of interst is %d meter by %d meter \n', length_x_side, length_y_side)
fprintf('Discretizised space is %d grids by %d grids\n', N, M)
fprintf('with grid size %d meter by %d meter', delta_x, delta_y)
fprintf('So basis counter goes to %d \n', basis_counter)
fprintf('\nNumber of unknowns in Z is than %d\n', N*M*N*M)
fprintf('Number of reduced unknowns is %d\n', sum(Rfo)*N*M)
fprintf('with %d percent of the is filled by contrast \n', sum(Rfo)/(N * M)*100)
fprintf('\nCG iteration error tollerance = %d \n', tol)
fprintf('Duration of reduced CG iteration = %d seconds \n', time_Red)
% CONVERT SOLUTIONS ON 2D GRID, FOR 3D PLOTS
vec2matSimulationEred = zeros(M, N);
vec2matSimulationVred = zeros(M, N);
for i = 1:M %without vec2mat:
    vec2matSimulationEred(i, :) = Ered((i - 1)*N+1:i*N);
    vec2matSimulationVred(i, :) = Vred((i - 1)*N+1:i*N);
end
Ered_2D = vec2matSimulationEred;
Vred_2D = vec2matSimulationVred;
ScatRed_2D = Ered_2D - Vred_2D;

x = real(position(1:N));
y = imag(position(1:N:N*M));

x = flip(x);
x = rot90(x, 2);
y = rot90(y, 2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CREATE ALL THE PLOTS
info_index = 0;
figure
set(gcf, 'units', 'normalized', 'outerposition', [0, 0, 1, 1])
subplot(1, 2, 1)
imagesc(object, 'XData', 1/2, 'YData', 1/2)
axis tight
title('Material Configuration After Simulation')
legend
hold on
L = plot(ones(height(materials_master(:, 2))), 'LineStyle', 'none', 'marker', 's', 'visible', 'on');
set(L, {'MarkerFaceColor'}, markerColor, {'MarkerEdgeColor'}, markerColor);
colormap(map)
legend(materials)

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
view(2)
shading interp
title('Reduced Incoming Wave Part Real');
xlabel('x (meters)')
ylabel('y (meter)')
axis tight

subplot(2, 3, 2)
surf(x, y, real(ScatRed_2D));
view(2)
shading interp
title('Reduced Scattered Field Part Real');
xlabel('x (meters)')
ylabel('y (meter)')
axis tight

subplot(2, 3, 3)
surf(x, y, real(Ered_2D))
view(2)
shading interp
title('Reduced Total Field Part Real');
xlabel('x (meters)')
ylabel('y (meter)')
axis tight

subplot(2, 3, 4)
surf(x, y, abs(Vred_2D));
view(2)
shading interp
title('Reduced Incoming Wave Part Absolute');
xlabel('x (meters)')
ylabel('y (meter)')
axis tight

subplot(2, 3, 5)
surf(x, y, abs(ScatRed_2D));
view(2)
shading interp
title('Reduced Scattered Field Part Absolute');
xlabel('x (meters)')
ylabel('y (meter)')
axis tight

subplot(2, 3, 6)
surf(x, y, abs(Ered_2D))
view(2)
shading interp
title('Reduced Total Field Part Absolute');
xlabel('x (meters)')
ylabel('y (meter)')
axis tight