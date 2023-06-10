%%%     2D Scattering Solver using 2D FFT and reduced CG algorithm     %%%
clear;
close all;
clc;
format compact;
% PRE-PROCESSING
% Input Data: Geometry; Materials; Boundary Conditions
% Processes: Discretisation; Approximation; Parametrisation; Coupling (Fields; Geomtery; Circuits; Motion; Methods)
% TBC: Shift all variables to init programme for simpler manipulation.
input = init();

% Hard-coded Variables
epsilon0 = 8.854e-12;
mu0 = 4.0 * pi * 1.0e-7;
omega = 2.0 * pi * input.f; % angular frequency (rad/s)

% SET_MATERIAL_VALUES
% Due to nature of study, the source must always sit in vacuum for this
% programme. Expanding on this may form part of future work, however, it is
% expected that assuming a small region of vacuum (air) surrounds the
% source this issue will be overcome.

% TBC: pulling in material by frequency should improve the assignment
% efficiency later in programme. There is an if condition for assignment
% that could be improved by assigning based on order of frequency. Small
% changes in this part of the code would be required to achieve this idea.
% material_freq = [material_id,histc(object(:),material_id)];

% Order of this list should not change unless the numeric identifiers in the imported geometry also reflect such changes.
% materials = {"vacuum", "concrete", "wood", "glass", "brick", "plasterboard", "ceiling-board", "chipboard", "floorboard", "metal"};
materials = table2array(input.materials_master(:,2));
hex = table2array(input.materials_master(:,3));
map = sscanf(hex', '#%2x%2x%2x', [3, size(hex, 1)]).' / 255;
markerColor = mat2cell(map, ones(1, height(input.materials_master(:,2))), 3);

% IMPORT MODEL WITH MATERIAL DESCRIPTIONS
if strcmp(input.object_gen,'Yes') == 1
    [~,materials_present] = ismember(input.object_materials,input.materials_master(:,2));
    material_id = unique(materials_present, 'sorted');
else
    object = readmatrix([input.directory_geom input.object_name]);
    % object = uint32(imread([input.directory_geom input.object_name]));
    [M_geo, N_geo]=size(object);
    material_id = unique(object, 'sorted');

    % Remap object so materials increment from 1 with no gaps. NO TOO SLOW
end

% object1 = uint16(imread([input.directory_geom 'factorio-furnace-layout_uint16.png']));
% [M_geo1, N_geo1]=size(object1);
% material_id1 = unique(object1, 'sorted');
% min(material_id1)
% max(material_id1)
% histogram(object1,10)
% imshow(object1)

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
    % [epsilonr(k), sigma(k), epsilonr_complex(k)] = buildingMaterialPermittivity(materials{material_id(k)}, input.f);
    [epsilonr(k), sigma(k), epsilonr_complex(k)] = buildingMaterialPermittivity(materials(material_id(k)), input.f);

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
    if lambda_smallest > cr(k) / input.f
        lambda_smallest = cr(k) / input.f;
    end
end

% DISCRETISE_DOMAIN: SPECIFY SPACE GRID
% 1) discretisize field of interest, imag = Yaxes, real = Xaxes
centre = 0.0 + 0.0 * 1i;
start_pt = centre - 0.5 * input.length_x_side - 0.5 * input.length_y_side * 1i;

N = floor(input.length_x_side/(abs(lambda_smallest) / input.disc_per_lambda)); % force N = multp 4
% if strcmp(input.object_gen,'Yes') ~= 1
    % if N < N_geo
    %     N = N_geo;
    % end
% end
fourth_of_N = ceil(N/4);
while (mod(N, fourth_of_N) ~= 0)
    N = N + 1;
end
delta_x = input.length_x_side / N;

M = floor(input.length_y_side/(delta_x)); % force M = multp 4, size dy near dx
% if strcmp(input.object_gen,'Yes') ~= 1
    % if M < M_geo
    %     M = M_geo;
    % end
% end
fourth_of_M = ceil(M/4);
while (mod(M, fourth_of_M) ~= 0)
    M = M + 1;
end
delta_y = input.length_y_side / M;

equiv_a = sqrt(delta_x*delta_y/pi);

fprintf('\n \nSize of field of interst is %d meter by %d meter \n', input.length_x_side, input.length_y_side)
fprintf('Discretizised space is %d grids by %d grids\n', N, M)
fprintf('with grid size %d meter by %d meter \n', delta_x, delta_y)

% IMPORT MODEL WITH MATERIAL DESCRIPTIONS
if strcmp(input.object_gen,'Yes') == 1
    object = object_generator(material_id,M,N,delta_x,delta_y,input);
end

% VISUALISE
figure
imagesc(object, 'XData', 1/2, 'YData', 1/2)
title('Material Configuration Before Scaling for f')
legend
set(gcf, 'units', 'normalized', 'outerposition', [0, 0, 1, 1])
hold on
L = plot(ones(height(input.materials_master(:,2))), 'LineStyle', 'none', 'marker', 's', 'visible', 'on');
set(L, {'MarkerFaceColor'}, markerColor, {'MarkerEdgeColor'}, markerColor);
colormap(map)
legend(materials)

% RESCALE OBJECT FOR SPECIFIC FREQUENCY
% TBC: This is required for standardised input into CNN
% Sophisticated Book uses 128x128
% Imported geometry will sit at resolution required to depict physical
% geometry of object. Then this discretization needs to be checked that it
% is sufficient to depict the electromagnetic materials of the object. If
% the discretisation is enough already, then it is maintained. If they
% discretization needs to be incresed then the imported geometry will be
% sliced up at a higher resolution. Ultimately, the final resolution of the
% exported geometry & output field needs to be at 256x256.
% if M > size(object,1)
% M = 128;
% N = 128;
% object = imresize(object, [M, N], "nearest");
% end



% Visualise imported object after rescaling for f.
figure
imagesc(object, 'XData', 1/2, 'YData', 1/2)
title('Material Configuration After Scaling for f')
legend
set(gcf, 'units', 'normalized', 'outerposition', [0, 0, 1, 1])
hold on
L = plot(ones(height(input.materials_master(:,2))), 'LineStyle', 'none', 'marker', 's', 'visible', 'on');
set(L, {'MarkerFaceColor'}, markerColor, {'MarkerEdgeColor'}, markerColor);
colormap(map)
legend(materials)

% TBC: Label axis with dual labels: length in meters; discretisation count.
% ax1 = gca;
% set(ax1,'XAxisLocation','top')
% set(ax1,'YAxisLocation','left')
% ax1.XLabel.String = 'Discretization Count (#)';
% ax1.YLabel.String = 'Discretization Count (#)';
% ax1.XGrid = 'on';
% ax1.YGrid = 'on';
% ax2 = axes('Position',get(ax1,'Position'),'XAxisLocation','bottom','YAxisLocation','right','Color','none','XColor','k','YColor','k');
% set(ax2,'XAxisLocation','bottom')
% set(ax2,'YAxisLocation','right')
% ax2.XLabel.String = 'Distance (meters)';
% ax2.YLabel.String = 'Distance (meters)';
% % ax2.XTick = 0:delta_x:length_x_side;
% ax2.YTick = [0:1:N]*delta_y;
% ax2.XTickLabel = 0:delta_x:length_x_side;
% ax2.YTickLabel = [0:1:N]*delta_y;
% ax2.XGrid = 'on';
% ax2.YGrid = 'on';

length_y_side = delta_y * M;
length_x_side = delta_x * N;

% % Plot object
% figure
% surf((1:N)*delta_x, (1:M)*delta_y, object)
% view(2)
% shading interp
% xlabel('Distance in meters')
% ylabel('Distance in meters')

% SPECIFY_MATERIALS: POSITION, K AND RHO, FOR EACH BASIS COUNTER
% INTERIOR SPECIFICATION: position, phi, k, and rho for each number of position (basis counter)
basis_counter = 0;
basis_wave_number(1:N*N, 1) = kr(1);
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

        basis_wave_number(basis_counter) = kr(object(ct1, ct2));
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
fprintf('\nUsed frequency of incomming wave = %d Hz \n', input.f)
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
L = plot(ones(height(input.materials_master(:,2))), 'LineStyle', 'none', 'marker', 's', 'visible', 'on');
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