import random
import numpy as np
import os
import sys
import time
from PIL import Image
from IPython import get_ipython
import matplotlib.pyplot as plt
import json
import pandas as pd
import struct
import matplotlib.colors as mcolors
from scipy.special import jv, yv

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=20)

random.seed(42)

# 2D Scattering Solver using 2D FFT and reduced CG algorithm
total_time = time.time()

# Inputs
path_geo = './Geometry/'
path_lut = './lut/materials.json'
object_name = 'placeholder.png'

# chosen accuracy
input_disc_per_lambda = 10
# frequency (Hz)
input_carrier_frequency = 60e6
# error tolerance
input_solver_tol = 1e-1

# Hard-coded Variables
epsilon0 = 8.854187817e-12
mu0 = 4.0 * np.pi * 1.0e-7
# angular frequency (rad/s)
angular_frequency = 2.0 * np.pi * input_carrier_frequency

image_object = Image.open(path_geo + object_name).convert('L')
unique_integers = np.sort(np.unique(image_object))


# % Pass list of master materials with associated visualisation colourings to be used in scene to the generator.
# % Order of this list should not change unless the numeric identifiers in the imported geometry also reflect such changes.
with open(path_lut, 'rb') as fid:
    raw = fid.read()
    str = raw.decode('utf-8')

materials_master = pd.DataFrame(json.loads(str))

# % Subset on entries in the scene.
materials_master = materials_master[materials_master['uint8'].isin(unique_integers)]
materials_master = materials_master.reset_index(drop=True)

# materials_master.('map') = sscanf(char(materials_master.('HEX'))', '#%2x%2x%2x', [3, size(materials_master.('HEX'), 1)]).' / 255;
# Convert HEX to RGB
materials_master['RGB'] = materials_master['HEX'].apply(lambda x: mcolors.hex2color(x))

# % Switch the source dictionary so only the required names are printed
# markerColor = mat2cell(materials_master.('map'), ones(1, height(materials_master.('name'))), 3);
markerColor = pd.DataFrame(materials_master['RGB'])


def buildingMaterialPermittivity(mtls, fc):
    mtlLib = {
        # Name                  a       b       c       d       range
        'vacuum':                [1.0,     0,       0.0,     0.0,     [1e6,  1e11]],
        'concrete':              [5.31,    0,       0.0326,  0.8095,  [1e9,  1e11]],
        'brick':                 [3.75,    0,       0.038,   0,       [1e9,  1e10]],
        'plasterboard':          [2.94,    0,       0.0116,  0.7076,  [1e9,  1e11]],
        'wood':                  [1.99,    0,       0.0047,  1.0718,  [1e6,  1e11]],
        'glass':                 [6.27,    0,       0.0043,  1.1925,  [1e8,  1e11]],
        'ceiling-board':         [1.50,    0,       0.0005,  1.1634,  [1e9,  1e11]],
        'chipboard':             [2.58,    0,       0.0217,  0.7800,  [1e9,  1e11]],
        'floorboard':            [3.66,    0,       0.0044,  1.3515,  [5e10, 1e11]],
        'metal':                 [1.0,     0,       1e7,     0,       [1e9,  1e11]],
        'very-dry-ground':       [3.0,     0,       0.00015, 2.52,    [1e9,  1e10]],
        'medium-dry-ground':     [15.0,    -0.1,    0.035,   1.63,    [1e9,  1e10]],
        'wet-ground':            [30.0,    -0.4,    0.15,    1.30,    [1e9,  1e10]]
    }
    fcGHz = input_carrier_frequency/1e9
    value = mtlLib.get(mtls)
    if value:
        # epsilon = [mtlParams{libIdx, 1}] .* (fcGHz.^[mtlParams{libIdx, 2}]);
        epsilon = value[0] * fcGHz**value[1]
        # sigma = [mtlParams{libIdx, 3}] .* (fcGHz.^[mtlParams{libIdx, 4}]);
        sigma = value[2] * fcGHz**value[3]
        # complexEpsilon = epsilon - 1i*sigma/(2*pi*fc*epsilon0);
        complexEpsilon = epsilon - 1j*sigma/(2*np.pi*fc*epsilon0)
        return epsilon, sigma, complexEpsilon
    return None


materials_master['epsilonr'] = None
materials_master['sigma'] = None
materials_master['epsilonr_complex'] = None
materials_master['mur'] = None
materials_master['mur_complex'] = 1.0 - (0.0 * 1j)
materials_master['cr'] = None
materials_master['cr_complex'] = None
materials_master['kr'] = None
materials_master['kr_complex'] = None
for k in range(0, len(unique_integers+1)):
    materials_master.loc[k, 'epsilonr'], materials_master.loc[k, 'sigma'], materials_master.loc[k, 'epsilonr_complex'] = buildingMaterialPermittivity(materials_master['name'][k], input_carrier_frequency)
    materials_master.loc[k, 'mur'] = 1.0
    materials_master.loc[k, 'mur_complex'] = 1.0 - (0.0 * 1j)
    materials_master.loc[k, 'cr'] = 1.0 / np.sqrt(materials_master.loc[k, 'epsilonr'] * epsilon0 * materials_master.loc[k, 'mur'] * mu0)
    materials_master.loc[k, 'cr_complex'] = 1.0 / np.sqrt(materials_master.loc[k, 'epsilonr_complex'] * epsilon0 * materials_master.loc[k, 'mur'] * mu0)
    materials_master.loc[k, 'kr'] = angular_frequency * np.sqrt(materials_master.loc[k, 'epsilonr'] * epsilon0 * materials_master.loc[k, 'mur'] * mu0)
    materials_master.loc[k, 'kr_complex'] = angular_frequency * np.sqrt(materials_master.loc[k, 'epsilonr_complex'] * epsilon0 * materials_master.loc[k, 'mur'] * mu0)

# % Assign vacuum for speed-up.
vacuum_kr = materials_master.loc[materials_master['name'] == 'vacuum', 'kr'].values[0]


# def image_object_render(image_object, materials_master, markerColor, title_str):
#     image_object_render = plt.cm.colors.ListedColormap(materials_master['RGB'])(image_object)
#     plt.figure()
#     plt.imshow(image_object_render, extent=(0.5, 1.5, 0.5, 1.5))
#     plt.title(title_str)
#     plt.legend()
#     plt.gcf().set_size_inches(10, 10)
#     plt.set_cmap(materials_master['RGB'])
#     plt.legend(materials_master['name'])
#     plt.show()


# # % VISUALISE
# title_str = 'Material Configuration Before Scaling for f'
# image_object_render(image_object, materials_master, markerColor, title_str)

# % This assumes that the scale of the imported geometry is 1m per cell in both directions.
# % As a result, the room in the original code will not work properly. Use
# % the old code to work with that room.
# [length_y_side, length_x_side] = size(image_object); % M~x in meters

# M~x in meters
length_y_side, length_x_side = np.shape(image_object)

# % Choose the smallest lambda (wavelength) of all materials in the configuration.
lambda_smallest = np.min(materials_master['cr']) / input_carrier_frequency

# % decision to be made here is which one should be calculated first to shortcut the mod loop
# % this decision is based on which  length_?_side is BIGGER
if length_x_side > length_y_side:
    # force N = multp 4
    N = np.floor(length_x_side/(np.abs(lambda_smallest) / input_disc_per_lambda))
    fourth_of_N = np.ceil(N/4)
    while (np.mod(N, fourth_of_N) != 0):
        N = N + 1
    delta_x = length_x_side / N
    # force M = multp 4, size dy near dx
    M = np.floor(length_y_side/(delta_x))
    fourth_of_M = np.ceil(M/4)
    while (np.mod(M, fourth_of_M) != 0):
        M = M + 1
    delta_y = length_y_side / M
    M = int(M)
    N = int(N)
else:
    # force N = multp 4
    M = np.floor(length_y_side/(np.abs(lambda_smallest) / input_disc_per_lambda))
    fourth_of_M = np.ceil(M/4)

    while (np.mod(M, fourth_of_M) != 0):
        M = M + 1
    delta_y = length_y_side / M
    # force N = multp 4, size dx near dy
    N = np.floor(length_x_side/(delta_y))
    fourth_of_N = np.ceil(N/4)
    while (np.mod(N, fourth_of_N) != 0):
        N = N + 1
    delta_x = length_x_side / N
    M = int(M)
    N = int(N)


equiv_a = np.sqrt(delta_x*delta_y/np.pi)





# % Need to change geometry resolution here if frequency is lower scale
# % Also need to check how upscaling and downscaling perform.
# image_resize = imresize(image_object, [M, N], "nearest");
# Resize the image using nearest neighbor interpolation
image_resize = image_object.resize((N, M), resample=Image.NEAREST)

# % VISUALISE
# % image_object_render(image_object, materials_master, markerColor, 'Material Configuration After Scaling for f')
# image_object_render(image_object, materials_master, markerColor, title_str)

centre = 0.0 + 0.0 * 1j
start_pt = centre - 0.5 * length_x_side - 0.5 * length_y_side * 1j
print('\n \nSize of field of interst is %d meter by %d meter \n', length_x_side, length_y_side)
print('Discretizised space is %d grids by %d grids\n', N, M)
print('with grid size %d meter by %d meter \n', delta_x, delta_y)

# % SPECIFY_MATERIALS: POSITION, K AND RHO, FOR EACH BASIS COUNTER
# % INTERIOR SPECIFICATION: position, phi, k, and rho for each number of position (basis counter)
basis_counter = 0
# % This is wrong I think unless the vacuum in the scene is indexed as one.
# % Would it be better to pre-assign based on most likely material occurence?
basis_wave_number = np.zeros((M*N, 1))
basis_wave_number[0:M*N, 0] = vacuum_kr
position = np.zeros((M*N, 1), dtype='complex')
rho = np.zeros((M*N, 1))
the_phi = np.zeros((M*N, 1))
temp_vec = np.zeros((M*N, 1), dtype='complex')
# runs in y direction
for ct1 in range(1, M+1):
    # runs in x direction
    for ct2 in range(1, N+1):
        # nr of position
        basis_counter = (ct1 - 1) * N + ct2
        # % ORIENTATION OF BASIS COUNTER WILL BE IN X DIRECTION!
        # CHECK
        # position(basis_counter, 1) = start_pt + (ct2 - 0.5) * delta_x + 1i * (ct1 - 0.5) * delta_y;
        position[basis_counter-1, 0] = start_pt + (ct2 - 0.5) * delta_x + 1j * (ct1 - 0.5) * delta_y

        temp_vec = position[basis_counter-1, 0] - centre
        rho[basis_counter-1, 0] = np.abs(temp_vec)
        the_phi[basis_counter-1, 0] = np.arctan2(np.imag(temp_vec), np.real(temp_vec))
        basis_wave_number[basis_counter-1, 0] = float(materials_master.loc[materials_master['uint8'] == image_resize.getpixel((ct2-1, ct1-1)), 'kr'].iloc[0])

# % PROCESSING
# % Input Data: Error Bound; Max Iteration Steps
# % Processes: Mesh Adaption; Numerical Method; Equation Solver
# % MAKE_VEFIE_ELEMENTS: CREATES G_VECTOR, D AND V FROM (I+GD)E=V
# % CREATION OF ELEMENTS FOR VEFIE: (I+GD)E=V (Volume-equivalent Electric Field Intergral Equation)
V = np.zeros((basis_counter, 1), dtype=np.complex128)
# Diagonal contrast matrix stored as a vector
D = np.zeros((basis_counter, 1), dtype=np.complex128)
print('\nStart creation of all %d ellements of V and D\n \n', basis_counter)

# % Incident Field
for ct1 in range(1, basis_counter+1):
    # % V(ct1) = exp(-1i*kr(1)*rho(ct1)*cos(the_phi(ct1)));
    V[ct1-1, 0] = np.exp(-1j*vacuum_kr*rho[ct1-1, 0]*np.cos(the_phi[ct1-1, 0]))
    # contrast function
    D[ct1-1, 0] = basis_wave_number[ct1-1, 0] * basis_wave_number[ct1-1, 0] - vacuum_kr * vacuum_kr


x = np.real(position[0:N, 0])
y = np.imag(position[np.arange(0, N*M, N)])

# % SOLVE_WITH_FFT_AND_REDUCEDCG
# % Inclsuion of matrix pre-conditioners to help control for condition
# % number may be required.

# Reduced CG algoritm with FFT, solving Z*E=V with Z =I+GD %%%
# Reduced foreward operator
Rfo = D.astype(bool).astype(int)
# create V reduced
Vred = Rfo * V

print('\nStart creation of all %d ellements of G\n \n', basis_counter)
# BMT dense matrix, stored as a vector
G_vector = np.zeros((basis_counter, 1), dtype=np.complex128, order='F')
start2 = time.time()
for ct1 in range(1, basis_counter+1):
    if (np.mod(ct1, 100) == 0):
        print(ct1, '%dth element \n', ct1)
#     R_mn2 = abs(position(ct1)-position(1));
    R_mn2 = np.abs(position[ct1-1, 0]-position[0, 0])

    if ct1 == 1:
        # % G_vector(ct1,1)=(1i/4.0)*((2.0*pi*equiv_a/kr(1))*(besselj(1,kr(1)*equiv_a)-1i*bessely(1,kr(1)*equiv_a))-4.0*1i/(kr(1)*kr(1)));
        # % G_vector(ct1, 1) = (1i / 4.0) * ((2.0 * pi * equiv_a / kr(1)) * besselh(1, 2, kr(1)*equiv_a) - 4.0 * 1i / (kr(1) * kr(1)));
        G_vector[ct1-1, 0] = (1j/4.0) * ((2.0 * np.pi * equiv_a / vacuum_kr) * (jv(1, vacuum_kr * equiv_a) - 1j * yv(1, vacuum_kr * equiv_a)) - 4.0 * 1j/(vacuum_kr * vacuum_kr))
    else:
        # % G_vector(ct1,1)=(1i/4.0)*(2.0*pi*equiv_a/kr(1))*besselj(1,kr(1)*equiv_a)*(besselj(0,kr(1)*R_mn2)-1i*bessely(0,kr(1)*R_mn2));
        # % G_vector(ct1, 1) = (1i / 4.0) * (2.0 * pi * equiv_a / kr(1)) * besselj(1, kr(1)*equiv_a) * besselh(0, 2, kr(1)*R_mn2);
        G_vector[ct1-1, 0] = (1j / 4.0) * (2.0 * np.pi * equiv_a / vacuum_kr) * jv(1, vacuum_kr * equiv_a) * (jv(0, vacuum_kr * R_mn2) - 1j * yv(0, vacuum_kr * R_mn2))
Time_creation_all_elements_from_G = time.time() - start2
print("Time_creation_all_elements_from_G", Time_creation_all_elements_from_G)


# % Solver
def BMT_FFT(X, V, N):
    import numpy as np
    from scipy import fftpack

    # # discretise_M = resolution_information["length_y_side"]
    # # discretise_N = resolution_information["length_x_side"]

    # # X = G_vector.transpose()
    # # N = discretise_N
    # # M = discretise_M

    M = int(np.max(X.shape) / N)
    X.resize(M, N)

    # This does not match back to MATLAB as the final as MATLAB is missing final row and column based on idea.
    X_fliplr = np.flip(X, 1)
    X = np.append(X, np.zeros((M, 1)), axis=1)
    X = np.append(X, X_fliplr, axis=1)
    X_flipud = np.flip(X, 0)
    X = np.append(X, np.zeros((1, 2*N + 1)), axis=0)
    X = np.append(X, X_flipud, axis=0)

    # TO MATCH BACK TO MATLAB THE FOLLOWING TWO OPERATIONS ARE CREATED
    # DELETE OUT THE LAST ROW
    # X = X[:-1, :]

    # DELETE OUT THE LAST COLUMN
    # X = X[:, :-1]

    # Both
    X = X[:-1, :-1]

    # This matches back to MATLAB
    # V = np.zeros((M, N), dtype=np.complex128)
    # V = field_incident_D * Ered
    V.resize(M, N)
    V = np.append(V, np.zeros((M, N)), axis=1)
    V = np.append(V, np.zeros((M, 2*N)), axis=0)

    full_solution = fftpack.ifft2(fftpack.fft2(X)*fftpack.fft2(V))
    output = np.zeros((M * N, 1), dtype=np.complex128, order='F')
    output[0:N*M, 0] = full_solution[0:M, 0:N].flatten('F')
    return output


Ered = np.zeros((basis_counter, 1), order='F')
# guess
# % Z*E - V (= error)
r = Rfo * Ered + Rfo * BMT_FFT(G_vector.T, D*Ered, N) - Vred
# -Z'*r
p = -(Rfo * r + np.conj(D) * (BMT_FFT(np.conj(G_vector.T), Rfo * r, N)))

print('\nStart reduced CG iteration with 2D FFT\n')
# definition of error
solver_error = np.abs(np.vdot(r, r))

Reduced_iteration_error = np.zeros((1, 1))
# iteration counter
icnt = 0

start3 = time.time()
while (solver_error > input_solver_tol) and (icnt <= basis_counter):
    icnt = icnt + 1
    if icnt % 50 == 0:
        print(icnt, "th iteration Red")
    # a = (norm(Rfo.*r+conj(D).*BMT_FFT(conj(G_vector.'), Rfo.*r, N)) / norm(Rfo.*p+Rfo.*(BMT_FFT(G_vector.', D.*p, N))))^2; %(norm(Z'*r)^2)/(norm(Z*p)^2);
    a = (np.linalg.norm(Rfo*r + np.conj(D)*BMT_FFT(np.conj(G_vector.T), Rfo*r, N))/np.linalg.norm(Rfo*p + Rfo*BMT_FFT(G_vector.T, D*p, N)))**2
    Ered = Ered.copy() + a * p
    r_old = r.copy()
    # r = r + a*z*p
    r = r.copy() + a * (Rfo * p + Rfo * (BMT_FFT(G_vector.T, D*p, N)))
#     b = (norm(Rfo.*r+conj(D).*BMT_FFT(conj(G_vector.'), Rfo.*r, N)) / norm(Rfo.*r_old+conj(D).*BMT_FFT(conj(G_vector.'), Rfo.*r_old, N)))^2; %b = (norm(Z'*r)^2) /(norm(Z'*r_old)^2);
    # b = (norm(Z'*r)^2) /(norm(Z'*r_old)^2);
    b = (np.linalg.norm(Rfo*r+np.conj(D)*BMT_FFT(np.conj(G_vector.T), Rfo*r, N)) / np.linalg.norm(Rfo*r_old+np.conj(D)*BMT_FFT(np.conj(G_vector.T), Rfo*r_old, N)))**2
#     p = -(Rfo .* r + conj(D) .* BMT_FFT(conj(G_vector.'), Rfo.*r, N)) + b * p; % p=-Z'*r+b*p
#     solver_error = abs(r'*r);
#     Reduced_iteration_error(icnt, 1) = abs(r'*r);
# end
time_Red = time.time() - start3


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % POST-PROCESSING
# % Input Data: Evaluation Locality for diagrams, colour plots etc.
# % Processes: Optimisation; Further Modelling (Lumped Parameters); Approximation of local field qualities; Field Coupling)
# % PLOT_RESULTS
# % Write variable information to Command Window and plot reults
# % WRITE INFORMATION TO COMMAND WINDOW
# fprintf('\nUsed frequency of incomming wave = %d Hz \n', input_carrier_frequency)
# % fprintf('\nRelative Epsilon: \nconcrete = %d \nwood = %d \nglass = %d \n', epsilonrd, epsilonrdw, epsilonrdg)
# % fprintf('\nWave Lengths: \nfree space %d meter\nconcrete %d meter\nwood %d meter \nglass %d meter\n', lambda0, lambda_d, lambda_w, lambda_g)
# % fprintf('\n \nSize of field of interst is %d meter by %d meter \n', length_x_side, length_y_side)
# fprintf('Discretizised space is %d grids by %d grids\n', N, M)
# fprintf('with grid size %d meter by %d meter', delta_x, delta_y)
# fprintf('So basis counter goes to %d \n', basis_counter)
# fprintf('\nNumber of unknowns in Z is than %d\n', N*M*N*M)
# fprintf('Number of reduced unknowns is %d\n', sum(Rfo)*N*M)
# fprintf('with %.2f%% percent of the is filled by contrast \n', sum(Rfo)/(N * M)*100)
# fprintf('\nCG iteration error tollerance = %d \n', input_solver_tol)
# % fprintf('Duration of reduced CG iteration = %d seconds \n', time_Red)

# % CONVERT SOLUTIONS ON 2D GRID, FOR 3D PLOTS
# vec2matSimulationVred = zeros(M, N);
# for i = 1:M %without vec2mat:
#     vec2matSimulationVred(i, :) = Vred((i - 1)*N+1:i*N);
# end
# vec2matSimulationEred = zeros(M, N);
# for i = 1:M %without vec2mat:
#     vec2matSimulationEred(i, :) = Ered((i - 1)*N+1:i*N);
# end

# Vred_2D = vec2matSimulationVred;
# Ered_2D = vec2matSimulationEred;
# ScatRed_2D = Ered_2D - Vred_2D;

# % CREATE ALL THE PLOTS
# figure
# set(gcf, 'units', 'normalized', 'outerposition', [0, 0, 1, 1])
# subplot(1, 2, 1)
# imagesc(ind2rgb(image_resize, materials_master.('map')), 'XData', 1/2, 'YData', 1/2)
# axis tight
# title('Material Configuration After Simulation')
# legend
# hold on
# L = plot(ones(height(materials_master(:, 2))), 'LineStyle', 'none', 'marker', 's', 'visible', 'on');
# set(L, {'MarkerFaceColor'}, markerColor, {'MarkerEdgeColor'}, markerColor);
# colormap(materials_master.('map'))
# legend(materials_master.('name'))

# subplot(1, 2, 2)
# semilogy(Reduced_iteration_error, 'r')
# title('Iteration Error')
# legend('Reduced')
# % TBC: The axis for the iteration should be fixed so that improvement can
# % be compared easily. This will be implemented when there is more than one
# % iterative approach being pursued.
# axis tight
# ylabel('Number of iteration')
# xlabel('Error')
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % 3D plots of real/abs of total/scatterd/incomming
# figure
# set(gcf, 'units', 'normalized', 'outerposition', [0, 0, 1, 1])

# subplot(2, 3, 1)
# surf(x, y, real(Vred_2D));
# % view(2)
# view(0, 270)
# shading interp
# title('Reduced Incoming Wave Part Real');
# xlabel('x (meters)')
# ylabel('y (meter)')
# axis tight

# subplot(2, 3, 2)
# surf(x, y, real(ScatRed_2D));
# % view(2)
# view(0, 270)
# shading interp
# title('Reduced Scattered Field Part Real');
# xlabel('x (meters)')
# ylabel('y (meter)')
# axis tight

# subplot(2, 3, 3)
# surf(x, y, real(Ered_2D))
# % view(2)
# view(0, 270)
# shading interp
# title('Reduced Total Field Part Real');
# xlabel('x (meters)')
# ylabel('y (meter)')
# axis tight

# subplot(2, 3, 4)
# surf(x, y, abs(Vred_2D));
# % view(2)
# view(0, 270)
# shading interp
# title('Reduced Incoming Wave Part Absolute');
# xlabel('x (meters)')
# ylabel('y (meter)')
# axis tight

# subplot(2, 3, 5)
# surf(x, y, abs(ScatRed_2D));
# % view(2)
# view(0, 270)
# shading interp
# title('Reduced Scattered Field Part Absolute');
# xlabel('x (meters)')
# ylabel('y (meter)')
# axis tight

# subplot(2, 3, 6)
# surf(x, y, abs(Ered_2D))
# % view(2)
# view(0, 270)
# shading interp
# title('Reduced Total Field Part Absolute');
# xlabel('x (meters)')
# ylabel('y (meter)')
# axis tight


# BELOW HERE DONE

total_time_ran = time.time() - total_time
print("total_time_ran", total_time_ran)
