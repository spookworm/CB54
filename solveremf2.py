import tensorflow as tf
import random
import numpy as np
import os
import sys
import time
from IPython import get_ipython
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import pandas as pd
from lib import custom_functions

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=20)
print(tf.config.list_physical_devices('GPU'))


# HARD-CODED VARIABLES
epsilon0 = 8.854187817e-12
mu0 = 4.0 * np.pi * 1.0e-7
seedling = 0
random.seed(42)

# Number of samples to generate
seed_count = 3000
# seed_count = 1
# Folder to save contrast scene array and visualisation
input_folder = "instances"
# Folder to save solved scene arrays and solution metric information
output_folder = "instances_output"
# Look-up table for material properties
path_lut = './lut/tissues.json'
# The scene will have up to four different materials: 'vacuum'; 'normal tissue'; 'benign tumor'; 'cancer'.
scene_tissues = ["vacuum", "normal tissue", "benign tumor", "cancer"]

# ESTABLISH EM PARAMETERS
# Time factor = exp(-iwt)
# Spatial units is in m
# Source wavelet M Z_0 / gamma_0  = 1   (Z_0 M = gamma_0)

# chosen accuracy
input_disc_per_lambda = 10

# wave speed in embedding
c_0 = 2.99792458e8
# Photo dimensions
length_x_side = 0.42
length_y_side = length_x_side
# length_y_side = length_x_side
# temporal frequency (Hz)
# f = 2.225e9
f = 1.1125e9
# wavelength
wavelength = c_0 / f
# angular frequency (rad/s)
angular_frequency = 2.0 * np.pi * f
# Laplace parameter
s = 1e-16 - 1j*angular_frequency
# propagation coefficient
gamma_0 = s/c_0

with open(path_lut, 'rb') as fid:
    raw = fid.read()
    str = raw.decode('utf-8')

materials_master = pd.DataFrame(json.loads(str))
materials_master = materials_master[materials_master['name'].isin(scene_tissues)]
materials_master = materials_master.reset_index(drop=True)
unique_integers = np.sort(np.unique(materials_master['uint8']))

materials_master['RGB'] = materials_master['HEX'].apply(lambda x: mcolors.hex2color(x))
markerColor = pd.DataFrame(materials_master['RGB'])

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
    materials_master.loc[k, 'epsilonr'], materials_master.loc[k, 'sigma'], materials_master.loc[k, 'epsilonr_complex'] = custom_functions.tissuePermittivity(materials_master['name'][k], f)
    materials_master.loc[k, 'mur'] = 1.0
    materials_master.loc[k, 'mur_complex'] = 1.0 - (0.0 * 1j)
    materials_master.loc[k, 'cr'] = 1.0 / np.sqrt(materials_master.loc[k, 'epsilonr'] * epsilon0 * materials_master.loc[k, 'mur'] * mu0)
    materials_master.loc[k, 'cr_complex'] = 1.0 / np.sqrt(materials_master.loc[k, 'epsilonr_complex'] * epsilon0 * materials_master.loc[k, 'mur'] * mu0)
    # Propagation constants
    materials_master.loc[k, 'kr'] = angular_frequency * np.sqrt(materials_master.loc[k, 'epsilonr'] * epsilon0 * materials_master.loc[k, 'mur'] * mu0)
    materials_master.loc[k, 'kr_complex'] = angular_frequency * np.sqrt(materials_master.loc[k, 'epsilonr_complex'] * epsilon0 * materials_master.loc[k, 'mur'] * mu0)

# Choose the smallest lambda (wavelength) of all materials in the configuration.
lambda_smallest = np.min(materials_master['cr']) / f

"""
Typical image size in Mammography:
    4,000 pixels by 5,000 pixels.
Pixel size can is typically around 100 to 200 micrometers
This results in an image size of approximately 10 cm by 12.5 cm.
Therefore aim for:
    dx = 200e-6
    length_x_side = 12.5e-2
    length_y_side = 10e-2
    N1 = 5000
    N2 = 4000

make a square 4000 x 4000 pixels where the dx is set by the lambda and the
N = 4000
dx = 200e-6
4000 * 200e-6

length_x_side =
lambda_smallest =
input_disc_per_lambda =

N = np.floor(length_x_side/(np.abs(lambda_smallest) / input_disc_per_lambda))
"""
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


# relative permittivity of scatterer
# eps_sct = 1.75
eps_sct = materials_master.loc[materials_master.loc[materials_master['name'] == 'cancer'].index[0], 'epsilonr']
# relative permeability of scatterer
mu_sct = 1.0
# mu_sct = materials_master.loc[materials_master.loc[materials_master['name'] == 'cancer'].index[0], 'mur']
# number of samples in x_1
N1 = N
# number of samples in x_2
N2 = M
# with meshsize dx
dx = delta_x

# GENERATE GEOMETRY
radius_min_pix = int(0.05 * np.minimum(N1, N2))
radius_max_pix = int(0.15 * np.minimum(N1, N2))

os.makedirs(input_folder, exist_ok=True)

custom_functions.generate_random_circles(N1,  N2, radius_min_pix, radius_max_pix, seedling, seed_count, input_folder)

# INITIALISE SCENE AND SAVE OUTPUTS
xS, NR, rcvr_phi, xR, X1, X2, FFTG, Errcri = custom_functions.initEM(c_0, eps_sct, mu_sct, gamma_0, N1, N2, dx)
E_inc, ZH_inc = custom_functions.IncEMwave(gamma_0, xS, dx, X1, X2)
custom_functions.plotComplexArray(cmd='abs', array=E_inc[0], cmap_name='jet', cmap_min=0, cmap_max=np.max(np.abs(E_inc)), title='abs')
custom_functions.plotComplexArray(cmd='abs', array=E_inc[1], cmap_name='jet', cmap_min=0, cmap_max=np.max(np.abs(E_inc)), title='abs')
custom_functions.plotComplexArray(cmd='real', array=E_inc[0], cmap_name='jet', cmap_min=0, cmap_max=np.max(np.abs(E_inc)), title='real_part')
custom_functions.plotComplexArray(cmd='real', array=E_inc[1], cmap_name='jet', cmap_min=0, cmap_max=np.max(np.abs(E_inc)), title='real_part')
custom_functions.plotComplexArray(cmd='imag', array=E_inc[0], cmap_name='jet', cmap_min=0, cmap_max=np.max(np.abs(E_inc)), title='imag')
custom_functions.plotComplexArray(cmd='imag', array=E_inc[1], cmap_name='jet', cmap_min=0, cmap_max=np.max(np.abs(E_inc)), title='imag')

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

np.save(os.path.join(output_folder, 'E_inc'), E_inc)
np.save(os.path.join(output_folder, 'ZH_inc'), ZH_inc)
np.save(os.path.join(output_folder, 'X1'), X1)
np.save(os.path.join(output_folder, 'X2'), X2)

# SOVLE THE SCENES AND SAVE OUTPUTS
# Get the list of numpy files in the input folder
numpy_files = [f for f in os.listdir(input_folder) if f.endswith(".npy")]

# Iterate over each numpy file
for file_name in numpy_files:
    # Load the numpy array
    geometry_file = os.path.join(input_folder, file_name)
    # array = np.load(file_path)

    # Perform the scattering simulation
    # add contrast distribution
    a, CHI_eps, CHI_mu = custom_functions.initEMContrast(eps_sct, mu_sct, X1, X2, geometry_file)

    tic0 = time.time()
    print("tic0", tic0)
    w_E_o, exit_code_o, information_o = custom_functions.ITERBiCGSTABwE(E_inc, CHI_eps, Errcri, N1, N2, dx, FFTG, gamma_0, x0=0)
    toc0 = time.time() - tic0
    # print("information[-1, :]", information_o[-1, :])
    print("toc", toc0)
    if exit_code_o == 0:
        # Save the result in the output folder with the same file name
        output_file_path = os.path.join(output_folder, file_name)
        final_array = np.concatenate((np.expand_dims(CHI_eps, axis=0), np.expand_dims(CHI_mu, axis=0), w_E_o), axis=0)
        np.save(output_file_path, final_array)

        output_file_path_info = os.path.join(output_folder, os.path.splitext(file_name)[0] + "_info")
        np.save(output_file_path_info, information_o)
    else:
        print("file_name : ", file_name, " has exit_code_o ", exit_code_o)


# numpy_files = [f for f in os.listdir(output_folder) if f.endswith(".npy") and not f.endswith("_info.npy") and f.startswith("instance_")]
# # Iterate over each numpy file
# for file_name in numpy_files:
#     # Load the numpy array
#     geometry_file = os.path.join(output_folder, file_name)
#     array = np.load(geometry_file)
#     # print("array.shape", array.shape)
#     # print(array.shape)
#     custom_functions.plotEMContrast(np.real(array[0, :, :]), np.real(array[1, :, :]), X1, X2)
#     custom_functions.plotContrastSourcewE(array[2:5], X1, X2)


# # x0 = np.concatenate([w_E_o[0, :, :].flatten('F'), w_E_o[1, :, :].flatten('F')], axis=0)

# # tic1 = time.time()
# # w_E, exit_code, information = custom_functions.ITERBiCGSTABwE(E_inc, CHI_eps, Errcri, N1, N2, dx, FFTG, gamma_0, x0)
# # toc1 = time.time() - tic1
# # # print("information[-1, :]", information[-1, :])
# # print("toc", toc1)

# # print("fewer iterations by", information_o[-1, 0] - information[-1, 0])
# # print("closer by", information_o[-1, 1] - information[-1, 1])
# # print("faster by", information_o[-1, 2] - information[-1, 2])

# # E_sct = custom_functions.KopE(w_E, gamma_0, N1, N2, dx, FFTG)

# # custom_functions.plotContrastSourcewE(w_E, X1, X2)
# # E_val = custom_functions.E(E_inc, E_sct)
# # custom_functions.plotEtotalwavefield(E_val, a, X1, X2, N1, N2)


# # ITERATION INFORMATION
# E_inc = np.load(os.path.join(output_folder, 'E_inc.npy'))
# ZH_inc = np.load(os.path.join(output_folder, 'ZH_inc.npy'))
# X1 = np.load(os.path.join(output_folder, 'X1.npy'))
# X2 = np.load(os.path.join(output_folder, 'X2.npy'))

# numpy_files = [f for f in os.listdir(output_folder) if f.endswith("_info.npy")]
# # Iterate over each numpy file
# for file_name in numpy_files:
#     # Load the numpy array
#     geometry_file = os.path.join(output_folder, file_name)
#     array = np.load(geometry_file)
#     # print(array.shape)

#     # Plot the data
#     plt.plot(array[:, 0], array[:, 1])
#     plt.yscale('log')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Plot of first column (X) vs. second column (Y)')
#     plt.show()

#     # Plot the data
#     plt.plot(array[:, 0], array[:, 2])
#     plt.yscale('log')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Plot of first column (X) vs. third column (Y)')
#     plt.show()

# Review individual information for _0 and _m
array_0 = np.load('./instances_output_300_0/instance_0000000046_info.npy')
array_m = np.load('./instances_output_300_m/instance_0000000046_info.npy')
