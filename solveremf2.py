import random
import numpy as np
import os
import sys
import time
from IPython import get_ipython
import matplotlib.pyplot as plt
from lib import custom_functions

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=20)

# HARD-CODED VARIABLES
epsilon0 = 8.854187817e-12
mu0 = 4.0 * np.pi * 1.0e-7
seedling = 0
random.seed(42)

# Number of samples to generate
seed_count = 2
# Folder to save contrast scene array and visualisation
input_folder = "instances"
# Folder to save solved scene arrays and solution metric information
output_folder = "instances_output"
# Look-up table for material properties
path_lut = './lut/tissues.json'


# ESTABLISH EM PARAMETERS
# Time factor = exp(-iwt)
# Spatial units is in m
# Source wavelet M Z_0 / gamma_0  = 1   (Z_0 M = gamma_0)

# chosen accuracy
input_disc_per_lambda = 10

# wave speed in embedding
c_0 = 3e8
# c_0 = 2.99792458e8
# relative permittivity of scatterer
eps_sct = 1.75
# relative permeability of scatterer
mu_sct = 1.0

# temporal frequency (Hz)
f = 10e6
# wavelength
wavelength = c_0 / f
# Laplace parameter where 2.0 * np.pi * f is the angular frequency (rad/s)
s = 1e-16 - 1j*2*np.pi*f
# propagation coefficient
gamma_0 = s/c_0

# number of samples in x_1
N1 = 120
# number of samples in x_2
N2 = 100
# with meshsize dx
dx = 2









# GENERATE GEOMETRY
radius_min_pix = int(0.05 * np.minimum(N1, N2))
radius_max_pix = int(0.15 * np.minimum(N1, N2))

os.makedirs(input_folder, exist_ok=True)

custom_functions.generate_random_circles(N1, N2, radius_min_pix, radius_max_pix, seedling, seed_count, input_folder)

# INITIALISE SCENE AND SAVE OUTPUTS
xS, NR, rcvr_phi, xR, X1, X2, FFTG, Errcri = custom_functions.initEM(c_0, eps_sct, mu_sct, gamma_0, N1, N2, dx)
E_inc, ZH_inc = custom_functions.IncEMwave(gamma_0, xS, dx, X1, X2)


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
    w_E_o, exit_code_o, information_o = custom_functions.ITERBiCGSTABwE(E_inc, CHI_eps, Errcri, N1, N2, dx, FFTG, gamma_0)
    toc0 = time.time() - tic0
    # print("information[-1, :]", information_o[-1, :])
    print("toc", toc0)

    # Save the result in the output folder with the same file name
    output_file_path = os.path.join(output_folder, file_name)
    final_array = np.concatenate((np.expand_dims(CHI_eps, axis=0), np.expand_dims(CHI_mu, axis=0), w_E_o), axis=0)
    np.save(output_file_path, final_array)

    output_file_path_info = os.path.join(output_folder, os.path.splitext(file_name)[0] + "_info")
    np.save(output_file_path_info, information_o)


numpy_files = [f for f in os.listdir(output_folder) if f.endswith(".npy") and not f.endswith("_info.npy") and f.startswith("instance_")]
# Iterate over each numpy file
for file_name in numpy_files:
    # Load the numpy array
    geometry_file = os.path.join(output_folder, file_name)
    array = np.load(geometry_file)
    print("array.shape", array.shape)
    # print(array.shape)
    custom_functions.plotEMContrast(np.real(array[0, :, :]), np.real(array[1, :, :]), X1, X2)
    # custom_functions.plotContrastSourcewE(array[2:5], X1, X2)


# custom_functions.plotEMContrast(CHI_eps, CHI_mu, X1, X2)

# x0 = np.concatenate([w_E_o[0, :, :].flatten('F'), w_E_o[1, :, :].flatten('F')], axis=0)

# tic1 = time.time()
# w_E, exit_code, information = custom_functions.ITERBiCGSTABwE(E_inc, CHI_eps, Errcri, N1, N2, dx, FFTG, gamma_0, x0)
# toc1 = time.time() - tic1
# # print("information[-1, :]", information[-1, :])
# print("toc", toc1)

# print("fewer iterations by", information_o[-1, 0] - information[-1, 0])
# print("closer by", information_o[-1, 1] - information[-1, 1])
# print("faster by", information_o[-1, 2] - information[-1, 2])

# E_sct = custom_functions.KopE(w_E, gamma_0, N1, N2, dx, FFTG)

# custom_functions.plotContrastSourcewE(w_E, X1, X2)
# E_val = custom_functions.E(E_inc, E_sct)
# custom_functions.plotEtotalwavefield(E_val, a, X1, X2, N1, N2)


# ITERATION INFORMATION
E_inc = np.load(os.path.join(output_folder, 'E_inc.npy'))
ZH_inc = np.load(os.path.join(output_folder, 'ZH_inc.npy'))
X1 = np.load(os.path.join(output_folder, 'X1.npy'))
X2 = np.load(os.path.join(output_folder, 'X2.npy'))

numpy_files = [f for f in os.listdir(output_folder) if f.endswith("_info.npy")]
# Iterate over each numpy file
for file_name in numpy_files:
    # Load the numpy array
    geometry_file = os.path.join(output_folder, file_name)
    array = np.load(geometry_file)
    print(array.shape)

    # Plot the data
    plt.plot(array[:, 0], array[:, 1])
    plt.yscale('log')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot of first column (X) vs. second column (Y)')
    plt.show()

    # Plot the data
    plt.plot(array[:, 0], array[:, 2])
    plt.yscale('log')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot of first column (X) vs. third column (Y)')
    plt.show()

