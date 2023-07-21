import tensorflow as tf
from IPython import get_ipython
import numpy as np
import sys
import os
import time
import pandas as pd
import json
import matplotlib.colors as mcolors
import random
from lib import custom_functions

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')
print(tf.config.list_physical_devices('GPU'))

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=20)

"""
Description:
This file is the script for the adapted Scalar 2D VDB code.
    Time factor = exp(-iwt)
    Spatial units is in m
    Source wavelet  Q = 1
help(custom_functions.f)
"""
random.seed(42)

# USER INPUTS
# Number of samples to generate
seedling = 0
seed_count = 1000
# seed_count = 1
# Folder to save contrast scene array and visualisation
input_folder = "instances"
# Folder to save solved scene arrays and solution metric information
output_folder = "instances_output"
# Look-up table for material properties
path_lut = './lut/tissues.json'
# The scene will have up to four different materials: 'vacuum'; 'normal tissue'; 'benign tumor'; 'cancer'.
scene_tissues = ["vacuum", "normal tissue", "benign tumor", "cancer"]

f = custom_functions.f(0.5e9)

# assume a square photo for data augmentation reasons.
ml_discretisation_N1 = 128
ml_discretisation_N2 = 128

# min geometry delta is 1000e-6 meters in order to designate a pixel as a cancer tumour in a photo
resolution_photo = 1000e-6

# if delta_x or delta_y is less than resolution_photo then the photo resolution does not capture the material properties
# therefore an image must have at minimum one length of 1000e-6*128 = 0.128 meters to allow for a pixel to be designated as a cancer cell in a photo
# to keep A4 proportions use *(0.297/0.210) at the ml stage
length_x_side = resolution_photo*ml_discretisation_N1
length_y_side = resolution_photo*ml_discretisation_N2

NR = custom_functions.NR(180)
Errcri = custom_functions.Errcri(1e-18)
input_disc_per_lambda = custom_functions.input_disc_per_lambda(10)

# DERIVED VARIABLES
epsilon0 = custom_functions.epsilon0()
mu0 = custom_functions.mu0()
c_0 = custom_functions.c_0(epsilon0, mu0)
angular_frequency = custom_functions.angular_frequency(f)

with open(path_lut, 'rb') as fid:
    raw = fid.read()
    string = raw.decode('utf-8')

materials_master = pd.DataFrame(json.loads(string))
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
    materials_master.loc[k, 'epsilonr'], materials_master.loc[k, 'sigma'], materials_master.loc[k, 'epsilonr_complex'] = custom_functions.tissuePermittivity(materials_master['name'][k], f, epsilon0)
    materials_master.loc[k, 'mur'] = 1.0
    materials_master.loc[k, 'mur_complex'] = 1.0 - (0.0 * 1j)
    materials_master.loc[k, 'cr'] = 1.0 / np.sqrt(materials_master.loc[k, 'epsilonr'] * epsilon0 * materials_master.loc[k, 'mur'] * mu0)
    materials_master.loc[k, 'cr_complex'] = 1.0 / np.sqrt(materials_master.loc[k, 'epsilonr_complex'] * epsilon0 * materials_master.loc[k, 'mur'] * mu0)
    # Propagation constants
    materials_master.loc[k, 'kr'] = angular_frequency * np.sqrt(materials_master.loc[k, 'epsilonr'] * epsilon0 * materials_master.loc[k, 'mur'] * mu0)
    materials_master.loc[k, 'kr_complex'] = angular_frequency * np.sqrt(materials_master.loc[k, 'epsilonr_complex'] * epsilon0 * materials_master.loc[k, 'mur'] * mu0)

# Choose the smallest lambda (wavelength) of all materials in the configuration.
lambda_smallest = custom_functions.lambda_smallest(materials_master, f)

# Check if number of discretisations is more than the photo resolution requirements.
if length_x_side > length_y_side:
    # force N = multp 4
    N = np.floor(length_x_side/(np.abs(lambda_smallest) / input_disc_per_lambda))
    fourth_of_N = np.ceil(N/4)
    while (np.mod(N, fourth_of_N) != 0):
        N = N + 1
    N = int(N)
    delta_x = length_x_side / N
    # force M = multp 4, size dy near dx
    M = np.floor(length_y_side/(delta_x))
    fourth_of_M = np.ceil(M/4)
    while (np.mod(M, fourth_of_M) != 0):
        M = M + 1
    M = int(M)
    delta_y = length_y_side / M
else:
    # force N = multp 4
    M = np.floor(length_y_side/(np.abs(lambda_smallest) / input_disc_per_lambda))
    fourth_of_M = np.ceil(M/4)
    while (np.mod(M, fourth_of_M) != 0):
        M = M + 1
    M = int(M)
    delta_y = length_y_side / M
    # force N = multp 4, size dx near dy
    N = np.floor(length_x_side/(delta_y))
    fourth_of_N = np.ceil(N/4)
    while (np.mod(N, fourth_of_N) != 0):
        N = N + 1
    delta_x = length_x_side / N
    N = int(N)

print("delta_x", delta_x)
print("delta_y", delta_y)
# M and N at this stage are minimum requirements based on material properties
if delta_x < resolution_photo or delta_y < resolution_photo:
    print("Problem")
    sys.exit()


N1 = custom_functions.N1(ml_discretisation_N1)
N2 = custom_functions.N2(ml_discretisation_N2)
dx = custom_functions.dx(resolution_photo)

radius_source = custom_functions.radius_source(N1*dx)
radius_receiver = custom_functions.radius_receiver((15/17)*radius_source)
# a = custom_functions.a((4/17)*radius_source)
# a = custom_functions.a((4/17)*np.minimum(N1*dx, N2*dx))
a = custom_functions.a(np.minimum(N1*dx, N2*dx)/2.0)


wavelength = custom_functions.wavelength(c_0, f)
s = custom_functions.s(f, angular_frequency)
gamma_0 = custom_functions.gamma_0(s, c_0)
xS = custom_functions.xS(radius_source)
rcvr_phi = custom_functions.rcvr_phi(NR)
xR = custom_functions.xR(radius_receiver, rcvr_phi, NR)

initGrid = custom_functions.initGrid(N1, N2, dx)

X1 = custom_functions.X1(initGrid)
X2 = custom_functions.X2(initGrid)

initFFTGreenGrid = custom_functions.initFFTGreenGrid(N1, N2, dx, gamma_0)

X1fft = custom_functions.X1fft(initFFTGreenGrid)
X2fft = custom_functions.X2fft(initFFTGreenGrid)

Green = custom_functions.Green(dx, gamma_0, X1fft, X2fft)
FFTG = custom_functions.FFTG(Green)
R = custom_functions.R(X1, X2)
u_inc = custom_functions.u_inc(gamma_0, xS, dx, X1, X2)


def complex_separation(complex_array):
    # Separate real and imaginary components
    real_array = np.real(complex_array)
    imaginary_array = np.imag(complex_array)

    # Compute absolute array
    absolute_array = np.abs(complex_array)

    # Stack the arrays together
    result_array = np.stack([real_array, imaginary_array, absolute_array])
    return result_array


u_inc_stacked = complex_separation(u_inc)

# DEFAULT SETTING
# The default primary scatter is 'normal tissue'. It will take up a circle which is the same as the Bessel-Aprroach geometry.
contrast_sct = custom_functions.contrast_sct(materials_master.loc[materials_master.loc[materials_master['name'] == 'normal tissue'].index[0], 'epsilonr'])
c_sct = custom_functions.c_sct(c_0, contrast_sct)
CHI_array = custom_functions.CHI_Bessel(c_0, c_sct, R, a)
CHI = custom_functions.CHI(CHI_array)
custom_functions.plotContrastSource(u_inc, CHI, X1, X2)
itmax = custom_functions.itmax(CHI)


# Generate the regions of interest
# The geometric dimensions of benign and cancerous tissue are comparable with the aim of classifying cancerous tissue before reaching fatal sizes.
# Set the max radius of benign tissue at 5% of the normal tissue area:
radius_max_pix_b = int(np.floor(np.sqrt(0.05)*np.minimum(N1, N2)))
# Set the max radius of cancerous tissue at 2.5% of the normal tissue area:
radius_max_pix_c = int(np.floor(np.sqrt(0.025)*np.minimum(N1, N2)))
# Set the min area of all non-normal tissue equal to four pixel to avoid simulating normal samples repeatidly
radius_min_pix = 4

def generate_ROI(CHI, radius_min_pix, radius_max_pix_b, radius_max_pix_c, seedling, seed_count, input_folder, R, a):
    from skimage import io
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    c_b = custom_functions.contrast_sct(materials_master.loc[materials_master.loc[materials_master['name'] == 'benign tumor'].index[0], 'epsilonr'])
    print("c_b", c_b)
    # contrast_c = custom_functions.contrast_sct(materials_master.loc[materials_master.loc[materials_master['name'] == 'cancer'].index[0], 'epsilonr'])
    for seed in range(seedling, seedling+seed_count):

        shape_array = CHI.copy()
        # Generate some benign tissue
        radius = random.uniform(radius_min_pix, radius_max_pix_b)
        center_x = random.uniform(radius, N1 - radius)
        center_y = random.uniform(radius, N2 - radius)

        for i in range(N2):
            for j in range(N1):
                if np.sqrt((j - center_x) ** 2 + (i - center_y) ** 2) <= radius:
                    shape_array[i, j] = 1 - c_b

        # if not in the original "normal tissue" region then set to zero.
        # A certain amount of the tumour tissues will land beyond the normal tissue region contributing to missed captures at the photo stage.
        shape_array[R > a] = 0.0

        # Only use as visualisation, not input data. Use the npy array as input data to avoid clipping etc.
        plt.imsave(os.path.join(input_folder, f"instance_{str(seed).zfill(10)}.png"), shape_array, cmap='gray')
        np.save(os.path.join(input_folder, f"instance_{str(seed).zfill(10)}.npy"), shape_array)
    return shape_array


# GENERATE GEOMETRY SAMPLES
os.makedirs(input_folder, exist_ok=True)
generate_ROI(CHI, radius_min_pix, radius_max_pix_b, radius_max_pix_c, seedling, seed_count, input_folder, R, a)
# SOLVE THE SCENES AND SAVE OUTPUTS
# Get the list of numpy files in the input folder
numpy_files = [f for f in os.listdir(input_folder) if f.endswith(".npy")]

os.makedirs(output_folder, exist_ok=True)
# Iterate over each numpy file
for file_name in numpy_files:
    # Load the numpy array
    geometry_file = os.path.join(input_folder, file_name)
    CHI = np.load(geometry_file)

    # Solve the instances
    b = custom_functions.b(CHI, u_inc)
    tic0 = time.time()
    # print("tic0", tic0)
    ITERBiCGSTABw = custom_functions.ITERBiCGSTABw(u_inc, CHI, Errcri, N1, N2, b, FFTG, itmax, x0=None)
    toc0 = time.time() - tic0
    print("toc", toc0)
    # Save the result in the output folder with the same file name
    exit_code_o = custom_functions.exit_code(ITERBiCGSTABw)
    if exit_code_o == 0:
        w_o = custom_functions.w(ITERBiCGSTABw)
        information_o = custom_functions.information(ITERBiCGSTABw)

        output_file_path = os.path.join(output_folder, file_name)
        # final_array = np.concatenate((u_inc, CHI, w_o), axis=0)

        # final_array = np.concatenate([u_inc[np.newaxis, :, :], CHI[np.newaxis, :, :], w_o[np.newaxis, :, :]], axis=0)
        final_array = np.concatenate([u_inc_stacked[np.newaxis, :, :], complex_separation(CHI)[np.newaxis, :, :], complex_separation(w_o)[np.newaxis, :, :]], axis=0)
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
#     # custom_functions.plotContrastSource(u_inc, CHI, X1, X2)
#     custom_functions.plotContrastSource(np.abs(array[0, :, :]), np.abs(array[1, :, :]), X1, X2)
#     # custom_functions.plotContrastSource(w, CHI, X1, X2)
#     custom_functions.plotContrastSource(np.abs(array[2, :, :]), np.abs(array[1, :, :]), X1, X2)
#     # custom_functions.plotContrastSource(u_inc + w, CHI, X1, X2)
#     custom_functions.plotContrastSource(np.abs(array[0, :, :]+array[2, :, :]), np.abs(array[1, :, :]), X1, X2)

#     test = array[0, :, :]+array[2, :, :]



# import matplotlib.pyplot as plt

# # Create a complex array
# complex_array = test

# # Get the real and imaginary parts
# real_part = np.real(complex_array)
# imaginary_part = np.imag(complex_array)

# # Find the maximum and minimum values
# max_value = np.max(np.abs(complex_array))
# min_value = -max_value

# # Plot the real part
# plt.imshow(real_part, cmap='jet', vmin=min_value, vmax=max_value)
# plt.colorbar()
# plt.title('Real Part')
# plt.show()

# # Plot the imaginary part
# plt.imshow(imaginary_part, cmap='jet', vmin=min_value, vmax=max_value)
# plt.colorbar()
# plt.title('Imaginary Part')
# plt.show()


# data = custom_functions.Dop(w, NR, N1, N2, xR, gamma_0, dx, X1, X2)
# angle = custom_functions.angle(rcvr_phi)
# # custom_functions.displayDataCSIEApproach(data, angle)

# data2D = custom_functions.WavefieldSctCircle(c_0, c_sct, gamma_0, xS, NR, rcvr_phi, xR, N1, N2, dx, X1, X2, FFTG, a, CHI, Errcri)
# custom_functions.displayDataCompareApproachs(data2D, data, angle)
# error = np.linalg.norm(data.flatten('F') - data2D.flatten('F'), ord=1)/np.linalg.norm(data2D.flatten('F'), ord=1)
# print("error", error)

"""
Documentation
"""
path_doc = "./doc/code_doc/"


def composer_call():
    """


    Returns
    -------
    None.

    """
    from fn_graph import Composer
    composer_1 = (
        Composer()
        .update(
            # list of custom functions goes here
            custom_functions.a,
            custom_functions.angle,
            custom_functions.angular_frequency,
            custom_functions.b,
            custom_functions.CHI,
            custom_functions.CHI_Bessel,
            custom_functions.contrast_sct,
            custom_functions.c_0,
            custom_functions.c_sct,
            custom_functions.displayDataCSIEApproach,
            custom_functions.Dop,
            custom_functions.dx,
            custom_functions.Errcri,
            custom_functions.exit_code,
            custom_functions.f,
            custom_functions.FFTG,
            custom_functions.gamma_0,
            custom_functions.Green,
            custom_functions.information,
            custom_functions.initFFTGreenGrid,
            custom_functions.initGrid,
            custom_functions.input_disc_per_lambda,
            custom_functions.ITERBiCGSTABw,
            custom_functions.itmax,
            custom_functions.lambda_smallest,
            custom_functions.N1,
            custom_functions.N2,
            custom_functions.NR,
            custom_functions.plotContrastSource,
            custom_functions.R,
            custom_functions.radius_receiver,
            custom_functions.radius_source,
            custom_functions.rcvr_phi,
            custom_functions.s,
            custom_functions.tissuePermittivity,
            custom_functions.u_inc,
            custom_functions.w,
            custom_functions.wavelength,
            custom_functions.X1,
            custom_functions.X1fft,
            custom_functions.X2,
            custom_functions.X2fft,
            custom_functions.xR,
            custom_functions.xS,
        )
        # .update_parameters(input_length_side=input_length_x_side)
        # .cache()
    )
    return composer_1


custom_functions.composer_render(composer_call(), path_doc, "function_flow")
# help(custom_functions.gamma_0)
