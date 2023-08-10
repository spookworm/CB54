from IPython import get_ipython
import numpy as np
import sys
import time
import random
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.metrics import MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError
from keras.models import load_model
from lib import custom_functions_EM
from lib import custom_architectures_EM

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=20)
random.seed(42)
# INPUTS: START
# Would you like to validate the code?
validation = 'True'
validation = 'False'
# Would you like to validate that the Krylov Solver accepts correct final answer as a good initial guess?
guess_validation_answer = 'True'
guess_validation_answer = 'False'
# Would you like to use a model to provide an initial guess?
guess_model = 'True'
guess_model = 'False'
# Number of samples to generate and where you stopped last time
seed_count = 100
seedling = 0
# Where should the outputs be saved?
folder_outputs = "F:\\single"
folder_outputs = "F:\\instances_seed_play"
folder_outputs = "F:\\instances_5000"
folder_outputs = "F:\\instances"
folder_outputs = "F:\\instances_500"
folder_outputs = "F:\\generic"
model_file = "model_checkpoint.h5"
# INPUTS: END

# Estimate the time to run and the time remaining
# Initial guess is based on it takes roughly 1 second to establish scene and roughly 5.8 per instance to solve
toc0_1 = (1.0 + 5.8)
toc0_1 = (1.1 * 0.85)
time_estimate = seed_count*toc0_1
time_estimate_inital = time_estimate
print("time_estimate", time_estimate_inital, "seconds")
tic_total_start = time.time()
state = random.getstate()
# random_steps is the number of random operations called in the geometery generation stage.
random_steps = 2
state_old = random.getstate()
for _ in range(random_steps*seedling):
    random_num = random.uniform(0, 9)
state_updated = random.getstate()
print("state_old == state_updated", state_old == state_updated)
os.makedirs(folder_outputs, exist_ok=True)

_, _, _, _, _, _, _, _, _, _, _, X1, X2, _, _, _, _, _ = custom_functions_EM.initEM()
file_name = f"X_array.npy"
output_file_path = os.path.join(folder_outputs + "\\" + file_name)
np.save(output_file_path, np.stack([X1, X2], axis=2))
if validation == 'True':
    c_0, eps_sct, mu_sct, gamma_0, xS, NR, rcvr_phi, xR, N1, N2, dx, X1, X2, FFTG, a, CHI_eps, CHI_mu, Errcri = custom_functions_EM.initEM(bessel=1)
    custom_functions_EM.plotEMContrast(CHI_eps, CHI_mu, X1, X2)
    E_inc, ZH_inc = custom_functions_EM.IncEMwave(gamma_0, xS, dx, X1, X2)
    tic0 = time.time()
    w_E, exit_code, information = custom_functions_EM.ITERBiCGSTABwE(E_inc, CHI_eps, Errcri, N1, N2, dx, FFTG, gamma_0, x0=None)
    toc0 = time.time() - tic0
    print("toc", toc0)
    Edata, Hdata = custom_functions_EM.DOPwE(w_E, gamma_0, dx, xR, NR, X1, X2)
    Edata2D, Hdata2D = custom_functions_EM.EMsctCircle()
    angle = rcvr_phi * 180 / np.pi
    # displayDataBesselApproach(Edata, angle)
    # displayDataBesselApproach(Hdata, angle)
    custom_functions_EM.displayDataCompareApproachs(Edata2D, Edata, angle)
    custom_functions_EM.displayDataCompareApproachs(Hdata2D, Hdata, angle)

    E_sct = custom_functions_EM.KopE(w_E, gamma_0, N1, N2, dx, FFTG)
    # Set the first row, last row, first column, and last column to zeros due to gradient calculation
    # This may make assisting the solver worse!
    E_sct[:, [0, -1], :] = E_sct[:, :, [0, -1]] = 0

    # Drop the first and last columns and rows due to finite differences at border
    custom_functions_EM.plotEtotalwavefield(E_inc[:, 1:-1, 1:-1], a, X1[1:-1, 1:-1], X2[1:-1, 1:-1], N1-1, N2-1)
    custom_functions_EM.plotEtotalwavefield(ZH_inc[[2, 2], 1:-1, 1:-1], a, X1[1:-1, 1:-1], X2[1:-1, 1:-1], N1-1, N2-1)
    custom_functions_EM.plotEtotalwavefield(E_sct[:, 1:-1, 1:-1], a, X1[1:-1, 1:-1], X2[1:-1, 1:-1], N1-1, N2-1)
    E_val = custom_functions_EM.E(E_inc, E_sct)
    custom_functions_EM.plotEtotalwavefield(E_val[:, 1:-1, 1:-1], a, X1[1:-1, 1:-1], X2[1:-1, 1:-1], N1-1, N2-1)

else:
    # START OF ForwardBiCGSTABFFTwE
    if guess_model == 'True':
        if os.path.exists(model_file):
            # model = load_model(model_file, custom_objects={'edge_loss': edge_loss})
            model = load_model(model_file)
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanSquaredError()])
        else:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("No model file found!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            sys.exit(1)
    for seed in range(seedling, seedling+seed_count):
        file_name = f"instance_{str(seed).zfill(10)}.npy"
        if guess_model == 'True':
            output_file_path = os.path.join(folder_outputs, os.path.splitext(file_name)[0] + "_m" + ".npy")
            if os.path.exists(output_file_path):
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Seed has been calculated. Examine inputs.")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                break
        else:
            output_file_path = os.path.join(folder_outputs, os.path.splitext(file_name)[0] + "_o" + ".npy")
            if os.path.exists(output_file_path):
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Seed has been calculated. Examine inputs.")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                break

        c_0, eps_sct, mu_sct, gamma_0, xS, NR, rcvr_phi, xR, N1, N2, dx, X1, X2, FFTG, a, CHI_eps, CHI_mu, Errcri = custom_functions_EM.initEM()
        E_inc, ZH_inc = custom_functions_EM.IncEMwave(gamma_0, xS, dx, X1, X2)

        if guess_model != 'True':
            plt.imsave(os.path.join(folder_outputs, f"instance_{str(seed).zfill(10)}_abs_o.png"), np.abs(CHI_eps), cmap='gray')
            # custom_functions_EM.plotEMContrast(CHI_eps, CHI_mu, X1, X2)
        else:
            # # Save visulistion of geometry for debugging reference
            # plt.imsave(os.path.join(folder_outputs, f"instance_{str(seed).zfill(10)}_abs_m.png"), np.abs(CHI_eps), cmap='gray')
            # # custom_functions_EM.plotEMContrast(CHI_eps, CHI_mu, X1, X2)
            # NEED TO EXTRACT EXISTING INFORMATION HERE TO REPLACE keras_stack items from init above as cannot trust seed method
            file_name = f"instance_{str(seed).zfill(10)}.npy"
            output_file_path = os.path.join(folder_outputs, os.path.splitext(file_name)[0] + "_o.npy")
            data = np.load(output_file_path)
            CHI_eps = np.squeeze(data[:, :, :, 0])
            E_inc[0, :, :] = np.squeeze(data[:, :, :, 1] + 1j*data[:, :, :, 2])
            E_inc[1, :, :] = np.squeeze(data[:, :, :, 4] + 1j*data[:, :, :, 5])
            E_inc[2, :, :] = E_inc[1, :, :]*0
            ZH_inc[2, :, :] = np.squeeze(data[:, :, :, 7] + 1j*data[:, :, :, 8])
            ZH_inc[1, :, :] = ZH_inc[2, :, :]*0
            ZH_inc[0, :, :] = ZH_inc[2, :, :]*0

        # MODEL INFERENCE
        tic0 = time.time()
        if guess_model == 'True':
            N = np.shape(CHI_eps)[0]*np.shape(CHI_eps)[1]
            x0 = np.zeros((2*N, 1), dtype=np.complex128, order='F')
            # x0[0:N, 0] = np.multiply(CHI_eps, E_inc[0, :, :]).flatten('F')
            # x0[N:2*N, 0] = np.multiply(CHI_eps, E_inc[1, :, :]).flatten('F')

            # np.expand_dims(np.real(CHI_eps), axis=0).shape
            # custom_functions_EM.complex_separation(E_inc[0, :, :])[0, :, :].shape
            keras_stack = np.concatenate([np.expand_dims(np.real(CHI_eps), axis=0),
                                          custom_functions_EM.complex_separation(E_inc[0, :, :])], axis=0)
                                          # np.expand_dims(custom_functions_EM.complex_separation(E_inc[0, :, :])[0, :, :], axis=0)], axis=0)
            # keras_stack.shape
            keras_stack_composed = custom_functions_EM.keras_format(keras_stack)
            input_data = keras_stack_composed.copy()
            # input_data = np.concatenate([np.expand_dims(data[:, :, :, 0], axis=-1), np.expand_dims(data[:, :, :, 3], axis=-1)], axis=-1)
            # x_list.append(input_data)

            # # output_data = np.concatenate([np.expand_dims(data[:, :, :, 10], axis=-1), np.expand_dims(data[:, :, :, 11], axis=-1)], axis=-1)
            # output_data = np.concatenate([np.expand_dims(data[:, :, :, 10], axis=-1), np.expand_dims(data[:, :, :, 11], axis=-1)], axis=-1)
            # y_list.append(output_data)

            # predicted_output = model.predict(input_data)
            predicted_output = model.predict(input_data)
            predicted_output = np.squeeze(predicted_output)
            predicted_output = np.transpose(predicted_output, (2, 0, 1))

            mean_per_channel = np.load(folder_outputs + "\\mean_per_channel.npy")
            adjusted_stddev_per_channel = np.load(folder_outputs + "\\adjusted_stddev_per_channel.npy")
            for channel in range(predicted_output.shape[0]):
                predicted_output[channel, :, :] = predicted_output[channel, :, :] * adjusted_stddev_per_channel[channel] + mean_per_channel[channel]
            w_E = E_inc.copy()

            # OPTION A: E_sct
            E_sct = E_inc.copy()
            E_sct[0, :, :] = predicted_output[0, :, :] + 1j*predicted_output[1, :, :]
            E_sct[1, :, :] = 0*E_inc[1, :, :].copy()
            E_val = custom_functions_EM.E(E_inc, E_sct)
            w_E[0, :, :] = CHI_eps * E_val[0, :, :]
            w_E[1, :, :] = CHI_eps * E_val[1, :, :]
            ###

            # OPTION B: STRAIGHT W_E
            # w_E[0, :, :] = predicted_output[0, :, :] + 1j*predicted_output[1, :, :]
            # w_E[1, :, :] = np.multiply(CHI_eps, E_inc[1, :, :])
            # w_E_old = w_E.copy()
            # E_sct = custom_functions_EM.KopE(w_E, gamma_0, N1, N2, dx, FFTG)
            ###

            # custom_functions_EM.plotEtotalwavefield(w_E[:, 1:-1, 1:-1], a, X1[1:-1, 1:-1], X2[1:-1, 1:-1], N1-1, N2-1)
            # custom_functions_EM.plotEtotalwavefield(E_sct[:, 1:-1, 1:-1], a, X1[1:-1, 1:-1], X2[1:-1, 1:-1], N1-1, N2-1)
            # print("np.linalg.norm(w_E_old - w_E): ", np.linalg.norm(w_E_old - w_E))
            x0[0:N, 0] = w_E[0, :, :].flatten('F')
            x0[N:2*N, 0] = w_E[1, :, :].flatten('F')
            w_E, exit_code, information = custom_functions_EM.ITERBiCGSTABwE(E_inc, CHI_eps, Errcri, N1, N2, dx, FFTG, gamma_0, x0=x0)
        else:
            w_E, exit_code, information = custom_functions_EM.ITERBiCGSTABwE(E_inc, CHI_eps, Errcri, N1, N2, dx, FFTG, gamma_0, x0=None)
        toc0 = time.time() - tic0
        print("toc", toc0)
        # custom_functions_EM.plotEtotalwavefield(w_E[:, 1:-1, 1:-1], a, X1[1:-1, 1:-1], X2[1:-1, 1:-1], N1-1, N2-1)

        if exit_code == 0:
            E_sct = custom_functions_EM.KopE(w_E, gamma_0, N1, N2, dx, FFTG)

            # Set the first row, last row, first column, and last column to zeros due to gradient calculation
            # This may make assisting the solver worse but should make training the model easier!
            E_sct[:, [0, -1], :] = E_sct[:, :, [0, -1]] = 0
            custom_functions_EM.plotEtotalwavefield(E_sct[:, 1:-1, 1:-1], a, X1[1:-1, 1:-1], X2[1:-1, 1:-1], N1-1, N2-1)
            E_val = custom_functions_EM.E(E_inc, E_sct)
            # custom_functions_EM.plotEtotalwavefield(E_val[:, 1:-1, 1:-1], a, X1[1:-1, 1:-1], X2[1:-1, 1:-1], N1-1, N2-1)

            # Save incident fields to allow for data augmentation
            # Not including any fields that are totally constant such as CHI_mu and the dead fields E3, ZH1, ZH2
            # CHI_eps only has real part so CHI_eps has 1 field which will be cast as real for safety
            keras_stack = np.concatenate([np.expand_dims(np.real(CHI_eps), axis=0),
                                          custom_functions_EM.complex_separation(E_inc[0, :, :]),
                                          custom_functions_EM.complex_separation(E_inc[1, :, :]),
                                          custom_functions_EM.complex_separation(ZH_inc[2, :, :]),
                                          custom_functions_EM.complex_separation(E_sct[0, :, :]),
                                          custom_functions_EM.complex_separation(E_sct[1, :, :])], axis=0)

            # Due to channel requirements in Keras, reorder arrays before saving to reduce loading time at deep learning stage.
            keras_stack_composed = custom_functions_EM.keras_format(keras_stack)
            # 16 dimensional array in format (:, N1, N2, channels)
            # np.save(os.path.join(folder_outputs, f"instance_{str(seed).zfill(10)}.npy"), keras_stack_composed)

            file_name = f"instance_{str(seed).zfill(10)}.npy"
            if guess_model == 'True':
                output_file_path = os.path.join(folder_outputs, os.path.splitext(file_name)[0] + "_m")
                np.save(output_file_path, keras_stack_composed)
                output_file_path_info = os.path.join(folder_outputs, os.path.splitext(file_name)[0] + "_info_m")
                np.save(output_file_path_info, information)
            else:
                output_file_path = os.path.join(folder_outputs, os.path.splitext(file_name)[0] + "_o")
                np.save(output_file_path, keras_stack_composed)
                output_file_path_info = os.path.join(folder_outputs, os.path.splitext(file_name)[0] + "_info_o")
                np.save(output_file_path_info, information)

        if guess_validation_answer == 'True':
            # Test Initial Guess as final answer
            N = np.shape(CHI_eps)[0]*np.shape(CHI_eps)[1]
            tic1 = time.time()
            x0 = np.zeros((2*N, 1), dtype=np.complex128, order='F')
            w_E_old = w_E
            E_sct = custom_functions_EM.KopE(w_E, gamma_0, N1, N2, dx, FFTG)
            E_val = custom_functions_EM.E(E_inc, E_sct)
            w_E[0, :, :] = CHI_eps * E_val[0, :, :]
            w_E[1, :, :] = CHI_eps * E_val[1, :, :]
            # custom_functions_EM.plotEtotalwavefield(w_E[:, 1:-1, 1:-1], a, X1[1:-1, 1:-1], X2[1:-1, 1:-1], N1-1, N2-1)
            print("np.linalg.norm(w_E_old - w_E): ", np.linalg.norm(w_E_old - w_E))
            x0[0:N, 0] = w_E[0, :, :].flatten('F')
            x0[N:2*N, 0] = w_E[1, :, :].flatten('F')
            w_E_m, exit_code_m, information_m = custom_functions_EM.ITERBiCGSTABwE(E_inc, CHI_eps, Errcri, N1, N2, dx, FFTG, gamma_0, x0=x0)
            toc1 = time.time() - tic1
            print("toc", toc1)
        time_estimate = time_estimate-((toc0+toc0_1)/2)
        toc0_1 = toc0
        print("remaining time_estimate: ", time_estimate, " in seconds")
        print("remaining time_estimate: ", time_estimate/60, " in minutes")
        print("remaining time_estimate: ", time_estimate/3600, " in hours")
        print("remaining seeds: ", (seedling + seed_count - 1) - seed)
        # custom_functions_EM.plotContrastSourcewE(w_E, X1, X2)
        # custom_functions_EM.plotEtotalwavefield(E_inc, a, X1, X2, N1, N2)

        # # Drop the first and last columns and rows due to finite differences at border
        # custom_functions_EM.plotEtotalwavefield(E_inc[:, 1:-1, 1:-1], a, X1[1:-1, 1:-1], X2[1:-1, 1:-1], N1-1, N2-1)
        # custom_functions_EM.plotEtotalwavefield(ZH_inc[[2, 2], 1:-1, 1:-1], a, X1[1:-1, 1:-1], X2[1:-1, 1:-1], N1-1, N2-1)
        # custom_functions_EM.plotEtotalwavefield(E_sct[:, 1:-1, 1:-1], a, X1[1:-1, 1:-1], X2[1:-1, 1:-1], N1-1, N2-1)
        # E_val = custom_functions_EM.E(E_inc, E_sct)
        # custom_functions_EM.plotEtotalwavefield(E_val[:, 1:-1, 1:-1], a, X1[1:-1, 1:-1], X2[1:-1, 1:-1], N1-1, N2-1)
tic_total_end = time.time() - tic_total_start
print("Total Running Time: ", tic_total_end)
print("Initial guess of running time was ", time_estimate_inital, " so (tic_total_end - time_estimate_inital): ", tic_total_end - time_estimate_inital)

print("start stats")
info_dataset = custom_functions_EM.info_data_harvest(folder_outputs)
custom_functions_EM.info_data_paired('.\\doc\\_stats\\dataset_instances_output.csv')
