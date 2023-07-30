from lib import custom_functions
import pickle
from IPython import get_ipython


# # Clear workspace
# from numba import cuda
# cuda.select_device(0)
# cuda.close()

get_ipython().run_line_magic('clear', '-sf')


# Load the training history from the pickle file
history_file = "training_history.pkl"
with open(history_file, 'rb') as file:
    history = pickle.load(file)
print(history.keys())
# print(history.history.keys())

ignore_entries = 0
result_dict = custom_functions.plot_history_ignore(history, ignore_entries)
custom_functions.plot_loss(result_dict)

ignore_entries = 10
result_dict = custom_functions.plot_history_ignore(history, ignore_entries)
custom_functions.plot_loss(result_dict)

ignore_entries = 110
result_dict = custom_functions.plot_history_ignore(history, ignore_entries)
custom_functions.plot_loss(result_dict)

ignore_entries = 210
result_dict = custom_functions.plot_history_ignore(history, ignore_entries)
custom_functions.plot_loss(result_dict)

ignore_entries = 310
result_dict = custom_functions.plot_history_ignore(history, ignore_entries)
custom_functions.plot_loss(result_dict)

ignore_entries = 410
result_dict = custom_functions.plot_history_ignore(history, ignore_entries)
custom_functions.plot_loss(result_dict)

ignore_entries = 510
result_dict = custom_functions.plot_history_ignore(history, ignore_entries)
custom_functions.plot_loss(result_dict)

ignore_entries = 610
result_dict = custom_functions.plot_history_ignore(history, ignore_entries)
custom_functions.plot_loss(result_dict)

ignore_entries = 710
result_dict = custom_functions.plot_history_ignore(history, ignore_entries)
custom_functions.plot_loss(result_dict)

ignore_entries = 810
result_dict = custom_functions.plot_history_ignore(history, ignore_entries)
custom_functions.plot_loss(result_dict)

ignore_entries = 910
result_dict = custom_functions.plot_history_ignore(history, ignore_entries)
custom_functions.plot_loss(result_dict)


print("start stats")
output_folder = "F:\\instances_output_0000000000-0000004999"
info_dataset = custom_functions.info_data_harvest(output_folder)
custom_functions.info_data_paired('.\\doc\\_stats\\dataset_instances_output_0000000000-0000004999.csv')


ignore_entries = 410
# Load the first training history
with open('training_history_sobel_L+.pkl', 'rb') as file:
    training_history1 = pickle.load(file)
    training_history1 = custom_functions.plot_history_ignore(training_history1, ignore_entries)

# Load the second training history
with open('training_history_sobel.pkl', 'rb') as file:
    training_history2 = pickle.load(file)
    training_history2 = custom_functions.plot_history_ignore(training_history2, ignore_entries)


for epoch in range(len(training_history1['loss'])):
    loss1 = training_history1['loss'][epoch]
    loss2 = training_history2['loss'][epoch]
    val_loss1 = training_history1['val_loss'][epoch]
    val_loss2 = training_history2['val_loss'][epoch]

    # Compare the loss and accuracy values
    if loss1 != loss2:
        print(f"Loss is different in epoch {epoch+1}: {loss1} vs {loss2}")

    if val_loss1 != val_loss2:
        print(f"val_loss is different in epoch {epoch+1}: {val_loss1} vs {val_loss2}")


import matplotlib.pyplot as plt

# Plot the loss
plt.plot(training_history1['loss'], label='Training History 1')
plt.plot(training_history2['loss'], label='Training History 2')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the accuracy
plt.plot(training_history1['val_loss'], label='Training History 1')
plt.plot(training_history2['val_loss'], label='Training History 2')
plt.xlabel('Epoch')
plt.ylabel('val_loss')
plt.legend()
plt.show()



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
            custom_functions.Aw,
            custom_functions.b,
            # custom_functions.callback,
            custom_functions.CHI,
            custom_functions.CHI_Bessel,
            custom_functions.complex_separation,
            custom_functions.composer_render,
            custom_functions.contrast_sct,
            # custom_functions.custom_matvec,
            custom_functions.c_0,
            custom_functions.c_sct,
            custom_functions.displayDataBesselApproach,
            custom_functions.displayDataCompareApproachs,
            custom_functions.displayDataCSIEApproach,
            custom_functions.Dop,
            custom_functions.dx,
            custom_functions.epsilon0,
            custom_functions.Errcri,
            custom_functions.exit_code,
            custom_functions.f,
            custom_functions.FFTG,
            custom_functions.gamma_0,
            custom_functions.generate_ROI,
            custom_functions.Green,
            custom_functions.information,
            custom_functions.info_data_harvest,
            custom_functions.info_data_paired,
            # custom_functions.init,
            custom_functions.initFFTGreenGrid,
            custom_functions.initGrid,
            custom_functions.input_disc_per_lambda,
            custom_functions.ITERBiCGSTABw,
            custom_functions.itmax,
            custom_functions.Kop,
            custom_functions.lambda_smallest,
            custom_functions.mu0,
            custom_functions.N1,
            custom_functions.N2,
            custom_functions.NR,
            custom_functions.plotContrastSource,
            custom_functions.prescient2DL_data,
            custom_functions.R,
            custom_functions.radius_receiver,
            custom_functions.radius_source,
            custom_functions.rcvr_phi,
            custom_functions.s,
            custom_functions.tissuePermittivity,
            custom_functions.unet_elu,
            custom_functions.u_inc,
            custom_functions.w,
            custom_functions.WavefieldSctCircle,
            custom_functions.wavelength,
            custom_functions.X1,
            custom_functions.X1fft,
            custom_functions.X2,
            custom_functions.X2fft,
            custom_functions.xR,
            custom_functions.xS
        )
        # .update_parameters(input_length_side=input_length_x_side)
        # .cache()
    )
    return composer_1


custom_functions.composer_render(composer_call(), path_doc, "function_flow")
# help(custom_functions.gamma_0)
