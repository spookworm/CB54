from lib import custom_functions
import pickle
from IPython import get_ipython


# # Clear workspace
# from numba import cuda
# cuda.select_device(0)
# cuda.close()

get_ipython().run_line_magic('clear', '-sf')


# Load the training history from the pickle file
with open('training_history.pkl', 'rb') as file:
    history = pickle.load(file)
print(history.keys())
# print(history.history.keys())
custom_functions.plot_loss(history)


# print("start stats")
# output_folder = "F:\\instances_output"
# info_dataset = custom_functions.info_data_harvest(output_folder)
# custom_functions.info_data_paired('.\\doc\\_stats\\dataset_instances_output.csv')

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
