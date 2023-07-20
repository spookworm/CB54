import tensorflow as tf
from IPython import get_ipython
import numpy as np
import sys
import time
from lib import custom_functions

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')
print(tf.config.list_physical_devices('GPU'))

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=20)

"""
Description:
This file is the script for the adapted Scalar 2D VDB code with the Bessel-Approach validation.
    Time factor = exp(-iwt)
    Spatial units is in m
    Source wavelet  Q = 1
"""

# USER INPUTS
c_0 = custom_functions.c_0(1500.0)
contrast_sct = custom_functions.contrast_sct(2.0)
f = custom_functions.f(50.0)
NR = custom_functions.NR(180)
radius_source = custom_functions.radius_source(170.0)
radius_receiver = custom_functions.radius_receiver(150.0)
N1 = custom_functions.N1(120)
N2 = custom_functions.N2(100)
dx = custom_functions.dx(2)
a = custom_functions.a(40.0)
Errcri = custom_functions.Errcri(1e-18)


c_sct = custom_functions.c_sct(c_0, contrast_sct)
wavelength = custom_functions.wavelength(c_0, f)
s = custom_functions.s(f)
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

CHI_array = custom_functions.CHI_Bessel(c_0, c_sct, R, a)
CHI = custom_functions.CHI(CHI_array)

data2D = custom_functions.WavefieldSctCircle(c_0, c_sct, gamma_0, xS, NR, rcvr_phi, xR, N1, N2, dx, X1, X2, FFTG, a, CHI, Errcri)

u_inc = custom_functions.u_inc(gamma_0, xS, dx, X1, X2)
itmax = custom_functions.itmax(CHI)
b = custom_functions.b(CHI, u_inc)

tic0 = time.time()
ITERBiCGSTABw = custom_functions.ITERBiCGSTABw(u_inc, CHI, Errcri, N1, N2, b, FFTG, itmax, x0=None)
toc0 = time.time() - tic0
w = custom_functions.w(ITERBiCGSTABw)
exit_code = custom_functions.exit_code(ITERBiCGSTABw)
information = custom_functions.information(ITERBiCGSTABw)


custom_functions.plotContrastSource(w, CHI, X1, X2)
data = custom_functions.Dop(w, NR, N1, N2, xR, gamma_0, dx, X1, X2)
angle = custom_functions.angle(rcvr_phi)
custom_functions.displayDataCSIEApproach(data, angle)

custom_functions.displayDataCompareApproachs(data2D, data, angle)
error = str(np.linalg.norm(data.flatten('F') - data2D.flatten('F'), ord=1)/np.linalg.norm(data2D.flatten('F'), ord=1))
print("error", error)


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
            custom_functions.Aw,
            custom_functions.b,
            custom_functions.CHI,
            custom_functions.CHI_Bessel,
            custom_functions.contrast_sct,
            custom_functions.c_0,
            custom_functions.c_sct,
            custom_functions.Dop,
            custom_functions.dx,
            custom_functions.Errcri,
            custom_functions.exit_code,
            custom_functions.f,
            custom_functions.gamma_0,
            custom_functions.Green,
            # custom_functions.init,
            custom_functions.FFTG,
            custom_functions.information,
            custom_functions.initFFTGreenGrid,
            custom_functions.initGrid,
            custom_functions.ITERBiCGSTABw,
            custom_functions.itmax,
            custom_functions.N1,
            custom_functions.N2,
            custom_functions.NR,
            custom_functions.plotContrastSource,
            custom_functions.R,
            custom_functions.radius_receiver,
            custom_functions.radius_source,
            custom_functions.rcvr_phi,
            custom_functions.s,
            custom_functions.u_inc,
            custom_functions.w,
            custom_functions.wavelength,
            custom_functions.WavefieldSctCircle,
            custom_functions.X1,
            custom_functions.X2,
            custom_functions.X1fft,
            custom_functions.X2fft,
            custom_functions.xR,
            custom_functions.xS,
        )
        # .update_parameters(input_length_side=input_length_x_side)
        # .cache()
    )
    return composer_1


custom_functions.composer_render(composer_call(), path_doc, "function_flow_bessel")
# help(custom_functions.gamma_0)
