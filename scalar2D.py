""" This file is the script for the Scalar 2D VDB code
Time factor = exp(-iwt)
Spatial units is in m
Source wavelet  Q = 1
"""
from IPython import get_ipython
from lib import solver_func
from lib import workspace_func
from lib import solveremf2_plot
import numpy as np
import sys
import time
from scipy.io import loadmat, savemat

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=20)

# USER INPUTS
loop_counter = 0
N1 = solver_func.N1(120)
N2 = solver_func.N2(100)
dx = solver_func.dx(2.0)
itmax = solver_func.itmax(1000)
xS = solver_func.xS(-170, 0)
x0_naive = solver_func.x0_naive(N1, N2)
Errcri = solver_func.Errcri(1e-13)
c_0 = solver_func.c_0(1500.0)
f = solver_func.f(50)
a = solver_func.a(40)
contrast_sct = solver_func.contrast_sct(2)

# for contrast_sct_parameter in np.arange(1, 2, 0.1):
#     contrast_sct = solver_func.contrast_sct(contrast_sct_parameter)
# for a_parameter in np.arange(5, 50, 10):
#     a = solver_func.a(a_parameter)
# for xS_parameter in np.arange(0, 361, 45):
#     x_pol = 170 * np.cos(xS_parameter * np.pi/180.0)
#     y_pol = 170 * np.sin(xS_parameter * np.pi/180.0)
#     xS = solver_func.xS(x_pol, y_pol)

for holder_loop in np.arange(0, 1, 1):

    # HOW TO LOOP GEOMETERY? THIS MEANS CREATING A GEOMETRY SECTION.
    # NEED TO SAVE OUTPUTS SOMEHOW SYSTEMATICALLY

    c_sct = solver_func.c_sct(c_0, contrast_sct)
    s = solver_func.s(f)
    wavelength = solver_func.wavelength(c_0, f)
    gamma_0 = solver_func.gamma_0(s, c_0)
    initGrid = solver_func.initGrid(N1, N2, dx)
    X1cap = solver_func.X1cap(initGrid)
    X2cap = solver_func.X2cap(initGrid)
    x1fft = solver_func.x1fft(N1, dx)
    x2fft = solver_func.x2fft(N2, dx)
    initFFTGreen = solver_func.initFFTGreen(x1fft, x2fft)
    X1fftcap = solver_func.X1fft(initFFTGreen)
    X2fftcap = solver_func.X2fft(initFFTGreen)
    delta = solver_func.delta(dx)
    IntG = solver_func.IntG(dx, gamma_0, X1fftcap, X2fftcap, N1, N2, delta)
    FFTG = solver_func.FFTG(IntG)
    R = solver_func.R(X1cap, X2cap)

    CHI = solver_func.CHI(c_0, c_sct, R, a)

    factoru = solver_func.factoru(gamma_0, delta)
    u_inc = solver_func.u_inc(gamma_0, xS, X1cap, X2cap, factoru)

    b = solver_func.b(CHI, u_inc, N1, N2)

    if loop_counter == 0:
        x0 = x0_naive

    time_total, w, exit_code, iterative_info = solver_func.data_gen(b, CHI, FFTG, N1, N2, Errcri, itmax, x0)
    if exit_code != 0:
        break
    solveremf2_plot.plotContrastSource(w, CHI, X1cap, X2cap)
    # savemat('w_P.mat', {'w': w_model})

    # x0_model = w_naive.flatten('F') * np.random.rand(*w_naive.shape).flatten('F')
    x0 = w.flatten('F')

    print("loop_counter", loop_counter)
    loop_counter += 1


solveremf2_plot.subplotsComplexArray(u_inc, cmap_min=0, cmap_max=1)


print("real min", np.min(np.real(u_inc)))
print("real max", np.max(np.real(u_inc)))
print("imag min", np.min(np.imag(u_inc)))
print("imag max", np.max(np.imag(u_inc)))
print("abs min", np.min(np.abs(u_inc)))
print("abs max", np.max(np.abs(u_inc)))


def plotComplexArray(cmd, array, cmap_name, cmap_min, cmap_max, title):
    import numpy as np
    from skimage import io
    import matplotlib.pyplot as plt
    from matplotlib import cm
    if cmd == 'real':
        array = np.real(array)
    elif cmd == 'imag':
        array = np.imag(array)
    elif cmd == 'abs':
        array = np.abs(array)
    else:
        print("error input cmd string")
    normalized_array = (array - cmap_min) / (cmap_max - cmap_min)
    colormap = cm.get_cmap('jet')
    colored_array = colormap(normalized_array)
    colored_array = (colored_array * 255).astype(np.uint8)
    io.imsave('output.png', colored_array)

    # Plot the image
    plt.imshow(colored_array)

    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('Value')

    # Display the plot
    plt.show()


plotComplexArray(cmd='real', array=u_inc, cmap_name='jet', cmap_min=0, cmap_max=0.125, title='real_part')
plotComplexArray(cmd='imag', array=u_inc, cmap_name='jet', cmap_min=0, cmap_max=0.125, title='real_part')
plotComplexArray(cmd='abs', array=u_inc, cmap_name='jet', cmap_min=0, cmap_max=0.125, title='real_part')

# bessel_check = 1
# if bessel_check == 1:
#     M = solver_func.M(100)
#     NR = solver_func.NR(180)
#     rcvr_phi = solver_func.rcvr_phi(NR)
#     xR = solver_func.xR(NR, rcvr_phi)
#     Dop_val = solver_func.Dop(w, gamma_0, dx, xR, NR, X1cap, X2cap, delta, factoru, N1, N2)
#     angle = solver_func.angle(rcvr_phi)
#     solveremf2_plot.displayDataCSIEApproach(Dop_val, angle)
#     gamma_sct = solver_func.gamma_sct(gamma_0, c_0, c_sct)
#     arg0 = solver_func.arg0(gamma_0, a)
#     args = solver_func.args(gamma_sct, a)
#     rR = solver_func.rR(xR)
#     phiR = solver_func.phiR(xR)
#     rS = solver_func.rS(xS)
#     phiS = solver_func.phiS(xS)
#     WavefieldSctCircle = solver_func.WavefieldSctCircle(M, arg0, args, gamma_sct, gamma_0, xR, xS, rR, phiR, rS, phiS)
#     solveremf2_plot.displayDataBesselApproach(WavefieldSctCircle, angle)
#     solveremf2_plot.displayDataCompareApproachs(WavefieldSctCircle, Dop_val, angle)

# # Validate code against MATLAB output
# if (c_0 == 1500) and (c_sct == 3000) and (f == 50) and (itmax == 1000) and (Errcri == 1e-13):
#     savemat('w_P.mat', {'w': w})
#     var_name_pyt = loadmat('w_P.mat')['w']
#     var_name_mat = loadmat('./code_ref/ScalarWavesMfiles/w_mat.mat')['w']

#     var_diff = var_name_pyt - var_name_mat
#     np.max(var_diff)
#     np.max(np.real(var_diff))
#     np.max(np.imag(var_diff))
#     workspace_func.plotDiff(var_diff, X1cap, X2cap)
#     print("Comaprision made...")
#     # os.remove('w_P.mat')

# workspace_func.tidy_workspace()
