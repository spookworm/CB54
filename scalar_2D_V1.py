""" This file is the script for the Scalar 2D VDB code
Time factor = ForwardBiCGSTABFFT.exp(-iwt)
Spatial units is in m
Source wavelet  Q = ForwardBiCGSTABFFT.1
"""

from IPython import get_ipython

import numpy as np
import time
import sys
import scipy.io
from varname import nameof
from scipy.special import kv, iv
from scipy.sparse.linalg import bicgstab, LinearOperator
import matplotlib.pyplot as plt
try:
    from lib import ForwardBiCGSTABFFT
except ImportError:
    import ForwardBiCGSTABFFT

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=18)


def mat_loader(var_name):
    # x1fft_m = mat_loader(nameof(x1fft))
    var_name_matlab = scipy.io.loadmat(var_name + '.mat')
    var_name_m = np.array(var_name_matlab[var_name])
    return var_name_m


def mat_checker(variable, var_name):
    # mat_checker(x1fft, nameof(x1fft))
    var_name_matlab = scipy.io.loadmat(var_name + '.mat')
    var_name_m = np.array(var_name_matlab[var_name])
    are_equal = np.array_equal(variable, var_name_m)
    if are_equal is False:
        print(var_name, "are_equal:", are_equal)
    are_close = np.allclose(variable, var_name_m, atol=1e-15)
    if are_close is False:
        print(var_name, "are_close:", are_close)
    var_diff = variable - var_name_m
    max_test = np.max(var_diff)
    if max_test != 0.0:
        print(var_name, "max diff: ", max_test)
        return var_diff
    else:
        return None


c_0 = 1500.0
c_sct = 3000.0
f = 50.0
wavelength = c_0 / f
s = 1e-16 - 1j * 2 * np.pi * f
gamma_0 = s / c_0
Errcri = 1e-15

# initSourceReceiver(input)
xS = np.zeros((2, 1), dtype=float)
xS = [-170, 0]

NR = 180

rcvr_phi = np.zeros((1, NR), dtype=float, order='F')
# rcvr_phi(1:input.NR) = (1:input.NR) * 2 * pi / input.NR;
rcvr_phi[0, 0:NR] = np.arange(1, NR+1, 1) * 2.0 * np.pi / NR
# mat_checker(rcvr_phi, nameof(rcvr_phi))


xR = np.zeros((2, NR), dtype=float, order='F')
xR[0, 0:NR] = 150 * np.cos(rcvr_phi)
xR[1, 0:NR] = 150 * np.sin(rcvr_phi)
# xR_diff = mat_checker(xR, nameof(xR))

N1 = 120
N2 = 100
dx = 2

# x1 = -(N1 + 1) * dx / 2 + (1:N1) * dx
x1 = np.zeros((1, N1), dtype=float, order='F')
x1[0, :] = -(N1 + 1) * dx / 2 + np.arange(1, N1+1, 1) * dx
# x1_diff = mat_checker(x1, nameof(x1))

# x2 = -(N2 + 1) * dx / 2 + (1:N2) * dx
x2 = np.zeros((1, N2), dtype=float, order='F')
x2[0, :] = -(N2 + 1) * dx / 2 + np.arange(1, N2+1, 1) * dx
# x2_diff = mat_checker(x2, nameof(x2))

# [X1cap, X2cap] = ndgrid(x1, x2);
X1cap, X2cap = np.meshgrid(x1, x2, indexing='ij')

# X1cap_diff = mat_checker(X1cap, nameof(X1cap))
# X2cap_diff = mat_checker(X2cap, nameof(X2cap))

# initFFTGreen(input)

N1fft = int(2**np.ceil(np.log2(2*N1)))
N2fft = int(2**np.ceil(np.log2(2*N2)))

# x1(1:N1fft) = [0 : N1fft / 2 - 1, N1fft / 2 : -1 : 1] * input.dx;
x1fft = np.concatenate((np.arange(0, N1fft/2), np.arange(N1fft/2, 0, -1))) * dx
x1fft = np.reshape(x1fft, (1, N1fft), order='F')
# x1fft_diff = mat_checker(x1fft, nameof(x1fft))

# x2(1:N2fft) = [0 : N2fft / 2 - 1, N2fft / 2 : -1 : 1] * input.dx;
x2fft = np.concatenate((np.arange(0, N2fft/2), np.arange(N2fft/2, 0, -1))) * dx
x2fft = np.reshape(x2fft, (1, N2fft), order='F')
# x2fft_diff = mat_checker(x2fft, nameof(x2fft))

# [temp.X1fft, temp.X2fft] = ndgrid(x1, x2);
X1fftcap, X2fftcap = np.meshgrid(x1fft, x2fft, indexing='ij')
# X1fftcap_diff = mat_checker(X1fftcap, nameof(X1fftcap))
# X2fftcap_diff = mat_checker(X2fftcap, nameof(X2fftcap))

DIS = np.sqrt(X1fftcap**2 + X2fftcap**2)
DIS[0, 0] = 1
# DIS_diff = mat_checker(DIS, nameof(DIS))

# G = 1 / (2 * pi) .* besselk(0, gamma_0*DIS);
G = np.zeros((N1, N2), dtype=np.complex_, order='F')
G = 1 / (2 * np.pi) * kv(0, gamma_0*DIS)
# G_diff = mat_checker(G, nameof(G))

# G_real = np.real(G)
# G_imag = np.imag(G)
# G_real_diff = mat_checker(G_real, nameof(G_real))
# G_imag_diff = mat_checker(G_imag, nameof(G_imag))

# delta = (pi)^(-1 / 2) * dx; % radius circle with area of dx^2
delta = (np.pi)**(-0.5) * dx

# factor = 2 * besseli(1, gamma_0*delta) / (gamma_0 * delta);
factor = 2 * iv(1, gamma_0*delta) / (gamma_0*delta)

# IntG = (gamma_0^2 * dx^2) * factor^2 * G; % integral includes gamma_0^2
IntG = (gamma_0**2 * dx**2) * factor**2 * G
# IntG(1, 1) = 1 - gamma_0 * delta * besselk(1, gamma_0*delta) * factor;
IntG[0, 0] = 1 - gamma_0*delta * kv(1, gamma_0*delta) * factor
# IntG_diff = mat_checker(IntG, nameof(IntG))

FFTG = np.fft.fftn(IntG)
# FFTG_diff = mat_checker(FFTG, nameof(FFTG))

a = 40
contrast = 1 - c_0**2 / c_sct**2

R = np.sqrt(X1cap**2 + X2cap**2)
# R_diff = mat_checker(R, nameof(R))

CHI = contrast * (R < a)
# CHI_diff = mat_checker(CHI, nameof(CHI))


DISu = np.sqrt((X1cap - xS[0])**2 + (X2cap - xS[1])**2)
# DISu_diff = mat_checker(DISu, nameof(DISu))

Gu = 1 / (2 * np.pi) * kv(0, gamma_0*DISu)
# Gu_diff = mat_checker(Gu, nameof(Gu))

factoru = 2 * iv(1, gamma_0*delta) / (gamma_0*delta)

u_inc = factoru * Gu
# u_inc_diff = mat_checker(u_inc, nameof(u_inc))

itmax = 1000
# b = CHI(:) * u_inc(:)
b = np.zeros((N1*N2, 1), dtype=np.complex_, order='F')
# b = np.multiply(CHI.flatten('F'), u_inc.flatten('F'))
b[:, 0] = CHI.flatten('F') * u_inc.flatten('F')
# b_diff = mat_checker(b, nameof(b))
# b_m = mat_loader(nameof(b))


def ITERBiCGSTABw(b, CHI, u_inc, FFTG, N1, N2, Errcri, itmax):
    # BiCGSTAB_FFT scheme for contrast source integral equation Aw = b
    norm_b = np.linalg.norm(b)

    def callback(xk):
        # print("Current solution:", xk)
        # relative residual norm(b-A*x)/norm(b)
        callback.iter_count += 1
        # residual = np.linalg.norm(b - Aw_operator.dot(xk))/norm_b
        residual = norm_b
        # residuals.append(residual)
        if residual < Errcri:
            return True
        else:
            return False
        print("Iteration:", "\t", "Residual norm:", "\t", "Time:")
        print(residual)

    # Initialise iteration count
    callback.iter_count = 0
    callback.start_time = time.time()

    # Call bicgstab with the LinearOperator instance and other inputs
    # w = bicgstab(@(w) Aw(w, input), b, Errcri, itmax);
    Aw_operator = LinearOperator((b.shape[0], b.shape[0]), matvec=lambda w: Aw(w, N1, N2, FFTG, CHI))
    x0 = np.zeros(b.shape, dtype=np.complex_)
    start_time = time.time()
    w, exit_code = bicgstab(Aw_operator, b, x0=x0, tol=Errcri, maxiter=itmax, callback=callback)
    time_total = time.time() - start_time
    print("time_total", time_total)

    # Output Matrix
    # w = vector2matrix(w, N1, N2)
    w = w.reshape((N1, N2), order='F')

    # # Display the convergence information
    # print("Convergence information:", exit_code)
    # print(exit_code)
    print("Iteration:", callback.iter_count)
    # print("time_total", callback.time_total)
    return w


def Aw(w, N1, N2, FFTG, CHI):
    # print("w.shape", w.shape)
    # # Define the matrix-vector multiplication function Aw
    w = np.reshape(w, (N1, N2), order='F')
    # print("w.shape reshaped", w.shape)
    y = np.zeros((N1, N2), dtype=np.complex_, order='F')
    # print("y.shape zeros", y.shape)
    y = w - CHI * Kop(w, FFTG)
    # print("y.shape", y.shape)
    y = y.flatten('F')
    # print("y.shape", y.shape)
    return y


def Kop(w_E, FFTG):
    # print("w_E.shape", w_E.shape)
    # Make FFT grid
    N1, N2 = w_E.shape
    Cv = np.zeros(FFTG.shape, dtype=np.complex_, order='F')
    Cv[0:N1, 0:N2] = w_E.copy()
    # Convolution by FFT
    Cv = np.fft.fftn(Cv)
    Cv = np.fft.ifftn(FFTG * Cv)
    Kv = np.zeros((N1, N2), dtype=np.complex_, order='F')
    Kv[0:N1, 0:N2] = Cv[0:N1, 0:N2]
    return Kv


w_out = ITERBiCGSTABw(b, CHI, u_inc, FFTG, N1, N2, Errcri, itmax)

ITERBiCGSTABw = w_out.copy()

ITERBiCGSTABw_diff = mat_checker(ITERBiCGSTABw, nameof(ITERBiCGSTABw))
ITERBiCGSTABw_m = mat_loader(nameof(ITERBiCGSTABw))
ITERBiCGSTABw_m_real = np.real(ITERBiCGSTABw_m)
ITERBiCGSTABw_m_imag = np.imag(ITERBiCGSTABw_m)

ITERBiCGSTABw_real = np.real(ITERBiCGSTABw)
ITERBiCGSTABw_imag = np.imag(ITERBiCGSTABw)


ITERBiCGSTABw_m_real_diff = ITERBiCGSTABw_real - ITERBiCGSTABw_m_real
ITERBiCGSTABw_m_imag_diff = ITERBiCGSTABw_imag - ITERBiCGSTABw_m_imag

ForwardBiCGSTABFFT.plotContrastSource(ITERBiCGSTABw, CHI, X1cap, X2cap)


def Dop(w_out, gamma_0, dx, xR, NR, X1cap, X2cap, delta, factoru, N1, N2):
    data = np.zeros((1, NR), dtype=np.complex_, order='F')
    G = np.zeros((N1, N2), dtype=np.complex_, order='F')
    for p in range(0, NR):
        DIS = np.sqrt((xR[0, p] - X1cap)**2 + (xR[1, p] - X2cap)**2)
        G = 1.0 / (2.0 * np.pi) * kv(0, gamma_0*DIS)
        data[0, p] = (gamma_0**2 * dx**2) * factoru * np.sum(G.flatten('F') * w_out.flatten('F'))
    return data


# DIS_final = data.copy()
# DIS_final_m = mat_loader(nameof(DIS_final))
# DIS_final_diff = mat_checker(DIS_final, nameof(DIS_final))

Dop = Dop(w_out, gamma_0, dx, xR, NR, X1cap, X2cap, delta, factoru, N1, N2)
Dop_diff = mat_checker(Dop, nameof(Dop))
# Dop_m = mat_loader(nameof(Dop))


def angle(rcvr_phi):
    return rcvr_phi * 180 / np.pi


angle = angle(rcvr_phi)


def displayDataCSIEApproach(Dop, angle):
    # Plot data at a number of receivers
    # fig = plt.figure(figsize=(0.39, 0.39), dpi=100)
    plt.plot(angle.T, np.abs(Dop).T, label='Integral-equation method')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.title('scattered wave data in 2D', fontsize=12)
    plt.axis('tight')
    plt.xlabel('observation angle in degrees')
    plt.xlim([0, 360])
    plt.ylabel('abs(data) $\\rightarrow$')
    plt.show()


displayDataCSIEApproach(Dop, angle)

Dop_diff_abs = abs(Dop_diff)
displayDataCSIEApproach(Dop_diff_abs, angle)

# ForwardBiCGSTABFFT.displayDataCompareApproachs(WavefieldSctCircle, Dop, rcvr_phi)

# Iterate over local variables and delete the types that are None to keep workspace tidy
local_variables = list(locals().keys())
for var_name in local_variables:
    var_value = locals()[var_name]
    if var_value is None:
        del locals()[var_name]
        # print(f"Deleted local variable: {var_name}")
