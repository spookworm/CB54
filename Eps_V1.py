""" This file is the script for the Electric Eps VDB code
Time factor = exp(-iwt)
Spatial units is in m
Source wavelet M Z_0 / gamma_0  = 1   (Z_0 M = gamma_0)
"""
from IPython import get_ipython
# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
get_ipython().run_line_magic('reset', '-sf')
try:
    from lib import ForwardBiCGSTABFFT
except ImportError:
    import ForwardBiCGSTABFFT
try:
    from lib import ForwardBiCGSTABFFTwE
except ImportError:
    import ForwardBiCGSTABFFTwE
from lib import graphviz_doc
import numpy as np
import sys
from scipy.special import kv, iv
import time
from scipy.sparse.linalg import bicgstab, LinearOperator
from varname import nameof
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=15)


c_0 = ForwardBiCGSTABFFT.c_0(3e8)
eps_sct = ForwardBiCGSTABFFT.eps_sct(1.75)
mu_sct = ForwardBiCGSTABFFT.mu_sct(1.0)
f = ForwardBiCGSTABFFT.f(10e6)
s = ForwardBiCGSTABFFT.s(f)
wavelength = ForwardBiCGSTABFFT.wavelength(c_0, f)
gamma_0 = ForwardBiCGSTABFFT.gamma_0(s, c_0)
xS = ForwardBiCGSTABFFT.xS()
NR = ForwardBiCGSTABFFT.NR(180)
Errcri = ForwardBiCGSTABFFT.Errcri(1e-13)
rcvr_phi = ForwardBiCGSTABFFT.rcvr_phi(NR)
xR = ForwardBiCGSTABFFT.xR(NR, rcvr_phi)
N1 = ForwardBiCGSTABFFT.N1(120)
N2 = ForwardBiCGSTABFFT.N2(100)
dx = ForwardBiCGSTABFFT.dx(2.0)
a = ForwardBiCGSTABFFT.a(40)
initGrid = ForwardBiCGSTABFFT.initGrid(N1, N2, dx)
X1cap = ForwardBiCGSTABFFT.X1cap(initGrid)
X2cap = ForwardBiCGSTABFFT.X2cap(initGrid)
x1fft = ForwardBiCGSTABFFT.x1fft(N1, dx)
x2fft = ForwardBiCGSTABFFT.x2fft(N2, dx)
initFFTGreen = ForwardBiCGSTABFFT.initFFTGreen(x1fft, x2fft)
X1fftcap = ForwardBiCGSTABFFT.X1fft(initFFTGreen)
X2fftcap = ForwardBiCGSTABFFT.X2fft(initFFTGreen)
delta = ForwardBiCGSTABFFT.delta(dx)
IntG = ForwardBiCGSTABFFT.IntG(dx, gamma_0, X1fftcap, X2fftcap, N1, N2, delta)
FFTG = ForwardBiCGSTABFFT.FFTG(IntG)
R = ForwardBiCGSTABFFT.R(X1cap, X2cap)
CHI_eps = ForwardBiCGSTABFFT.CHI_eps(eps_sct, a, R)
CHI_mu = ForwardBiCGSTABFFT.CHI_mu(mu_sct, a, R)
factoru = ForwardBiCGSTABFFT.factoru(gamma_0, delta)
itmax = ForwardBiCGSTABFFT.itmax(1000)
EZH_inc = ForwardBiCGSTABFFT.EZH_inc(gamma_0, xS, delta, X1cap, X2cap, factoru, N1, N2)
E_inc = ForwardBiCGSTABFFT.E_inc(EZH_inc)
ZH_inc = ForwardBiCGSTABFFT.ZH_inc(EZH_inc)
N = ForwardBiCGSTABFFT.N(CHI_eps)
b_E = ForwardBiCGSTABFFT.b_E(CHI_eps, E_inc, N1, N2, N)
x0 = ForwardBiCGSTABFFT.x0(b_E)


def w(b, CHI_eps, FFTG, N1, N2, Errcri, itmax, gamma_0, dx, N):
    # BiCGSTAB_FFT scheme for contrast source integral equation Aw = b
    norm_b = np.linalg.norm(b)

    def callback(xk):
        # Define the callback function
        # relative residual norm(b-A*x)/norm(b)
        callback.iter_count += 1
        # residual = np.linalg.norm(b - Aw_operator.dot(xk))/norm_b
        # residual = norm_b
        # residual_norm = np.linalg.norm(b - Aw_operator(xk))
        residual_norm = norm_b
        # residuals.append(residual)
        # CHECK: Not sure that this time is correct
        callback.time_total = time.time() - callback.start_time
        # print("Current solution:", xk)
        # print("Iteration:", callback.iter_count, "Residual norm:", residual_norm, "Time:", time.time() - callback.start_time)
        if residual_norm < Errcri:
            return True
        else:
            return False
        print(callback.iter_count, "\t", residual_norm, "\t", callback.time_total)
        print(residual_norm)

    # Initialise iteration count
    callback.iter_count = 0
    callback.start_time = time.time()

    # Call bicgstab with the LinearOperator instance and other inputs
    # w = bicgstab(@(w) Aw(w, input), b, Errcri, itmax);
    Aw_operator = LinearOperator((b.shape[0], b.shape[0]), matvec=lambda w: Aw(w, N1, N2, FFTG, CHI_eps, gamma_0, dx, N))
    start_time = time.time()
    w, exit_code = bicgstab(Aw_operator, b, x0=x0, tol=Errcri, maxiter=itmax, callback=callback)
    time_total = time.time() - start_time
    print("time_total", time_total)

    # Output Matrix
    w = vector2matrix(w, N1, N2, N)

    # # Display the convergence information
    # print("Convergence information:", exit_code)
    # print(exit_code)
    print("Iteration:", callback.iter_count)
    # print("time_total", callback.time_total)
    return w


def Aw(w, N1, N2, FFTG, CHI_eps, gamma_0, dx, N):
    w_E = vector2matrix(w, N1, N2, N)
    Kw_E = KopE(w_E, FFTG, dx, N1, N2, gamma_0)
    y = np.zeros((2*N, 1), dtype=np.complex_, order='F')
    y[0:N, 0] = w_E[0, :, :].flatten('F') - (CHI_eps.flatten('F') * Kw_E[0, :, :].flatten('F'))
    y[N:2*N, 0] = w_E[1, :, :].flatten('F') - (CHI_eps.flatten('F') * Kw_E[1, :, :].flatten('F'))
    return y


def vector2matrix(w, N1, N2, N):
    # Modify vector output from 'bicgstab' to matrices for further computation
    DIM = (N1, N2)
    w_E = np.zeros((2, N1, N2), dtype=np.complex_, order='F')
    w_E[0, :, :] = np.reshape(w[0:N], DIM, order='F')
    w_E[1, :, :] = np.reshape(w[N:2*N], DIM, order='F')
    return w_E


def KopE(wE, FFTG, dx, N1, N2, gamma_0):
    KwE = np.zeros((2, N1, N2), dtype=np.complex_, order='F')
    for n in range(0, 2):
        KwE[n, :, :] = Kop(wE[n, :, :], FFTG)
    # Dummy is temporary storage
    dummy = graddiv(KwE, dx, N1, N2)
    for n in range(0, 2):
        KwE[n, :, :] = KwE[n, :, :] - dummy[n, :, :] / gamma_0**2
    return KwE


def Kop(v, FFTG):
    # Make FFT grid
    Cv = np.zeros(FFTG.shape, dtype=np.complex_, order='F')
    N1, N2 = v.shape
    Cv[0:N1, 0:N2] = v.copy()
    # Convolution by FFT
    Cv = np.fft.fftn(Cv)
    Cv = np.fft.ifftn(FFTG * Cv)
    Kv = Cv[0:N1, 0:N2]
    return Kv


def graddiv(v, dx, N1, N2):
    u = np.zeros((2, N1, N2), dtype=np.complex_, order='F')
    u[0, :, :] = np.zeros(v[1, :, :].shape, dtype=np.complex_, order='F')
    u[1, :, :] = u[0, :, :].copy()
    # % Compute d1d1_v1, d2d2_v2
    # u{1}(2:N1 - 1, :) = v{1}(1:N1 - 2, :) - 2 * v{1}(2:N1 - 1, :) + v{1}(3:N1, :);
    u[0, 1:N1 - 1, :] = v[0, 0:N1 - 2, :] - 2 * v[0, 1:N1 - 1, :] + v[0, 2:N1, :]

    # u{2}(:, 2:N2 - 1) = v{2}(:, 1:N2 - 2) - 2 * v{2}(:, 2:N2 - 1) + v{2}(:, 3:N2);
    u[1, :, 1:N2 - 1] = v[1, :, 0:N2 - 2] - 2 * v[1, :, 1:N2 - 1] + v[1, :, 2:N2]
    # % Replace the input vector v1 by d1_v and v2 by d2_v2
    # v{1}(2:N1 - 1, :) = (v{1}(3:N1, :) - v{1}(1:N1 - 2, :)) / 2; % d1_v1
    v[0, 1:N1 - 1, :] = (v[0, 2:N1, :] - v[0, 0:N1 - 2, :]) / 2
    # v{2}(:, 2:N2 - 1) = (v{2}(:, 3:N2) - v{2}(:, 1:N2 - 2)) / 2; % d2_v2
    v[1, :, 1:N2 - 1] = (v[1, :, 2:N2] - v[1, :, 0:N2 - 2]) / 2

    # % Add d1_v2 = d1d2_v2 to output vector u1
    # u{1}(2:N1 - 1, :) = u{1}(2:N1 - 1, :) + (v{2}(3:N1, :) - v{2}(1:N1 - 2, :)) / 2;
    u[0, 1:N1 - 1, :] = u[0, 1:N1 - 1, :] + (v[1, 2:N1, :] - v[1, 0:N1 - 2, :]) / 2
    # % Add d2_v1 = d2d1_v1 to output vector u2
    # u{2}(:, 2:N2 - 1) = u{2}(:, 2:N2 - 1) + (v{1}(:, 3:N2) - v{1}(:, 1:N2 - 2)) / 2;
    u[1, :, 1:N2 - 1] = u[1, :, 1:N2 - 1] + (v[0, :, 2:N2] - v[0, :, 0:N2 - 2]) / 2

    # % divide by dx^2
    u[0, :, :] = u[0, :, :] / dx**2
    u[1, :, :] = u[1, :, :] / dx**2
    return u


w_E = w(b_E, CHI_eps, FFTG, N1, N2, Errcri, itmax, gamma_0, dx, N)

# w_E_0 = w_E[0]
# w_E_0_m = ForwardBiCGSTABFFT.mat_loader(nameof(w_E_0))
# w_E_0_m = ForwardBiCGSTABFFT.mat_checker(w_E_0, nameof(w_E_0))

w_E_1 = w_E[1]
w_E_1_m = ForwardBiCGSTABFFT.mat_checker(w_E_1, nameof(w_E_1))

np.max(w_E_1) - np.max(w_E_1_m)


def plotContrastSourcewE(w_E, X1, X2):
    # Plot 2D contrast/source distribution
    x1 = X1[:, 0]
    x2 = X2[0, :]

    fig = plt.figure(figsize=(7.09, 4.72))
    fig.subplots_adjust(wspace=0.3)

    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(np.abs(w_E[0]), extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax1.set_xlabel('x$_2$ \u2192')
    ax1.set_ylabel('\u2190 x_1')
    ax1.set_aspect('equal', adjustable='box')
    fig.colorbar(im1, ax=ax1, orientation='horizontal')
    ax1.set_title(r'abs(w$_1^E$)', fontsize=13)

    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(np.abs(w_E[1]), extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax2.set_xlabel('x$_2$ \u2192')
    ax2.set_ylabel('\u2190 x_1')
    ax2.set_aspect('equal', adjustable='box')
    fig.colorbar(im2, ax=ax2, orientation='horizontal')
    ax2.set_title(r'abs(w$_2^E$)', fontsize=13)

    plt.show()


plotContrastSourcewE(w_E, X1cap, X2cap)
