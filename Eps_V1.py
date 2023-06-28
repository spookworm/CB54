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

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=15)


c_0 = 3e8
eps_sct = 1.75
mu_sct = 1
f = 10e6
wavelength = c_0 / f
s = 1e-16 - 1j * 2 * np.pi * f
gamma_0 = s / c_0

xS = np.zeros((2), dtype=float)
xS[0] = -170
xS[1] = 0

NR = 180

# input.rcvr_phi(1:input.NR) = (1:input.NR) * 2 * pi / input.NR;


rcvr_phi = np.zeros(NR, dtype=float)
rcvr_phi[0:NR] = np.arange(1, NR+1, 1) * 2 * np.pi / NR

xR = np.zeros((2, NR), dtype=float)
xR[0, 0:NR] = 150 * np.cos(rcvr_phi)
xR[1, 0:NR] = 150 * np.sin(rcvr_phi)

N1 = 120
N2 = 100
dx = 2
x1 = -(N1 + 1) * dx / 2 + np.arange(1, N1+1, 1) * dx
x2 = -(N2 + 1) * dx / 2 + np.arange(1, N2+1, 1) * dx

[X1, X2] = np.meshgrid(x1, x2, indexing='ij')

N1fft = int(2**np.ceil(np.log2(2*N1)))
N2fft = int(2**np.ceil(np.log2(2*N2)))

# x1(1:N1fft) = [0 : N1fft/2 - 1, N1fft/2 : -1 : 1] * dx;
x11 = np.arange(0, N1fft / 2, 1) * dx
x12 = np.arange(N1fft / 2, 0, -1) * dx
x1 = np.concatenate((x11, x12))

# x2(1:N2fft) = [0 : N2fft/2 - 1, N2fft/2 : -1 : 1] * dx;
x21 = np.arange(0, N2fft / 2, 1) * dx
x22 = np.arange(N2fft / 2, 0, -1) * dx
x2 = np.concatenate((x21, x22))

del x11, x12, x21, x22

# [temp.X1fft, temp.X2fft] = ndgrid(x1, x2);
[X1fft, X2fft] = np.meshgrid(x1, x2, indexing='ij')


def Green(dx, gamma_0, X1fft, X2fft):
    DIS = np.sqrt(X1fft**2 + X2fft**2)
    # avoid Green's singularity for DIS = 0
    DIS[0, 0] = 1
    G = 1 / (2 * np.pi) * kv(0, gamma_0 * DIS)
    # radius circle with area of dx^2
    delta = (np.pi)**(-0.5) * dx
    factor = 2 * iv(1, gamma_0 * delta) / (gamma_0 * delta)
    # integral includes gam0^2
    IntG = (gamma_0**2 * dx**2) * factor**2 * G
    IntG[0, 0] = 1 - gamma_0 * delta * kv(1, gamma_0 * delta) * factor
    return IntG


IntG = Green(dx, gamma_0, X1fft, X2fft)


def FFTG(IntG):
    # Apply n-dimensional Fast Fourier transform
    # input.FFTG = fftn(IntG);
    return np.fft.fftn(IntG)


FFTG = FFTG(IntG)

# radius circle cylinder / radius sphere
a = 40

# def initEMcontrast()
R = np.sqrt(X1**2 + X2**2)

# (1) Compute permittivity contrast
CHI_eps = (1 - eps_sct) * (R < a)

# (2) Compute permeability contrast
CHI_mu = (1 - mu_sct) * (R < a)

Errcri = 1e-3

# def IncEMwave():
# incident wave from electric dipole in negative x_1
# radius circle with area of dx^2
delta = (np.pi)**(-0.5) * dx
factor = 2 * iv(1, gamma_0*delta) / (gamma_0*delta)

X1 = X1.copy() - xS[0]
X2 = X2.copy() - xS[1]
# # DIS = np.sqrt(X1**2 + X2**2)
DIS = np.sqrt(np.power(X1, 2) + np.power(X2, 2))
# DIS = np.linalg.norm(np.array([X1, X2]), axis=0)
X1 = X1.copy() / DIS
X2 = X2.copy() / DIS

G = factor * 1 / (2 * np.pi) * kv(0, gamma_0*DIS)
dG = -factor * gamma_0 * 1 / (2 * np.pi) * kv(1, gamma_0*DIS)
dG11 = (2 * X1 * X1 - 1) * (-dG / DIS) + gamma_0**2 * X1 * X1 * G
dG21 = 2 * X2 * X1 * (-dG / DIS) + gamma_0**2 * X2 * X1 * G

E_inc = np.zeros((3, N1, N2), dtype=complex)
E_inc[0, :, :] = -(-gamma_0**2 * G + dG11)
E_inc[1, :, :] = -dG21
E_inc[2, :, :] = np.zeros((1, N1, N2), dtype=complex)

ZH_inc = np.zeros((3, N1, N2), dtype=complex)
ZH_inc[0, :, :] = np.zeros((1, N1, N2), dtype=complex)
ZH_inc[1, :, :] = np.zeros((1, N1, N2), dtype=complex)
ZH_inc[2, :, :] = gamma_0 * X2 * dG

# BiCGSTAB scheme for contrast source integral equation Aw = b
itmax = 1000
N = CHI_eps.flatten('F').size



# b = np.zeros((2*N), dtype=complex)
# # b(1:N, 1) = input.CHI_eps(:) .* E_inc{1}(:);
# b[0:N] = np.multiply(CHI_eps.flatten(), E_inc[0, :, :].flatten('F'))
# # b(N+1:2*N, 1) = input.CHI_eps(:) .* E_inc{2}(:);
# b[N:2*N] = np.multiply(CHI_eps.flatten(), E_inc[1, :, :].flatten('F'))



import scipy.io
b_matlab = scipy.io.loadmat('b.mat', squeeze_me=True)
b = np.array(b_matlab['b'])

print("b", b[4130])
print("b", b[4142] - (3.944180228434370e-05 - 1.582151986875243e-05j))  # same as 4144 in excel and 4143 in matlab
print("b", b[4143])
b_real = np.real(b)
b_imag = np.imag(b)


def w(b, CHI_eps, FFTG, N1, N2, Errcri, itmax, gamma_0, dx, N):
    # BiCGSTAB_FFT scheme for contrast source integral equation Aw = b

    def callback(xk):
        # Define the callback function
        callback.iter_count += 1
        residual_norm = np.linalg.norm(b - Aw_operator(xk))
        # CHECK: Not sure that this time is correct
        callback.time_total = time.time() - callback.start_time
        # print("Current solution:", xk)
        # print("Iteration:", callback.iter_count, "Residual norm:", residual_norm, "Time:", time.time() - callback.start_time)
        print(callback.iter_count, "\t", residual_norm, "\t", callback.time_total)
        if residual_norm < Errcri:
            return True
        else:
            return False
        print("Iteration:", "\t", "Residual norm:", "\t", "Time:")

    # Initialise iteration count
    callback.iter_count = 0
    callback.start_time = time.time()

    # Call bicgstab with the LinearOperator instance and other inputs
    # w = bicgstab(@(w) Aw(w, input), b, Errcri, itmax);
    Aw_operator = LinearOperator((b.shape[0], b.shape[0]), matvec=lambda w: Aw(w, N1, N2, FFTG, CHI_eps, gamma_0, dx, N))
    w, exit_code = bicgstab(Aw_operator, b, tol=Errcri, maxiter=itmax, callback=callback)

    # Display the convergence information
    print("Convergence information:", exit_code)
    print(exit_code)
    print("Iteration:", callback.iter_count)
    print("time_total", callback.time_total)
    return w


def Aw(w, N1, N2, FFTG, CHI_eps, gamma_0, dx, N):
    w_E = vector2matrix(w, N, N1, N2)
    Kw_E = KopE(w_E, FFTG, dx, N1, N2, gamma_0)
    y = np.zeros((2*N), dtype=complex)
    y[0:N] = w_E[0, :, :].flatten() - (CHI_eps.flatten('F') * Kw_E[0, :, :].flatten('F'))
    y[N:2*N] = w_E[1, :, :].flatten() - (CHI_eps.flatten('F') * Kw_E[1, :, :].flatten('F'))
    return y


def vector2matrix(w, N, N1, N2):
    # Modify vector output from 'bicgstab' to matrices for further computation
    DIM = (N1, N2)
    w_E = np.zeros((2, N1, N2), dtype=complex)
    w_E[0, :, :] = np.reshape(w[0:N], DIM)
    w_E[1, :, :] = np.reshape(w[N:2*N], DIM)
    return w_E


def KopE(wE, FFTG, dx, N1, N2, gamma_0):
    KwE = np.zeros((2, N1, N2), dtype=complex)
    for n in range(0, 2):
        KwE[n, :, :] = Kop(wE[n, :, :], FFTG)
    # Dummy is temporary storage
    dummy = graddiv(KwE, dx, N1, N2)
    for n in range(0, 2):
        KwE[n, :, :] = KwE[n, :, :] - dummy[n, :, :] / gamma_0**2
    return KwE


def Kop(v, FFTG):
    # import scipy.fftpack
    # Make FFT grid
    Cv = np.zeros(FFTG.shape, dtype=np.complex_)
    N1, N2 = v.shape
    Cv[0:N1, 0:N2] = v.copy()
    # Convolution by FFT
    Cv = np.fft.fftn(Cv)
    # Cv = scipy.fftpack.fftn(Cv)
    Cv = np.fft.ifftn(FFTG * Cv)
    # Cv = scipy.fftpack.ifftn(FFTG * Cv)
    Kv = Cv[0:N1, 0:N2]
    return Kv


def graddiv(v, dx, N1, N2):
    u = np.zeros((2, N1, N2), dtype=complex)
    u[0, :, :] = np.zeros(v[1, :, :].shape, dtype=np.complex_)
    u[1, :, :] = u[0, :, :].copy()
    # % Compute d1d1_v1, d2d2_v2
    # u{1}(2:N1 - 1, :) = v{1}(1:N1 - 2, :) - 2 * v{1}(2:N1 - 1, :) + v{1}(3:N1, :);
    u[0, 1:N1-1, :] = v[0, 0:N1-2, :] - 2 * v[0, 1:N1-1, :] + v[0, 2:N1, :]

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


w = w(b, CHI_eps, FFTG, N1, N2, Errcri, itmax, gamma_0, dx, N)

w_E = ForwardBiCGSTABFFTwE.w_E(w, N1, N2, N)

ForwardBiCGSTABFFTwE.plotContrastSourcewE(w_E, X1, X2)
