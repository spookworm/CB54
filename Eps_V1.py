""" This file is the script for the Electric Eps VDB code
Time factor = exp(-iwt)
Spatial units is in m
Source wavelet M Z_0 / gamma_0  = 1   (Z_0 M = gamma_0)
"""
from IPython import get_ipython
try:
    from lib import ForwardBiCGSTABFFT
    from lib import graphviz_doc
    from lib import workspace_func
    from lib import solveremf2_plot
except ImportError:
    import ForwardBiCGSTABFFT
    import graphviz_doc
    import workspace_func
    import solveremf2_plot

import numpy as np
import sys
import time
from varname import nameof

from scipy.special import kv, iv
from scipy.sparse.linalg import bicgstab, LinearOperator

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=18)

start_time_wp = time.time()
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
a = ForwardBiCGSTABFFT.a(40)
N1 = ForwardBiCGSTABFFT.N1(120)
N2 = ForwardBiCGSTABFFT.N2(100)
dx = ForwardBiCGSTABFFT.dx(2.0)
initGrid = ForwardBiCGSTABFFT.initGrid(N1, N2, dx)
X1cap = ForwardBiCGSTABFFT.X1cap(initGrid)
X2cap = ForwardBiCGSTABFFT.X2cap(initGrid)
M = ForwardBiCGSTABFFT.M(100)
x1fft = ForwardBiCGSTABFFT.x1fft(N1, dx)
x2fft = ForwardBiCGSTABFFT.x2fft(N2, dx)
angle = ForwardBiCGSTABFFT.angle(rcvr_phi)

gamma_sct = gamma_0 * np.sqrt(eps_sct*mu_sct)
arg0 = ForwardBiCGSTABFFT.arg0(gamma_0, a)
args = ForwardBiCGSTABFFT.args(gamma_sct, a)

rR = ForwardBiCGSTABFFT.rR(xR)
phiR = ForwardBiCGSTABFFT.phiR(xR)
rS = ForwardBiCGSTABFFT.rS(xS)
phiS = ForwardBiCGSTABFFT.phiS(xS)

Z_sct = np.sqrt(mu_sct/eps_sct)

EMsctCircle = ForwardBiCGSTABFFT.EMsctCircle(c_0, eps_sct, mu_sct, gamma_0, xR, xS, M, a, gamma_sct, Z_sct, arg0, args, rR, phiR, rS, phiS)
Edata2D = ForwardBiCGSTABFFT.Edata2D(EMsctCircle)
# Edata2D_diff = workspace_func.mat_checker(Edata2D, nameof(Edata2D))

Hdata2D = ForwardBiCGSTABFFT.Hdata2D(EMsctCircle)
# Hdata2D_diff = workspace_func.mat_checker(Hdata2D, nameof(Hdata2D))

solveremf2_plot.displayDataBesselApproach(Edata2D, angle)
solveremf2_plot.displayDataBesselApproach(Hdata2D, angle)

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
itmax = ForwardBiCGSTABFFT.itmax(1)

EZH_inc = ForwardBiCGSTABFFT.EZH_inc(gamma_0, xS, delta, X1cap, X2cap, factoru, N1, N2)
E_inc = ForwardBiCGSTABFFT.E_inc(EZH_inc)
E_inc_0 = E_inc[0, :, :]
kE_inc_0_diff = workspace_func.mat_checker(E_inc_0, nameof(E_inc_0))
E_inc_1 = E_inc[1, :, :]
# E_inc_1_diff = workspace_func.mat_checker(E_inc_1, nameof(E_inc_1))


ZH_inc = ForwardBiCGSTABFFT.ZH_inc(EZH_inc)
N = ForwardBiCGSTABFFT.N(CHI_eps)

b_E = ForwardBiCGSTABFFT.b_E(CHI_eps, E_inc, N)
# b_E_diff = workspace_func.mat_checker(b_E, nameof(b_E))
x0 = ForwardBiCGSTABFFT.x0(b_E)


def w(b, CHI_eps, FFTG, N1, N2, Errcri, itmax, x0, gamma_0, dx):
    # BiCGSTAB_FFT scheme for contrast source integral equation Aw = b
    norm_b = np.linalg.norm(b)

    def callback(xk):
        # Define the callback function
        # relative residual norm(b-A*x)/norm(b)
        callback.iter_count += 1
        # residual = np.linalg.norm(b - Aw_operator.dot(xk))/norm_b
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

    def custom_matvec(w):
        # return Aw(w, N1, N2, FFTG, CHI)
        return Aw(w, N1, N2, FFTG, CHI_eps, gamma_0, dx)

    Aw_operator = LinearOperator((b.shape[0], b.shape[0]), matvec=custom_matvec)

    start_time = time.time()
    w, exit_code = bicgstab(Aw_operator, b, x0=x0, tol=Errcri, maxiter=itmax, callback=callback)
    time_total = time.time() - start_time
    print("time_total", time_total)

    # Output Matrix
    w = vector2matrix(w, N1, N2)

    # # Display the convergence information
    # print("Convergence information:", exit_code)
    # print(exit_code)
    print("Iteration:", callback.iter_count)
    # print("time_total", callback.time_total)
    return w


def Aw(w, N1, N2, FFTG, CHI_eps, gamma_0, dx):
    from scipy.io import savemat
    # # w = np.where(w < 1e-300, 0.0, w)
    savemat('w_P.mat', {'w': w})
    # time.sleep(5)
    # sys.exit()
    w_E = vector2matrix(w, N1, N2)

    # # SAVE DATA HERE AT EACH STEP WITH A SAVE DATA FUNCTION
    savemat('w_E_P.mat', {'w_E': w_E})
    # time.sleep(5)
    # sys.exit()

    Kw_E = KopE(w_E, FFTG, dx, N1, N2, gamma_0)

    # # SAVE DATA HERE AT EACH STEP WITH A SAVE DATA FUNCTION
    savemat('Kw_E_P.mat', {'Kw_E': Kw_E})

    y = np.zeros((2*N1*N2, 1), dtype=np.complex_, order='F')
    y[0:N, 0] = w_E[0, :, :].flatten('F') - (CHI_eps.flatten('F') * Kw_E[0, :, :].flatten('F'))
    y[N:2*N, 0] = w_E[1, :, :].flatten('F') - (CHI_eps.flatten('F') * Kw_E[1, :, :].flatten('F'))

    # # SAVE DATA HERE AT EACH STEP WITH A SAVE DATA FUNCTION
    savemat('Kw_E_P.mat', {'Kw_E': Kw_E})

    # ForwardBiCGSTABFFT.data_save('', 'y_P', y)
    return y


def vector2matrix(w, N1, N2):
    # from scipy.io import savemat
    # savemat('w_P.mat', {'w': w})

    # print("w:\t", w.shape)
    # Modify vector output from 'bicgstab' to matrices for further computation
    N = N1 * N2
    DIM = [N1, N2]
    w_E = np.zeros((2, N1, N2), dtype=np.complex_, order='F')
    w_E[0, :, :] = np.reshape(w[0:N], DIM, order='F')
    w_E[1, :, :] = np.reshape(w[N:2*N], DIM, order='F')
    # print("vector2matrix:\t", w_E.shape)
    return w_E


def KopE(w_E, FFTG, dx, N1, N2, gamma_0):
    KwE = np.zeros((2, N1, N2), dtype=np.complex_, order='F')
    for n in range(0, 2):
        KwE[n, :, :] = Kop(w_E[n, :, :], FFTG)
    # Dummy is temporary storage
    dummy = graddiv(KwE, dx, N1, N2)
    # print(dummy.shape)
    for n in range(0, 2):
        KwE[n, :, :] = KwE[n, :, :] - dummy[n, :, :] / gamma_0**2
    return KwE


def Kop(v, FFTG):
    # Make FFT grid
    N1, N2 = v.shape
    Cv = np.zeros(FFTG.shape, dtype=np.complex_, order='F')
    Cv[0:N1, 0:N2] = v.copy()
    # Convolution by FFT
    Cv = np.fft.fftn(Cv)
    Cv = np.fft.ifftn(FFTG * Cv)
    Kv = np.zeros((N1, N2), dtype=np.complex_, order='F')
    Kv[0:N1, 0:N2] = Cv[0:N1, 0:N2]
    return Kv


def graddiv(v, dx, N1, N2):
    # v = np.asfortranarray(v)

    u = np.zeros((v.shape), dtype=np.complex_, order='F')
    # u{1} = zeros(size(v{1}));
    u[0, :, :] = np.zeros(v[0, :, :].shape, dtype=np.complex_, order='F')
    # u{2} = u{1};
    u[1, :, :] = np.zeros(v[1, :, :].shape, dtype=np.complex_, order='F')

    # % Compute d1d1_v1, d2d2_v2
    # u{1}(2:N1 - 1, :) = v{1}(1:N1 - 2, :) - 2 * v{1}(2:N1 - 1, :) + v{1}(3:N1, :);
    u[0, 1:N1 - 1, :] = v[0, 0:N1 - 2, :] - 2 * v[0, 1:N1 - 1, :] + v[0, 2:N1, :]

    # u{2}(:, 2:N2 - 1) = v{2}(:, 1:N2 - 2) - 2 * v{2}(:, 2:N2 - 1) + v{2}(:, 3:N2);
    u[1, :, 1:N2 - 1] = v[1, :, 0:N2 - 2] - 2 * v[1, :, 1:N2 - 1] + v[1, :, 2:N2]

    # % Replace the input vector v1 by d1_v and v2 by d2_v2 ---------------------
    # v{1}(2:N1 - 1, :) = (v{1}(3:N1, :) - v{1}(1:N1 - 2, :)) / 2; % d1_v1
    v[0, 1:N1 - 1, :] = (v[0, 2:N1, :] - v[0, 0:N1 - 2, :]) / 2.0

    # v{2}(:, 2:N2 - 1) = (v{2}(:, 3:N2) - v{2}(:, 1:N2 - 2)) / 2; % d2_v2
    v[1, :, 1:N2 - 1] = (v[1, :, 2:N2] - v[1, :, 0:N2 - 2]) / 2.0

    # % Add d1_v2 = d1d2_v2 to output vector u1 ---------------------------------
    # u{1}(2:N1 - 1, :) = u{1}(2:N1 - 1, :) + (v{2}(3:N1, :) - v{2}(1:N1 - 2, :)) / 2;
    u[0, 1:N1 - 1, :] = u[0, 1:N1 - 1, :] + (v[1, 2:N1, :] - v[1, 0:N1 - 2, :]) / 2.0

    # % Add d2_v1 = d2d1_v1 to output vector u2 ---------------------------------
    # u{2}(:, 2:N2 - 1) = u{2}(:, 2:N2 - 1) + (v{1}(:, 3:N2) - v{1}(:, 1:N2 - 2)) / 2;
    u[1, :, 1:N2 - 1] = u[1, :, 1:N2 - 1] + (v[0, :, 2:N2] - v[0, :, 0:N2 - 2]) / 2.0

    # % divide by dx^2
    u[0, :, :] = u[0, :, :] / dx**2
    u[1, :, :] = u[1, :, :] / dx**2
    return u


w_E = w(b_E, CHI_eps, FFTG, N1, N2, Errcri, itmax, x0, gamma_0, dx)


from scipy.io import loadmat
var_name_pyt = loadmat('w_P.mat')['w']
var_name_mat = loadmat('w_mat.mat')['w']
var_diff = var_name_pyt.T - var_name_mat
workspace_func.plotDiff(var_diff, X1cap, X2cap)


from scipy.io import loadmat
var_name_pyt_0 = loadmat('w_E_P.mat')['w_E'][0]
var_name_mat_0 = loadmat('w_E_mat.mat')['w_E'][0, 0]
var_diff_0 = var_name_pyt_0 - var_name_mat_0
workspace_func.plotDiff(var_diff_0, X1cap, X2cap)

var_name_pyt_1 = loadmat('w_E_P.mat')['w_E'][1]
var_name_mat_1 = loadmat('w_E_mat.mat')['w_E'][0, 1]
var_diff_1 = var_name_pyt_1 - var_name_mat_1
workspace_func.plotDiff(var_diff_1, X1cap, X2cap)


# var_name_mat = np.zeros(var_name_pyt.shape, dtype=np.complex_)
# var_name_mat[0, :, :] = var_name_mat_0
# var_name_mat[1, :, :] = var_name_mat_1
# del var_name_mat_0, var_name_mat_1

# var_diff = np.zeros(var_name_mat.shape, dtype=np.complex_)
# var_diff[0, :, :] = var_name_pyt[0, :, :] - var_name_mat[0, :, :]
# var_diff[1, :, :] = var_name_pyt[1, :, :] - var_name_mat[1, :, :]

var_diff = var_name_pyt.T - var_name_mat

workspace_func.plotDiff(var_diff, X1cap, X2cap)
# workspace_func.plotDiff(var_diff[0, :, :], X1cap, X2cap)
# workspace_func.plotDiff(var_diff[1, :, :], X1cap, X2cap)


# from scipy.io import loadmat
# var_name_pyt = loadmat('w_P.mat')['w']
# var_name_mat = loadmat('w_mat.mat')['w']

# var_diff = var_name_pyt - var_name_mat.T
# workspace_func.plotDiff(var_diff, X1cap, X2cap)


# from scipy.io import loadmat
# from scipy.io import savemat
# savemat('w_E_P.mat', {'w_E': w_E})

# var_name_pyt = loadmat('w_E_P.mat')['w_E']
# var_name_mat_0 = loadmat('w_E_mat.mat')['w_E'][0, 0]
# var_name_mat_1 = loadmat('w_E_mat.mat')['w_E'][0, 1]
# var_name_mat = np.zeros(var_name_pyt.shape, dtype=np.complex_)
# var_name_mat[0, :, :] = var_name_mat_0
# var_name_mat[1, :, :] = var_name_mat_1
# del var_name_mat_0, var_name_mat_1

# var_diff = np.zeros(var_name_mat.shape, dtype=np.complex_)
# var_diff[0, :, :] = var_name_pyt[0, :, :] - var_name_mat[0, :, :]
# var_diff[1, :, :] = var_name_pyt[1, :, :] - var_name_mat[1, :, :]

# workspace_func.plotDiff(var_diff[0, :, :], X1cap, X2cap)
# workspace_func.plotDiff(var_diff[1, :, :], X1cap, X2cap)



## BELOW HERE IS NON-CHECKER CODE
# solveremf2_plot.plotContrastSourcewE(w_E, X1cap, X2cap)

# def E_sct(w_E, FFTG, dx, N1, N2, gamma_0):
#     return KopE(w_E, FFTG, dx, N1, N2, gamma_0)


# E_sct = E_sct(w_E, FFTG, dx, N1, N2, gamma_0)


# def E(E_inc, E_sct):
#     E = np.zeros((E_inc.shape), dtype=np.complex_, order='F')
#     for n in range(0, 2):
#         E[n, :, :] = E_inc[n, :, :] + E_sct[n, :, :]
#     return E


# E = E(E_inc, E_sct)
# phi = ForwardBiCGSTABFFT.phi()
# solveremf2_plot.plotEtotalwavefield(E, a, X1cap, X2cap, N1, N2, phi)
# DOPwE = ForwardBiCGSTABFFT.DOPwE(w_E, gamma_0, dx, xR, NR, X1cap, X2cap, delta, factoru)
# Edata = ForwardBiCGSTABFFT.Edata2D(DOPwE)
# Hdata = ForwardBiCGSTABFFT.ZH_inc(DOPwE)

# solveremf2_plot.displayDataCSIEApproach(Edata, angle)
# solveremf2_plot.displayDataCSIEApproach(Hdata, angle)

# Edata2D = ForwardBiCGSTABFFT.Edata2D(EMsctCircle)
# Hdata2D = ForwardBiCGSTABFFT.Hdata2D(EMsctCircle)



# solveremf2_plot.displayDataCompareApproachs(Edata2D, Edata, angle)
# solveremf2_plot.displayDataCompareApproachs(Hdata2D, Hdata, angle)
# time_total_wp = time.time() - start_time_wp
# workspace_func.tidy_workspace()
