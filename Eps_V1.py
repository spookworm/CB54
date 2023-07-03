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
get_ipython().run_line_magic('reset', '-sf')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=15)


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
arg0 = ForwardBiCGSTABFFT.arg0(gamma_0, a)
gamma_sct = gamma_0 * np.sqrt(eps_sct*mu_sct)
args = ForwardBiCGSTABFFT.args(gamma_sct, a)
M = ForwardBiCGSTABFFT.M(20)
Z_sct = np.sqrt(mu_sct/eps_sct)
angle = ForwardBiCGSTABFFT.angle(rcvr_phi)
EMsctCircle = ForwardBiCGSTABFFT.EMsctCircle(c_0, eps_sct, mu_sct, gamma_0, xR, xS, M, a)
Edata2D = ForwardBiCGSTABFFT.Edata2D(EMsctCircle)
Hdata2D = ForwardBiCGSTABFFT.Hdata2D(EMsctCircle)

Edata2D_m = workspace_func.mat_checker(Edata2D, nameof(Edata2D))
Hdata2D_m = workspace_func.mat_checker(Hdata2D, nameof(Hdata2D))

solveremf2_plot.displayDataBesselApproach(Edata2D, angle)
solveremf2_plot.displayDataBesselApproach(Hdata2D, angle)
# solveremf2_plot.displayEdata(Edata2D, angle)
# solveremf2_plot.displayHdata(Hdata2D, angle)


N1 = ForwardBiCGSTABFFT.N1(120)
N2 = ForwardBiCGSTABFFT.N2(100)
dx = ForwardBiCGSTABFFT.dx(2.0)
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
# E_inc_0 = E_inc[0]
# E_inc_0_m = workspace_func.mat_checker(E_inc_0, nameof(E_inc_0))
# E_inc_1 = E_inc[1]
# E_inc_1_m = workspace_func.mat_checker(E_inc_1, nameof(E_inc_1))

ZH_inc = ForwardBiCGSTABFFT.ZH_inc(EZH_inc)
N = ForwardBiCGSTABFFT.N(CHI_eps)

b_E = ForwardBiCGSTABFFT.b_E(CHI_eps, E_inc, N1, N2, N)
# b_E_m = workspace_func.mat_checker(b_E, nameof(b_E))


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

    def custom_matvec(w):
        # return Aw(w, N1, N2, FFTG, CHI)
        return Aw(w, N1, N2, FFTG, CHI_eps, gamma_0, dx, N)

    # Aw_operator = LinearOperator((b.shape[0], b.shape[0]), matvec=lambda w: Aw(w, N1, N2, FFTG, CHI_eps, gamma_0, dx, N))
    Aw_operator = LinearOperator((b.shape[0], b.shape[0]), matvec=custom_matvec)

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


w_E = w(b_E, CHI_eps, FFTG, N1, N2, Errcri, itmax, gamma_0, dx, N)

w_E_0 = w_E[0]
w_E_0_m = workspace_func.mat_checker(w_E_0, nameof(w_E_0))
# w_E_0_m = workspace_func.mat_loader(nameof(w_E_0))

w_E_1 = w_E[1]
w_E_1_m = workspace_func.mat_checker(w_E_1, nameof(w_E_1))


np.max(w_E_1) - np.max(w_E_1_m)

solveremf2_plot.plotContrastSourcewE(w_E, X1cap, X2cap)


def E_sct(w_E, FFTG, dx, N1, N2, gamma_0):
    return KopE(w_E, FFTG, dx, N1, N2, gamma_0)


E_sct = E_sct(w_E, FFTG, dx, N1, N2, gamma_0)


def E(E_inc, E_sct):
    E = np.zeros((E_inc.shape), dtype=np.complex_, order='F')
    for n in range(0, 2):
        E[n, :, :] = E_inc[n, :, :] + E_sct[n, :, :]
    return E


E = E(E_inc, E_sct)
phi = ForwardBiCGSTABFFT.phi()
solveremf2_plot.plotEtotalwavefield(E, a, X1cap, X2cap, N1, N2, phi)


def DOPwE(w_E, gamma_0, dx, xR, NR, delta, X1, X2):
    Edata = np.zeros((1, NR), dtype=np.complex_, order='F')
    Hdata = np.zeros((1, NR), dtype=np.complex_, order='F')

    # Weak Form
    factor = 2 * iv(1, gamma_0*delta) / (gamma_0*delta)

    for p in range(1, NR+1):
        X1 = xR[0, p-1] - X1
        X2 = xR[1, p-1] - X2

        DIS = np.sqrt(X1**2 + X2**2)
        X1 = X1 / DIS
        X2 = X2 / DIS

        G = factor * 1.0 / (2.0 * np.pi) * kv(0, gamma_0 * DIS)
        dG = -factor * gamma_0 * 1.0 / (2.0 * np.pi) * kv(1, gamma_0 * DIS)
        d1_G = X1 * dG
        d2_G = X2 * dG

        dG11 = (2.0 * X1 * X1 - 1.0) * (-dG / DIS) + gamma_0**2 * X1 * X1 * G
        dG22 = (2.0 * X2 * X2 - 1.0) * (-dG / DIS) + gamma_0**2 * X2 * X2 * G
        dG21 = (2.0 * X2 * X1      ) * (-dG / DIS) + gamma_0**2 * X2 * X1 * G

        # E1rfl = dx^2 * sum((gam0^2 * G(:) - dG11(:)).*w_E{1}(:) - dG21(:).*w_E{2}(:));
        E1rfl = dx**2 * np.sum(gamma_0**2 * G.flatten('F') - dG11.flatten('F') * w_E[0].flatten('F') - dG21.flatten('F') * w_E[1].flatten('F'))
        E2rfl = dx**2 * np.sum(-dG21.flatten('F') * w_E[0].flatten('F') + (gamma_0**2 * G.flatten('F') - dG22.flatten('F')) * w_E[1].flatten('F'))
        ZH3rfl = gamma_0 * dx**2 * np.sum(d2_G.flatten('F') * w_E[0].flatten('F') - d1_G.flatten('F') * w_E[1].flatten('F'))

        Edata[0, p-1] = np.sqrt(np.abs(E1rfl)**2 + np.abs(E2rfl)**2)
        Hdata[0, p-1] = np.abs(ZH3rfl)
    return Edata, Hdata


DOPwE = DOPwE(w_E, gamma_0, dx, xR, NR, delta, X1cap, X2cap)
Edata = ForwardBiCGSTABFFT.Edata2D(DOPwE)
Hdata = ForwardBiCGSTABFFT.ZH_inc(DOPwE)

solveremf2_plot.displayDataCSIEApproach(Edata, angle)
solveremf2_plot.displayDataCSIEApproach(Hdata, angle)

Edata2D = ForwardBiCGSTABFFT.Edata2D(EMsctCircle)
Hdata2D = ForwardBiCGSTABFFT.Hdata2D(EMsctCircle)
solveremf2_plot.displayDataCompareApproachs(Edata2D, Edata, angle)
solveremf2_plot.displayDataCompareApproachs(Hdata2D, Hdata, angle)

time_total_wp = time.time() - start_time_wp


workspace_func.tidy_workspace()
