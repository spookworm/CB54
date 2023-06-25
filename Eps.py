""" This file is the script for the Electric Eps VDB code
Time factor = exp(-iwt)
Spatial units is in m
Source wavelet M Z_0 / gamma_0  = 1   (Z_0 M = gamma_0)
"""

try:
    from lib import ForwardBiCGSTABFFT
except ImportError:
    import ForwardBiCGSTABFFT
try:
    from lib import ForwardBiCGSTABFFTwE
except ImportError:
    import ForwardBiCGSTABFFTwE
from lib import graphviz_doc

from scipy.sparse.linalg import bicgstab, LinearOperator
import numpy as np
import time
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=15)


# def composer_call():
#     from fn_graph import Composer
#     composer_1 = (
#         Composer()
#         .update(
#             # list of custom functions goes here
#             ForwardBiCGSTABFFT.Errcri,
#             ForwardBiCGSTABFFT.FFTG,
#             ForwardBiCGSTABFFT.IntG,
#             ForwardBiCGSTABFFT.N1,
#             ForwardBiCGSTABFFT.N2,
#             ForwardBiCGSTABFFT.NR,
#             ForwardBiCGSTABFFT.R,
#             ForwardBiCGSTABFFT.X1,
#             ForwardBiCGSTABFFT.X1fft,
#             ForwardBiCGSTABFFT.X2,
#             ForwardBiCGSTABFFT.X2fft,
#             ForwardBiCGSTABFFT.a,
#             ForwardBiCGSTABFFT.dx,
#             ForwardBiCGSTABFFT.gamma_0,
#             ForwardBiCGSTABFFT.initFFTGreen,
#             ForwardBiCGSTABFFT.initFFTGreen1,
#             ForwardBiCGSTABFFT.initFFTGreen2,
#             ForwardBiCGSTABFFT.initGrid,
#             ForwardBiCGSTABFFT.itmax,
#             ForwardBiCGSTABFFT.rcvr_phi,
#             ForwardBiCGSTABFFT.s,
#             ForwardBiCGSTABFFT.wavelength,
#             ForwardBiCGSTABFFT.xR,
#             ForwardBiCGSTABFFT.xS,
#             ForwardBiCGSTABFFTwE.CHI_eps,
#             ForwardBiCGSTABFFTwE.CHI_mu,
#             ForwardBiCGSTABFFTwE.EMsctCircle,
#             ForwardBiCGSTABFFTwE.E_inc,
#             ForwardBiCGSTABFFTwE.Edata2D,
#             ForwardBiCGSTABFFTwE.Hdata2D,
#             # ForwardBiCGSTABFFTwE.ITERBiCGSTABwE,
#             ForwardBiCGSTABFFTwE.IncEMwave,
#             ForwardBiCGSTABFFTwE.M,
#             ForwardBiCGSTABFFTwE.ZH_inc,
#             ForwardBiCGSTABFFTwE.b,
#             ForwardBiCGSTABFFTwE.c_0,
#             ForwardBiCGSTABFFTwE.eps_sct,
#             ForwardBiCGSTABFFTwE.f,
#             ForwardBiCGSTABFFTwE.mu_sct,
#             ForwardBiCGSTABFFTwE.plotEMcontrast,
#         )
#         # .update_parameters(input_length_side=input_length_x_side)
#         # .cache()
#     )
#     return composer_1


# composer = graphviz_doc.composer_render(composer_call(), '', "digraph")


c_0 = ForwardBiCGSTABFFTwE.c_0()
eps_sct = ForwardBiCGSTABFFTwE.eps_sct()
mu_sct = ForwardBiCGSTABFFTwE.mu_sct()
f = ForwardBiCGSTABFFTwE.f()
wavelength = ForwardBiCGSTABFFT.wavelength(c_0, f)
s = ForwardBiCGSTABFFT.s(f)
gamma_0 = ForwardBiCGSTABFFT.gamma_0(s, c_0)

xS = ForwardBiCGSTABFFT.xS()
NR = ForwardBiCGSTABFFT.NR()
rcvr_phi = ForwardBiCGSTABFFT.rcvr_phi(NR)
xR = ForwardBiCGSTABFFT.xR(NR, rcvr_phi)

N1 = ForwardBiCGSTABFFT.N1()
N2 = ForwardBiCGSTABFFT.N2()
dx = ForwardBiCGSTABFFT.dx()

initGrid = ForwardBiCGSTABFFT.initGrid(N1, N2, dx)
X1 = ForwardBiCGSTABFFT.X1(initGrid)
X2 = ForwardBiCGSTABFFT.X2(initGrid)

initFFTGreen1 = ForwardBiCGSTABFFT.initFFTGreen1(N1, dx)
initFFTGreen2 = ForwardBiCGSTABFFT.initFFTGreen2(N2, dx)
initFFTGreen = ForwardBiCGSTABFFT.initFFTGreen(initFFTGreen1, initFFTGreen2)
X1fft = ForwardBiCGSTABFFT.X1fft(initFFTGreen)
X2fft = ForwardBiCGSTABFFT.X2fft(initFFTGreen)

IntG = ForwardBiCGSTABFFT.IntG(dx, gamma_0, X1fft, X2fft)
FFTG = ForwardBiCGSTABFFT.FFTG(IntG)
a = ForwardBiCGSTABFFT.a()
R = ForwardBiCGSTABFFT.R(X1, X2)

CHI_eps = ForwardBiCGSTABFFTwE.CHI_eps(eps_sct, a, R)
CHI_mu = ForwardBiCGSTABFFTwE.CHI_mu(mu_sct, a, R)

Errcri = ForwardBiCGSTABFFT.Errcri()
M = ForwardBiCGSTABFFTwE.M()

EMsctCircle = ForwardBiCGSTABFFTwE.EMsctCircle(c_0, eps_sct, mu_sct, gamma_0, xR, xS, M, a)
Edata2D = ForwardBiCGSTABFFTwE.Edata2D(EMsctCircle)
Hdata2D = ForwardBiCGSTABFFTwE.Hdata2D(EMsctCircle)

ForwardBiCGSTABFFTwE.displayEdata(Edata2D, rcvr_phi)
ForwardBiCGSTABFFTwE.displayHdata(Hdata2D, rcvr_phi)
plotEMcontrast = ForwardBiCGSTABFFTwE.plotEMcontrast(X1, X2, CHI_eps, CHI_mu)

IncEMwave = ForwardBiCGSTABFFTwE.IncEMwave(gamma_0, xS, dx, X1, X2)
E_inc = ForwardBiCGSTABFFTwE.E_inc(IncEMwave)
ZH_inc = ForwardBiCGSTABFFTwE.ZH_inc(IncEMwave)

itmax = ForwardBiCGSTABFFT.itmax()


def b(CHI_eps, E_inc):
    # Known 1D vector right-hand side
    # [N, ~] = size(CHI_eps(:));
    [N, ] = CHI_eps.flatten().shape

    # print("type(E_inc[1])", type(E_inc[1]))
    # print("(E_inc[1]).shape", (E_inc[1]).shape)
    # print("E_inc[1][0]", E_inc[1][0])
    b = np.zeros(2*N, dtype=np.complex_)
    # # b(1:N, 1) = CHI_eps(:) .* E_inc{1}(:);
    # b[0:N] = CHI_eps.flatten() * E_inc[1].ravel()
    # # b(N+1:2*N, 1) = CHI_eps(:) .* E_inc{2}(:);
    # b[N:0] = CHI_eps.flatten() * E_inc[2].ravel()

    b1 = CHI_eps[:].T.flatten() * E_inc[1][:].flatten()
    # b[0:N] = CHI_eps.flatten() * list(E_inc.values())[0].flatten()
    b2 = CHI_eps[:].T.flatten() * E_inc[2][:].flatten()
    # b[N:2*N] = CHI_eps.flatten() * list(E_inc.values())[1].flatten()
    b = np.concatenate((b1, b2))

    b_real = np.real(b)
    b_imag = np.imag(b)
    b_abs = abs(b)
    # max(abs(b))
    # min(abs(b))

    asfbdfb = CHI_eps[:].T.flatten()

    asdfafdsgadfgafdgafd = np.real(E_inc[1])

    return b


b = b(CHI_eps, E_inc)


def vector2matrix(w, N1, N2, CHI_eps):
    # Modify vector output from 'bicgstab' to matrix for further computations
    # [N, ~] = size(CHI_eps(:));
    N = np.shape(CHI_eps.flatten())[0]

    print("w.shape", w.shape)
    # w_E = np.zeros((N, 2), dtype=np.complex_)
    # w_E = []
    w_E = [[] for i in range(2)]
    # print("w_E.shape", w_E.shape)

    DIM = [N1, N2]
    # w_E{1} = reshape(w(1:N, 1), DIM);
    w_E[0] = np.reshape(w[0:N], DIM)

    print("w.shape", w.shape)
    print("np.reshape(w[N:2*N], DIM)", np.reshape(w[N:2*N], DIM))

    w_E[1] = np.reshape(w[N:2*N], DIM)
    return w_E


def Aw(w, N1, N2, FFTG, CHI_eps, gamma_0):
    # [N, ~] = size(input.CHI_eps(:));
    [N, ] = CHI_eps.flatten().shape

    # [w_E] = vector2matrix(w, input);
    w_E = vector2matrix(w, N1, N2, CHI_eps)

    # [Kw_E] = KopE(w_E, input);
    Kw_E = KopE(w_E, FFTG, gamma_0)

    y = []
    # y(1:N, 1) = w_E{1}(:) - CHI_eps(:) .* Kw_E{1}(:)
    y[0:N, 0] = w_E[0].flatten() - CHI_eps.flatten() * Kw_E[0].flatten()
    # y(N+1:2*N, 1) = w_E{2}(:) - CHI_eps(:) .* Kw_E{2}(:)
    y[N:2*N, 0] = w_E[1].flatten() - CHI_eps.flatten() * Kw_E[1].flatten()

    return y


def KopE(w_E, FFTG, gamma_0):
    KwE = []
    for n in range(1, 3):
        # KwE{n} = Kop(wE{n}, input.FFTG);
        KwE[n] = Kop(w_E[n], FFTG)
    # Dummy is temporary storage
    # dummy = graddiv(KwE, input);
    dummy = graddiv(KwE, input)
    for n in range(1, 3):
        # KwE{n} = KwE{n} - dummy{n} / input.gamma_0^2;
        KwE[n] = KwE[n] - dummy[n] / gamma_0**2
    return KwE


def Kop(v, FFTG):
    # Make FFT grid
    Cv = np.zeros(FFTG.shape, dtype=np.complex_)
    N1, N2 = v.shape
    Cv[0:N1, 0:N2] = v
    # Convolution by FFT
    Cv = np.fft.fftn(Cv)
    Cv = np.fft.ifftn(FFTG * Cv)
    Kv = Cv[0:N1, 0:N2]
    return Kv


def graddiv(v, dx, N1, N2):
    u = []
    # u{1} = zeros(size(v{1}));
    u[0] = np.zeros(v[0].shape)
    u[1] = u[0].copy

    # Compute d1d1_v1, d2d2_v2
    # u{1}(2:N1 - 1, :) = v{1}(1:N1 - 2, :) - 2 * v{1}(2:N1 - 1, :) + v{1}(3:N1, :);
    u[0][1:N1-1, :] = v[0][0:N1-2, :] - 2 * v[0][1:N1-1, :] + v[0][2:N1, :]

    # u{2}(:, 2:N2 - 1) = v{2}(:, 1:N2 - 2) - 2 * v{2}(:, 2:N2 - 1) + v{2}(:, 3:N2);
    u[1][:, 1:N2-2] = v[1][:, 0:N2-3] - 2 * v[1][:, 1:N2-2] + v[1][:, 2:N2-1]

    # Replace the input vector v1 by d1_v and v2 by d2_v2
    # d1_v1
    # v{1}(2:N1 - 1, :) = (v{1}(3:N1, :) - v{1}(1:N1 - 2, :)) / 2;
    v[0][1:N1-2, :] = (v[0][2:N1, :] - v[0][0:N1-3, :]) / 2
    # d2_v2
    # v{2}(:, 2:N2 - 1) = (v{2}(:, 3:N2) - v{2}(:, 1:N2 - 2)) / 2;
    v[1][:, 1:N2-2] = (v[1][:, 2:N2-1] - v[1][:, 0:N2-3]) / 2

    # Add d1_v2 = d1d2_v2 to output vector u1
    # u{1}(2:N1 - 1, :) = u{1}(2:N1 - 1, :) + (v{2}(3:N1, :) - v{2}(1:N1 - 2, :)) / 2;
    u[0][1:N1-2, :] = u[0][1:N1-2, :] + (v[1][2:N1, :] - v[1][0:N1-3, :]) / 2.0

    # Add d2_v1 = d2d1_v1 to output vector u2
    # u{2}(:, 2:N2 - 1) = u{2}(:, 2:N2 - 1) + (v{1}(:, 3:N2) - v{1}(:, 1:N2 - 2)) / 2;
    u[1][:, 1:N2-2] = u[1][:, 1:N2-2] + (v[0][:, 2:N2] - v[0][:, 0:N2-3]) / 2.0

    # divide by dx^2
    u[0] = u[0] / dx**2
    u[1] = u[1] / dx**2
    return u


def ITERBiCGSTABwE(b, CHI_eps, E_inc, ZH_inc, FFTG, N1, N2, Errcri, itmax, gamma_0):
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
    Aw_operator = LinearOperator((N1*N2, N1*N2), matvec=lambda w: Aw(w, N1, N2, FFTG, CHI_eps, gamma_0))
    w, exit_code = bicgstab(Aw_operator, b.flatten(), tol=Errcri, maxiter=itmax, callback=callback)

    # Output Matrix
    w_E = vector2matrix(w, N1, N2, CHI_eps)

    # Display the convergence information
    print("Convergence information:", exit_code)
    print(exit_code)
    print("Iteration:", callback.iter_count)
    print("time_total", callback.time_total)
    return w_E


ITERBiCGSTABwE = ITERBiCGSTABwE(b, CHI_eps, E_inc, ZH_inc, FFTG, N1, N2, Errcri, itmax, gamma_0)
