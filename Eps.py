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

from scipy.sparse.linalg import bicgstab, LinearOperator
import numpy as np
import time
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=15)


def composer_call():
    from fn_graph import Composer
    composer_1 = (
        Composer()
        .update(
            # list of custom functions goes here
            ForwardBiCGSTABFFT.Errcri,
            ForwardBiCGSTABFFT.FFTG,
            ForwardBiCGSTABFFT.IntG,
            ForwardBiCGSTABFFT.N1,
            ForwardBiCGSTABFFT.N2,
            ForwardBiCGSTABFFT.NR,
            ForwardBiCGSTABFFT.R,
            ForwardBiCGSTABFFT.X1,
            ForwardBiCGSTABFFT.X1fft,
            ForwardBiCGSTABFFT.X2,
            ForwardBiCGSTABFFT.X2fft,
            ForwardBiCGSTABFFT.a,
            ForwardBiCGSTABFFT.dx,
            ForwardBiCGSTABFFT.gamma_0,
            ForwardBiCGSTABFFT.initFFTGreen,
            ForwardBiCGSTABFFT.initFFTGreen1,
            ForwardBiCGSTABFFT.initFFTGreen2,
            ForwardBiCGSTABFFT.initGrid,
            ForwardBiCGSTABFFT.itmax,
            ForwardBiCGSTABFFT.rcvr_phi,
            ForwardBiCGSTABFFT.s,
            ForwardBiCGSTABFFT.wavelength,
            ForwardBiCGSTABFFT.xR,
            ForwardBiCGSTABFFT.xS,
            ForwardBiCGSTABFFTwE.CHI_eps,
            ForwardBiCGSTABFFTwE.CHI_mu,
            ForwardBiCGSTABFFTwE.EMsctCircle,
            ForwardBiCGSTABFFTwE.E_inc,
            ForwardBiCGSTABFFTwE.Edata2D,
            ForwardBiCGSTABFFTwE.Hdata2D,
            ForwardBiCGSTABFFTwE.ITERBiCGSTABwE,
            ForwardBiCGSTABFFTwE.IncEMwave,
            ForwardBiCGSTABFFTwE.M,
            ForwardBiCGSTABFFTwE.ZH_inc,
            ForwardBiCGSTABFFTwE.b,
            ForwardBiCGSTABFFTwE.c_0,
            ForwardBiCGSTABFFTwE.eps_sct,
            ForwardBiCGSTABFFTwE.f,
            ForwardBiCGSTABFFTwE.mu_sct,
            ForwardBiCGSTABFFTwE.plotEMcontrast,
        )
        # .update_parameters(input_length_side=input_length_x_side)
        # .cache()
    )
    return composer_1


composer = graphviz_doc.composer_render(composer_call(), '', "digraph")


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

b = ForwardBiCGSTABFFTwE.b(CHI_eps, E_inc)
# print(np.real(b)[19871] - 9.533884691144613e-05)
# print(np.real(b)[19872] - 3.541293300961272e-05)

# print(sum(np.real(b)) - (- 0.002404557207356))
# print(sum(np.imag(b)) - (-7.340318325805991e-04))

# print(np.sum(np.real(E_inc[1])) - (-0.019832576190409))
# print(np.sum(np.real(E_inc[2])) - (-1.301042606982605e-18))

ITERBiCGSTABwE = ForwardBiCGSTABFFTwE.ITERBiCGSTABwE(b, CHI_eps, E_inc, ZH_inc, FFTG, N1, N2, Errcri, itmax, gamma_0, dx)


ITERBiCGSTABwE_1_real = np.real(ITERBiCGSTABwE[0])
ITERBiCGSTABwE_1_imag = np.imag(ITERBiCGSTABwE[0])

