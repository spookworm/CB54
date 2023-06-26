""" This file is the script for the Electric Eps VDB code
Time factor = exp(-iwt)
Spatial units is in m
Source wavelet M Z_0 / gamma_0  = 1   (Z_0 M = gamma_0)
"""
from IPython import get_ipython
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
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=15)

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
get_ipython().run_line_magic('reset', '-sf')


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
            ForwardBiCGSTABFFT.delta,
            ForwardBiCGSTABFFT.displayDataCSIEApproach,
            ForwardBiCGSTABFFT.displayDataCompareApproachs,
            ForwardBiCGSTABFFT.dx,
            ForwardBiCGSTABFFT.gamma_0,
            ForwardBiCGSTABFFT.initFFTGreen,
            ForwardBiCGSTABFFT.initGrid,
            ForwardBiCGSTABFFT.itmax,
            ForwardBiCGSTABFFT.rcvr_phi,
            ForwardBiCGSTABFFT.s,
            ForwardBiCGSTABFFT.wavelength,
            ForwardBiCGSTABFFT.x1fft,
            ForwardBiCGSTABFFT.x2fft,
            ForwardBiCGSTABFFT.xR,
            ForwardBiCGSTABFFT.xS,
            ForwardBiCGSTABFFTwE.Aw,
            ForwardBiCGSTABFFTwE.CHI_eps,
            ForwardBiCGSTABFFTwE.CHI_mu,
            ForwardBiCGSTABFFTwE.DOPwE,
            ForwardBiCGSTABFFTwE.E,
            ForwardBiCGSTABFFTwE.EMsctCircle,
            ForwardBiCGSTABFFTwE.E_inc,
            ForwardBiCGSTABFFTwE.E_sct,
            ForwardBiCGSTABFFTwE.Edata,
            ForwardBiCGSTABFFTwE.Edata2D,
            ForwardBiCGSTABFFTwE.Hdata,
            ForwardBiCGSTABFFTwE.Hdata2D,
            ForwardBiCGSTABFFTwE.IncEMwave,
            ForwardBiCGSTABFFTwE.Kop,
            ForwardBiCGSTABFFTwE.KopE,
            ForwardBiCGSTABFFTwE.KwE,
            ForwardBiCGSTABFFTwE.M,
            ForwardBiCGSTABFFTwE.N,
            ForwardBiCGSTABFFTwE.ZH_inc,
            ForwardBiCGSTABFFTwE.b,
            ForwardBiCGSTABFFTwE.c_0,
            ForwardBiCGSTABFFTwE.eps_sct,
            ForwardBiCGSTABFFTwE.f,
            ForwardBiCGSTABFFTwE.graddiv,
            ForwardBiCGSTABFFTwE.mu_sct,
            ForwardBiCGSTABFFTwE.phi,
            ForwardBiCGSTABFFTwE.plotContrastSourcewE,
            ForwardBiCGSTABFFTwE.plotEtotalwavefield,
            ForwardBiCGSTABFFTwE.vector2matrix,
            ForwardBiCGSTABFFTwE.w,
            ForwardBiCGSTABFFTwE.w_E,
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

delta = ForwardBiCGSTABFFT.delta(dx)

x1fft = ForwardBiCGSTABFFT.x1fft(N1, dx)
x2fft = ForwardBiCGSTABFFT.x2fft(N2, dx)
initFFTGreen = ForwardBiCGSTABFFT.initFFTGreen(x1fft, x2fft)
X1fft = ForwardBiCGSTABFFT.X1fft(initFFTGreen)
X2fft = ForwardBiCGSTABFFT.X2fft(initFFTGreen)

IntG = ForwardBiCGSTABFFT.IntG(dx, gamma_0, X1fft, X2fft, delta)
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

ForwardBiCGSTABFFTwE.plotEMcontrast(X1, X2, CHI_eps, CHI_mu)

IncEMwave = ForwardBiCGSTABFFTwE.IncEMwave(gamma_0, xS, dx, X1, X2)
E_inc = ForwardBiCGSTABFFTwE.E_inc(IncEMwave)
ZH_inc = ForwardBiCGSTABFFTwE.ZH_inc(IncEMwave)

itmax = ForwardBiCGSTABFFT.itmax()
N = ForwardBiCGSTABFFTwE.N(CHI_eps)
b = ForwardBiCGSTABFFTwE.b(CHI_eps, E_inc, N)
# print(np.real(b)[19871] - 9.533884691144613e-05)
# print(np.real(b)[19872] - 3.541293300961272e-05)

# print(sum(np.real(b)) - (- 0.002404557207356))
# print(sum(np.imag(b)) - (-7.340318325805991e-04))

# print(np.sum(np.real(E_inc[1])) - (-0.019832576190409))
# print(np.sum(np.real(E_inc[2])) - (-1.301042606982605e-18))


w = ForwardBiCGSTABFFTwE.w(b, CHI_eps, E_inc, ZH_inc, FFTG, N1, N2, Errcri, itmax, gamma_0, dx, N)

w_E = ForwardBiCGSTABFFTwE.w_E(w, N1, N2, N)

ForwardBiCGSTABFFTwE.plotContrastSourcewE(w_E, X1, X2)
E_sct = ForwardBiCGSTABFFTwE.E_sct(w_E, FFTG, gamma_0, dx, N1, N2)

print("sum(sum(np.real(w_E[1])))", sum(sum(np.real(w_E[1]))))
print("sum(sum(np.imag(w_E[1])))", sum(sum(np.imag(w_E[1]))))
print("sum(sum(np.real(w_E[2])))", sum(sum(np.real(w_E[2]))))
print("sum(sum(np.imag(w_E[2])))", sum(sum(np.imag(w_E[2]))))

# print("sum(sum(np.real(w_E[1])))", sum(sum(np.real(w_E[1]))) - (-0.001251086990038))
# print("sum(sum(np.imag(w_E[1])))", sum(sum(np.imag(w_E[1]))) - (8.943406881272415e-05))
# print("sum(sum(np.real(w_E[2])))", sum(sum(np.real(w_E[2]))) - (-8.673617379884035e-19))
# print("sum(sum(np.imag(w_E[2])))", sum(sum(np.imag(w_E[2]))) - (2.439454888092385e-18))

E = ForwardBiCGSTABFFTwE.E(E_inc, E_sct)
phi = ForwardBiCGSTABFFTwE.phi()
ForwardBiCGSTABFFTwE.plotEtotalwavefield(E, a, X1, X2, N1, N2, phi)

DOPwE = ForwardBiCGSTABFFTwE.DOPwE(w_E, gamma_0, dx, xR, NR, delta, X1, X2)
Edata = ForwardBiCGSTABFFTwE.Edata(DOPwE)
Hdata = ForwardBiCGSTABFFTwE.Hdata(DOPwE)

ForwardBiCGSTABFFT.displayDataCSIEApproach(Edata, rcvr_phi)
ForwardBiCGSTABFFT.displayDataCSIEApproach(Hdata, rcvr_phi)
ForwardBiCGSTABFFT.displayDataCompareApproachs(Edata2D, Edata, rcvr_phi)
ForwardBiCGSTABFFT.displayDataCompareApproachs(Hdata2D, Hdata, rcvr_phi)
