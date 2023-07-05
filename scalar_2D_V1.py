""" This file is the script for the Scalar 2D VDB code
Time factor = exp(-iwt)
Spatial units is in m
Source wavelet  Q = 1
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

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=18)

start_time_wp = time.time()
c_0 = ForwardBiCGSTABFFT.c_0(3e8)
c_sct = ForwardBiCGSTABFFT.c_sct(c_0*2.750)

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

gamma_sct = gamma_0 * c_0 / c_sct
arg0 = ForwardBiCGSTABFFT.arg0(gamma_0, a)
args = ForwardBiCGSTABFFT.args(gamma_sct, a)

rR = ForwardBiCGSTABFFT.rR(xR)
phiR = ForwardBiCGSTABFFT.phiR(xR)
rS = ForwardBiCGSTABFFT.rS(xS)
phiS = ForwardBiCGSTABFFT.phiS(xS)
WavefieldSctCircle = ForwardBiCGSTABFFT.WavefieldSctCircle(M, arg0, args, gamma_sct, gamma_0, xR, xS, rR, phiR, rS, phiS)
solveremf2_plot.displayDataBesselApproach(WavefieldSctCircle, angle)
initFFTGreen = ForwardBiCGSTABFFT.initFFTGreen(x1fft, x2fft)
X1fftcap = ForwardBiCGSTABFFT.X1fft(initFFTGreen)
X2fftcap = ForwardBiCGSTABFFT.X2fft(initFFTGreen)
delta = ForwardBiCGSTABFFT.delta(dx)
IntG = ForwardBiCGSTABFFT.IntG(dx, gamma_0, X1fftcap, X2fftcap, N1, N2, delta)
FFTG = ForwardBiCGSTABFFT.FFTG(IntG)
R = ForwardBiCGSTABFFT.R(X1cap, X2cap)

CHI = ForwardBiCGSTABFFT.CHI(c_0, c_sct, R, a)

factoru = ForwardBiCGSTABFFT.factoru(gamma_0, delta)
itmax = ForwardBiCGSTABFFT.itmax(1000)

u_inc = ForwardBiCGSTABFFT.u_inc(gamma_0, xS, X1cap, X2cap, factoru)
b = ForwardBiCGSTABFFT.b(CHI, u_inc, N1, N2)
x0 = ForwardBiCGSTABFFT.x0(b)

w_out = ForwardBiCGSTABFFT.ITERBiCGSTABw(b, CHI, FFTG, N1, N2, Errcri, itmax, x0)
solveremf2_plot.plotContrastSource(w_out, CHI, X1cap, X2cap)

Dop_val = ForwardBiCGSTABFFT.Dop(w_out, gamma_0, dx, xR, NR, X1cap, X2cap, delta, factoru, N1, N2)
solveremf2_plot.displayDataCSIEApproach(Dop_val, angle)
solveremf2_plot.displayDataCompareApproachs(WavefieldSctCircle, Dop_val, angle)
time_total_wp = time.time() - start_time_wp
workspace_func.tidy_workspace()

from scipy.io import loadmat, savemat
savemat('w_P.mat', {'w': w_out})
var_name_pyt = loadmat('w_P.mat')['w']
var_name_mat = loadmat('w_mat.mat')['w']

var_diff = var_name_pyt - var_name_mat
np.max(var_diff)
np.max(np.real(var_diff))
np.max(np.imag(var_diff))
workspace_func.plotDiff(var_diff, X1cap, X2cap)
