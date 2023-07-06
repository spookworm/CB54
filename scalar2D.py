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
import os
import time
from scipy.io import loadmat, savemat

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=18)

time_start_wp = time.time()
c_0 = solver_func.c_0(1500)
c_sct = solver_func.c_sct(c_0*2.0)
f = solver_func.f(50.0)
s = solver_func.s(f)
wavelength = solver_func.wavelength(c_0, f)
gamma_0 = solver_func.gamma_0(s, c_0)
xS = solver_func.xS()
NR = solver_func.NR(180)
Errcri = solver_func.Errcri(1e-13)
rcvr_phi = solver_func.rcvr_phi(NR)
xR = solver_func.xR(NR, rcvr_phi)
a = solver_func.a(40)
N1 = solver_func.N1(120)
N2 = solver_func.N2(100)
dx = solver_func.dx(2.0)
initGrid = solver_func.initGrid(N1, N2, dx)
X1cap = solver_func.X1cap(initGrid)
X2cap = solver_func.X2cap(initGrid)
M = solver_func.M(100)
x1fft = solver_func.x1fft(N1, dx)
x2fft = solver_func.x2fft(N2, dx)
angle = solver_func.angle(rcvr_phi)

gamma_sct = gamma_0 * c_0 / c_sct
arg0 = solver_func.arg0(gamma_0, a)
args = solver_func.args(gamma_sct, a)

rR = solver_func.rR(xR)
phiR = solver_func.phiR(xR)
rS = solver_func.rS(xS)
phiS = solver_func.phiS(xS)
WavefieldSctCircle = solver_func.WavefieldSctCircle(M, arg0, args, gamma_sct, gamma_0, xR, xS, rR, phiR, rS, phiS)
solveremf2_plot.displayDataBesselApproach(WavefieldSctCircle, angle)
initFFTGreen = solver_func.initFFTGreen(x1fft, x2fft)
X1fftcap = solver_func.X1fft(initFFTGreen)
X2fftcap = solver_func.X2fft(initFFTGreen)
delta = solver_func.delta(dx)
IntG = solver_func.IntG(dx, gamma_0, X1fftcap, X2fftcap, N1, N2, delta)
FFTG = solver_func.FFTG(IntG)
R = solver_func.R(X1cap, X2cap)
CHI = solver_func.CHI(c_0, c_sct, R, a)
factoru = solver_func.factoru(gamma_0, delta)
itmax = solver_func.itmax(1000)

u_inc = solver_func.u_inc(gamma_0, xS, X1cap, X2cap, factoru)
b = solver_func.b(CHI, u_inc, N1, N2)
x0 = solver_func.x0(b)

w_out, exit_code, residuals, time_total = solver_func.ITERBiCGSTABw(b, CHI, FFTG, N1, N2, Errcri, itmax, x0)
solveremf2_plot.plotContrastSource(w_out, CHI, X1cap, X2cap)

for i, residual in enumerate(residuals):
    print(f"Iteration {i+1}: {residual}")

Dop_val = solver_func.Dop(w_out, gamma_0, dx, xR, NR, X1cap, X2cap, delta, factoru, N1, N2)
solveremf2_plot.displayDataCSIEApproach(Dop_val, angle)
solveremf2_plot.displayDataCompareApproachs(WavefieldSctCircle, Dop_val, angle)
time_total_wp = time.time() - time_start_wp
workspace_func.tidy_workspace()

# Validate code against MATLAB output
if (c_0 == 1500) and (c_sct == 3000) and (f == 50) and (itmax == 1000) and (Errcri == 1e-13):
    savemat('w_P.mat', {'w': w_out})
    var_name_pyt = loadmat('w_P.mat')['w']
    var_name_mat = loadmat('./code_ref/ScalarWavesMfiles/w_mat.mat')['w']

    var_diff = var_name_pyt - var_name_mat
    np.max(var_diff)
    np.max(np.real(var_diff))
    np.max(np.imag(var_diff))
    workspace_func.plotDiff(var_diff, X1cap, X2cap)
    print("Comaprision made...")
    os.remove('w_P.mat')

