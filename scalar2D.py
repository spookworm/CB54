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
np.set_printoptions(precision=20)

c_0 = solver_func.c_0(1500)
c_sct = solver_func.c_sct(c_0*2.0)
f = solver_func.f(50.0)
NR = solver_func.NR(180)
Errcri = solver_func.Errcri(1e-13)
a = solver_func.a(40)
N1 = solver_func.N1(120)
N2 = solver_func.N2(100)
dx = solver_func.dx(2.0)
M = solver_func.M(100)
itmax = solver_func.itmax(1000)

xS = solver_func.xS()
rcvr_phi = solver_func.rcvr_phi(NR)
xR = solver_func.xR(NR, rcvr_phi)

s = solver_func.s(f)
wavelength = solver_func.wavelength(c_0, f)
gamma_0 = solver_func.gamma_0(s, c_0)
initGrid = solver_func.initGrid(N1, N2, dx)
X1cap = solver_func.X1cap(initGrid)
X2cap = solver_func.X2cap(initGrid)
x1fft = solver_func.x1fft(N1, dx)
x2fft = solver_func.x2fft(N2, dx)
angle = solver_func.angle(rcvr_phi)

gamma_sct = solver_func.gamma_sct(gamma_0, c_0, c_sct)
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

u_inc = solver_func.u_inc(gamma_0, xS, X1cap, X2cap, factoru)
b = solver_func.b(CHI, u_inc, N1, N2)

x0 = solver_func.x0(b)

time_start_wp = time.time()
w_out, exit_code, iterative_information = solver_func.ITERBiCGSTABw(b, CHI, FFTG, N1, N2, Errcri, itmax, x0)
time_total_wp = time.time() - time_start_wp
solveremf2_plot.graph_resivec_ter(iterative_information)

# Display the convergence information
print("exit_code:", exit_code)
print("iter,\tresvec,\ttime_total")
for i, row in enumerate(iterative_information):
    print(f"{row[0]}\t{row[1]}\t{row[2]}")
    print()

relres = iterative_information[-1, 1]/np.linalg.norm(b)
print("relres", relres)
# matlab_relres = 0.00000000000008873759176939078997
# relres - matlab_relres
print("time_total_wp", time_total_wp)


solveremf2_plot.plotContrastSource(w_out, CHI, X1cap, X2cap)
Dop_val = solver_func.Dop(w_out, gamma_0, dx, xR, NR, X1cap, X2cap, delta, factoru, N1, N2)
solveremf2_plot.displayDataCSIEApproach(Dop_val, angle)
solveremf2_plot.displayDataCompareApproachs(WavefieldSctCircle, Dop_val, angle)

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
