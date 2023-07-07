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
import time
from scipy.io import loadmat, savemat

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=20)

# USER INPUTS
c_0 = solver_func.c_0(1500.0)
contrast_sct = solver_func.contrast_sct(2.0)
f = solver_func.f(50)
Errcri = solver_func.Errcri(1e-13)
a = solver_func.a(40)

N1 = solver_func.N1(120)
N2 = solver_func.N2(100)
dx = solver_func.dx(2.0)
itmax = solver_func.itmax(1000)
xS = solver_func.xS()

c_sct = solver_func.c_sct(c_0, contrast_sct)
s = solver_func.s(f)
wavelength = solver_func.wavelength(c_0, f)
gamma_0 = solver_func.gamma_0(s, c_0)
initGrid = solver_func.initGrid(N1, N2, dx)
X1cap = solver_func.X1cap(initGrid)
X2cap = solver_func.X2cap(initGrid)
x1fft = solver_func.x1fft(N1, dx)
x2fft = solver_func.x2fft(N2, dx)
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
x0_naive = solver_func.x0_naive(b)


def data_gen(b, CHI, FFTG, N1, N2, Errcri, itmax, x0):
    time_start = time.time()
    w, exit_code, iterative_info = solver_func.ITERBiCGSTABw(b, CHI, FFTG, N1, N2, Errcri, itmax, x0)
    time_total = time.time() - time_start
    # # Display the convergence information
    # print("exit_code:", exit_code)
    # print("iter,\tresvec,\ttime_total")
    for i, row in enumerate(iterative_info):
        print(f"{row[0]}\t{row[1]}\t{row[2]}")
        print()
    # relres = iterative_info[-1, 1]/np.linalg.norm(b)
    # print("relres", relres)
    return time_total, w, exit_code, iterative_info


time_total_naive, w_naive, exit_code_naive, iterative_info_naive = data_gen(b, CHI, FFTG, N1, N2, Errcri, itmax, x0_naive)
x0_model = w_naive.flatten('F') * np.random.rand(*w_naive.shape).flatten('F')
time_total_model, w_model, exit_code_model, iterative_info_model = data_gen(b, CHI, FFTG, N1, N2, Errcri, itmax, x0_model)

# COMPARE THE RESVEC GRAPHS FOR THE NAIVE AND THE MODEL INFORMED ITERATIVE INFORMATION

# solveremf2_plot.plotContrastSource(w_model, CHI, X1cap, X2cap)

bessel_check = 0
if bessel_check == 1:
    M = solver_func.M(100)
    NR = solver_func.NR(180)
    rcvr_phi = solver_func.rcvr_phi(NR)
    xR = solver_func.xR(NR, rcvr_phi)
    Dop_val = solver_func.Dop(w_model, gamma_0, dx, xR, NR, X1cap, X2cap, delta, factoru, N1, N2)
    angle = solver_func.angle(rcvr_phi)
    solveremf2_plot.displayDataCSIEApproach(Dop_val, angle)
    gamma_sct = solver_func.gamma_sct(gamma_0, c_0, c_sct)
    arg0 = solver_func.arg0(gamma_0, a)
    args = solver_func.args(gamma_sct, a)
    rR = solver_func.rR(xR)
    phiR = solver_func.phiR(xR)
    rS = solver_func.rS(xS)
    phiS = solver_func.phiS(xS)
    WavefieldSctCircle = solver_func.WavefieldSctCircle(M, arg0, args, gamma_sct, gamma_0, xR, xS, rR, phiR, rS, phiS)
    solveremf2_plot.displayDataBesselApproach(WavefieldSctCircle, angle)
    solveremf2_plot.displayDataCompareApproachs(WavefieldSctCircle, Dop_val, angle)

# Validate code against MATLAB output
if (c_0 == 1500) and (c_sct == 3000) and (f == 50) and (itmax == 1000) and (Errcri == 1e-13):
    savemat('w_P.mat', {'w': w_model})
    var_name_pyt = loadmat('w_P.mat')['w']
    var_name_mat = loadmat('./code_ref/ScalarWavesMfiles/w_mat.mat')['w']

    var_diff = var_name_pyt - var_name_mat
    np.max(var_diff)
    np.max(np.real(var_diff))
    np.max(np.imag(var_diff))
    workspace_func.plotDiff(var_diff, X1cap, X2cap)
    print("Comaprision made...")
    # os.remove('w_P.mat')

workspace_func.tidy_workspace()
