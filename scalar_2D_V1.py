""" This file is the script for the Scalar 2D VDB code
Time factor = ForwardBiCGSTABFFT.exp(-iwt)
Spatial units is in m
Source wavelet  Q = ForwardBiCGSTABFFT.1
"""
from IPython import get_ipython

import sys
import time
from varname import nameof
import numpy as np
from lib import graphviz_doc
from inspect import getmembers, isfunction
# from numba import jit
try:
    from lib import ForwardBiCGSTABFFT
except ImportError:
    import ForwardBiCGSTABFFT

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=18)


# def composer_call(subfolder_name, module_name):
#     subfolder_name = "lib"
#     module_name = "ForwardBiCGSTABFFT"
#     from fn_graph import Composer
#     import importlib
#     module = importlib.import_module(f"{subfolder_name}.{module_name}")
#     attributes = [f"{module_name}.{attr}" for attr in dir(module)]
#     composer = Composer().update(attr for attr in attributes)
#     composer = Composer().update(ForwardBiCGSTABFFT.Aw)
#     return composer
# # subfolder_name = "lib"
# # module_name = "ForwardBiCGSTABFFT"
# # from fn_graph import Composer
# # import importlib

# # module = importlib.import_module(f"{subfolder_name}.{module_name}")
# # attributes = [attr for attr in dir(module)]
# # composer = Composer().update({attr: getattr(module, attr) for attr in attributes if callable(getattr(module, attr))})

# composer = graphviz_doc.composer_render(composer_call("lib", "ForwardBiCGSTABFFT"), '', "digraph")
start_time_wp = time.time()
c_0 = ForwardBiCGSTABFFT.c_0(1500.00)
c_sct = ForwardBiCGSTABFFT.c_sct(3000.00)
f = ForwardBiCGSTABFFT.f(50.00)
s = ForwardBiCGSTABFFT.s(f)
wavelength = ForwardBiCGSTABFFT.wavelength(c_0, f)
gamma_0 = ForwardBiCGSTABFFT.gamma_0(s, c_0)
Errcri = ForwardBiCGSTABFFT.Errcri(1e-15)
xS = ForwardBiCGSTABFFT.xS()
NR = ForwardBiCGSTABFFT.NR(180)
rcvr_phi = ForwardBiCGSTABFFT.rcvr_phi(NR)
xR = ForwardBiCGSTABFFT.xR(NR, rcvr_phi)
N1 = ForwardBiCGSTABFFT.N1(120)
N2 = ForwardBiCGSTABFFT.N2(100)
M = ForwardBiCGSTABFFT.M(100)
dx = ForwardBiCGSTABFFT.dx(2.0)
a = ForwardBiCGSTABFFT.a(40)
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
CHI = ForwardBiCGSTABFFT.CHI(c_0, c_sct, R, a)
gam_sct = ForwardBiCGSTABFFT.gam_sct(gamma_0, c_0, c_sct)
arg0 = ForwardBiCGSTABFFT.arg0(gamma_0, a)
args = ForwardBiCGSTABFFT.args(gam_sct, a)
WavefieldSctCircle = ForwardBiCGSTABFFT.WavefieldSctCircle(M, arg0, args, gam_sct, gamma_0, xR, xS)
angle = ForwardBiCGSTABFFT.angle(rcvr_phi)
ForwardBiCGSTABFFT.displayDataBesselApproach(WavefieldSctCircle, angle)
factoru = ForwardBiCGSTABFFT.factoru(gamma_0, delta)
u_inc = ForwardBiCGSTABFFT.u_inc(gamma_0, xS, X1cap, X2cap, factoru)
itmax = ForwardBiCGSTABFFT.itmax(1000)
b = ForwardBiCGSTABFFT.b(CHI, u_inc, N1, N2)
x0 = ForwardBiCGSTABFFT.x0(b)

w_out = ForwardBiCGSTABFFT.ITERBiCGSTABw(b, CHI, u_inc, FFTG, N1, N2, Errcri, itmax, x0)

ForwardBiCGSTABFFT.plotContrastSource(w_out, CHI, X1cap, X2cap)
Dop_val = ForwardBiCGSTABFFT.Dop(w_out, gamma_0, dx, xR, NR, X1cap, X2cap, delta, factoru, N1, N2)
ForwardBiCGSTABFFT.displayDataCSIEApproach(Dop_val, angle)
ForwardBiCGSTABFFT.displayDataCompareApproachs(WavefieldSctCircle, Dop_val, angle)
time_total_wp = time.time() - start_time_wp

# Iterate over local variables and delete the types that are None to keep workspace tidy
local_variables = list(locals().keys())
for var_name in local_variables:
    var_value = locals()[var_name]
    if var_value is None:
        del locals()[var_name]
        # print(f"Deleted local variable: {var_name}")
