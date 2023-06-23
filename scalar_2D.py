""" This file is the script for the Scalar 2D VDB code
Time factor = ForwardBiCGSTABFFT.exp(-iwt)
Spatial units is in m
Source wavelet  Q = ForwardBiCGSTABFFT.1
"""

# from IPython import get_ipython
# # Clear workspace
# get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')

# from numba import jit
# import numpy as np
# import time
import os
try:
    from lib import ForwardBiCGSTABFFT
except ImportError:
    import ForwardBiCGSTABFFT

c_0 = ForwardBiCGSTABFFT.c_0()
c_sct = ForwardBiCGSTABFFT.c_sct()
f = ForwardBiCGSTABFFT.f()
wavelength = ForwardBiCGSTABFFT.wavelength(c_0, f)
s = ForwardBiCGSTABFFT.s(f)
gamma_0 = ForwardBiCGSTABFFT.gamma_0(s, c_0)
NR = ForwardBiCGSTABFFT.NR()
xS = ForwardBiCGSTABFFT.xS()
rcvr_phi = ForwardBiCGSTABFFT.rcvr_phi(NR)
xR = ForwardBiCGSTABFFT.xR(NR, rcvr_phi)
N1 = ForwardBiCGSTABFFT.N1()
N2 = ForwardBiCGSTABFFT.N2()
dx = ForwardBiCGSTABFFT.dx()
X1, X2 = ForwardBiCGSTABFFT.initGrid(N1, N2, dx)
X1fft, X2fft = ForwardBiCGSTABFFT.initFFTGreen(N1, N2, dx)
IntG = ForwardBiCGSTABFFT.IntG(dx, gamma_0, X1fft, X2fft)
# Apply n-dimensional Fast Fourier transform
FFTG = ForwardBiCGSTABFFT.np.fft.fftn(IntG)
a = ForwardBiCGSTABFFT.a()
contrast = ForwardBiCGSTABFFT.contrast(c_0, c_sct)
R = ForwardBiCGSTABFFT.R(X1, X2)
CHI = ForwardBiCGSTABFFT.CHI(contrast, a, R)
Errcri = ForwardBiCGSTABFFT.Errcri()
M = ForwardBiCGSTABFFT.M()
WavefieldSctCircle = ForwardBiCGSTABFFT.WavefieldSctCircle(c_0, c_sct, gamma_0, xR, xS, M, a)

# Delete existing data if the file exists
file_path = 'data2D.txt'
if os.path.exists(file_path):
    os.remove(file_path)

# ok so there is a trick here.
# when the function is initially called it assumes that the data doesn't exist.
# when the function runs and there already exists data, it compares this new data to the saved data.
# the initial run is the bessel-function approach while the second os the contrast source MoM
ForwardBiCGSTABFFT.displayDataBesselApparoach(WavefieldSctCircle, rcvr_phi)

ForwardBiCGSTABFFT.data_save('', 'data2D', WavefieldSctCircle)
data_load = ForwardBiCGSTABFFT.data_load('', 'data2D.txt')

u_inc = ForwardBiCGSTABFFT.u_inc(gamma_0, xS, dx, X1, X2)
itmax = ForwardBiCGSTABFFT.itmax()
w = ForwardBiCGSTABFFT.ITERBiCGSTABw(CHI, u_inc, FFTG, N1, N2, Errcri, itmax)
# # # print("w: ", w)
# print("w.shape: ", w.shape)
# print("np.real(w).min()", np.real(w).min())
# print("np.real(w).max()", np.real(w).max())
# print("np.imag(w).min()", np.imag(w).min())
# print("np.imag(w).max()", np.imag(w).max())

ForwardBiCGSTABFFT.plotContrastSource(w, CHI, X1, X2)
Dop = ForwardBiCGSTABFFT.Dop(w, gamma_0, dx, xR, NR, X1, X2)

# from line_profiler import LineProfiler
# profiler = LineProfiler()
# profiler.add_function(ForwardBiCGSTABFFT.WavefieldSctCircle)
# profiler.run('ForwardBiCGSTABFFT.WavefieldSctCircle')
# profiler.print_stats()

# # @jit(nopython=True, parallel=True)
# # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
# start = time.perf_counter()
# ForwardBiCGSTABFFT.ITERBiCGSTABw(CHI, u_inc, FFTG, N1, N2, Errcri, itmax)
# end = time.perf_counter()
# print("Elapsed (with compilation) = {}s".format((end - start)))

# # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
# start = time.perf_counter()
# ForwardBiCGSTABFFT.ITERBiCGSTABw(CHI, u_inc, FFTG, N1, N2, Errcri, itmax)
# end = time.perf_counter()
# print("Elapsed (after compilation) = {}s".format((end - start)))


# if __name__ == '__main__':
#     start = ForwardBiCGSTABFFT.perf_counter()
#     load_array()
#     duration = ForwardBiCGSTABFFT.perf_counter() - start
#     print('load_array', duration)

#     start = ForwardBiCGSTABFFT.perf_counter()
#     load_file()
#     duration = ForwardBiCGSTABFFT.perf_counter() - start
#     print('load_file', duration)
