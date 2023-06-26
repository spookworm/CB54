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
from lib import graphviz_doc


def composer_call():
    from fn_graph import Composer
    composer_1 = (
        Composer()
        .update(
            # list of custom functions goes here
            ForwardBiCGSTABFFT.M,
            ForwardBiCGSTABFFT.NR,
            ForwardBiCGSTABFFT.a,
            ForwardBiCGSTABFFT.c_sct,
            ForwardBiCGSTABFFT.f,
            ForwardBiCGSTABFFT.xS,
            ForwardBiCGSTABFFT.c_0,
            ForwardBiCGSTABFFT.gamma_0,
            ForwardBiCGSTABFFT.rcvr_phi,
            ForwardBiCGSTABFFT.s,
            ForwardBiCGSTABFFT.xR,
            ForwardBiCGSTABFFT.WavefieldSctCircle,
            ForwardBiCGSTABFFT.displayDataBesselApproach,
            ForwardBiCGSTABFFT.Errcri,
            ForwardBiCGSTABFFT.N1,
            ForwardBiCGSTABFFT.N2,
            ForwardBiCGSTABFFT.dx,
            ForwardBiCGSTABFFT.itmax,
            ForwardBiCGSTABFFT.CHI,
            ForwardBiCGSTABFFT.IntG,
            ForwardBiCGSTABFFT.X1,
            ForwardBiCGSTABFFT.X2,
            ForwardBiCGSTABFFT.R,
            ForwardBiCGSTABFFT.contrast,
            ForwardBiCGSTABFFT.x1fft,
            ForwardBiCGSTABFFT.x2fft,
            ForwardBiCGSTABFFT.initFFTGreen,
            ForwardBiCGSTABFFT.X1fft,
            ForwardBiCGSTABFFT.X2fft,
            ForwardBiCGSTABFFT.initFFTGreen,
            ForwardBiCGSTABFFT.initGrid,
            ForwardBiCGSTABFFT.wavelength,
            ForwardBiCGSTABFFT.data_load,
            ForwardBiCGSTABFFT.u_inc,
            ForwardBiCGSTABFFT.FFTG,
            ForwardBiCGSTABFFT.ITERBiCGSTABw,
            ForwardBiCGSTABFFT.plotContrastSource,
            ForwardBiCGSTABFFT.Dop,
            ForwardBiCGSTABFFT.displayDataCSIEApproach,
            ForwardBiCGSTABFFT.displayDataCompareApproachs,
        )
        # .update_parameters(input_length_side=input_length_x_side)
        # .cache()
    )
    return composer_1


composer = graphviz_doc.composer_render(composer_call(), '', "digraph")

M = ForwardBiCGSTABFFT.M()
a = ForwardBiCGSTABFFT.a()
c_sct = ForwardBiCGSTABFFT.c_sct()
xS = ForwardBiCGSTABFFT.xS()
c_0 = ForwardBiCGSTABFFT.c_0()
f = ForwardBiCGSTABFFT.f()
s = ForwardBiCGSTABFFT.s(f)
gamma_0 = ForwardBiCGSTABFFT.gamma_0(s, c_0)
NR = ForwardBiCGSTABFFT.NR()
rcvr_phi = ForwardBiCGSTABFFT.rcvr_phi(NR)
xR = ForwardBiCGSTABFFT.xR(NR, rcvr_phi)
WavefieldSctCircle = ForwardBiCGSTABFFT.WavefieldSctCircle(c_0, c_sct, gamma_0, xR, xS, M, a)
ForwardBiCGSTABFFT.displayDataBesselApproach(WavefieldSctCircle, rcvr_phi)


N1 = ForwardBiCGSTABFFT.N1()
N2 = ForwardBiCGSTABFFT.N2()
dx = ForwardBiCGSTABFFT.dx()
Errcri = ForwardBiCGSTABFFT.Errcri()
itmax = ForwardBiCGSTABFFT.itmax()

wavelength = ForwardBiCGSTABFFT.wavelength(c_0, f)
x1fft = ForwardBiCGSTABFFT.x1fft(N1, dx)
x2fft = ForwardBiCGSTABFFT.x2fft(N2, dx)
initFFTGreen = ForwardBiCGSTABFFT.initFFTGreen(x1fft, x2fft)
X1fft = ForwardBiCGSTABFFT.X1fft(initFFTGreen)
X2fft = ForwardBiCGSTABFFT.X2fft(initFFTGreen)

IntG = ForwardBiCGSTABFFT.IntG(dx, gamma_0, X1fft, X2fft)
contrast = ForwardBiCGSTABFFT.contrast(c_0, c_sct)
initGrid = ForwardBiCGSTABFFT.initGrid(N1, N2, dx)
X1 = ForwardBiCGSTABFFT.X1(initGrid)
X2 = ForwardBiCGSTABFFT.X2(initGrid)
R = ForwardBiCGSTABFFT.R(X1, X2)
CHI = ForwardBiCGSTABFFT.CHI(contrast, a, R)

# Delete existing data if the file exists
file_path = 'data2D.txt'
if os.path.exists(file_path):
    os.remove(file_path)
ForwardBiCGSTABFFT.data_save('', 'data2D', WavefieldSctCircle)
data_load = ForwardBiCGSTABFFT.data_load('', 'data2D.txt')

u_inc = ForwardBiCGSTABFFT.u_inc(gamma_0, xS, dx, X1, X2)
FFTG = ForwardBiCGSTABFFT.FFTG(IntG)
b = ForwardBiCGSTABFFT.b(CHI, u_inc)
ITERBiCGSTABw = ForwardBiCGSTABFFT.ITERBiCGSTABw(b, CHI, u_inc, FFTG, N1, N2, Errcri, itmax)
# # # print("ITERBiCGSTABw: ", ITERBiCGSTABw)
# print("ITERBiCGSTABw.shape: ", ITERBiCGSTABw.shape)
# print("np.real(ITERBiCGSTABw).min()", np.real(ITERBiCGSTABw).min())
# print("np.real(ITERBiCGSTABw).max()", np.real(ITERBiCGSTABw).max())
# print("np.imag(ITERBiCGSTABw).min()", np.imag(ITERBiCGSTABw).min())
# print("np.imag(ITERBiCGSTABw).max()", np.imag(ITERBiCGSTABw).max())
ForwardBiCGSTABFFT.plotContrastSource(ITERBiCGSTABw, CHI, X1, X2)
Dop = ForwardBiCGSTABFFT.Dop(ITERBiCGSTABw, gamma_0, dx, xR, NR, X1, X2)
ForwardBiCGSTABFFT.displayDataCSIEApproach(Dop, rcvr_phi)
ForwardBiCGSTABFFT.displayDataCompareApproachs(WavefieldSctCircle, Dop, rcvr_phi)

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
