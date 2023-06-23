from IPython import get_ipython
# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
get_ipython().run_line_magic('reset', '-sf')

from numba import jit
import numpy as np
try:
    from lib import vdb
except ImportError:
    import vdb


def c_0():
    # wave speed in embedding
    return 1500.00


def c_sct():
    # wave speed in scatterer
    return 3000.00


def f():
    # temporal frequency
    return 50.00


def wavelength(c_0, f):
    # wavelength
    return c_0 / f


def s(f):
    # Laplace Parameter
    import math
    return 1e-16 - 1j * 2 * math.pi * f


def gamma_0(s, c_0):
    # Propagation Co-efficient
    return s / c_0


def NR():
    # Reciever Count
    return 180


def xS():
    # Source Position
    return [-170, 0]


def rcvr_phi(NR):
    # input.rcvr_phi(1:input.NR) = (1:input.NR) * 2 * pi / input.NR;
    import numpy as np
    return np.arange(1, NR+1, 1, dtype=np.double) * 2 * np.pi / NR


def xR(NR, rcvr_phi):
    # Receiver Positions
    import numpy as np
    xR = np.full((2, NR), np.inf)
    # input.xR(1, 1:input.NR) = 150 * cos(input.rcvr_phi);
    xR[0, :] = 150.00 * np.cos(rcvr_phi)
    # input.xR(2, 1:input.NR) = 150 * sin(input.rcvr_phi);
    xR[1, :] = 150.00 * np.sin(rcvr_phi)
    return xR


def N1():
    # number of samples in x_1
    return 120


def N2():
    # number of samples in x_2
    return 100


def dx():
    # meshsize dx
    return 2.0


def initGrid(N1, N2, dx):
    # Grid in two-dimensional space
    import numpy as np
    # x1 = -(input.N1 + 1) * input.dx / 2 + (1:input.N1) * input.dx;
    x1 = -(N1 + 1) * dx / 2.0 + np.arange(1, N1+1, 1, dtype=np.double) * dx
    # x2 = -(input.N2 + 1) * input.dx / 2 + (1:input.N2) * input.dx;
    x2 = -(N2 + 1) * dx / 2.0 + np.arange(1, N2+1, 1, dtype=np.double) * dx
    # [input.X1, input.X2] = ndgrid(x1, x2);
    return np.meshgrid(x1, x2)


def initFFTGreen(N1, N2, dx):
    # Compute FFT of Green function
    import numpy as np

    # N1fft = 2^ceil(log2(2*input.N1));
    N1fft = (2**np.ceil(np.log2(2 * N1))).astype(int)

    # N2fft = 2^ceil(log2(2*input.N2));
    N2fft = (2**np.ceil(np.log2(2 * N2))).astype(int)

    # x1(1:N1fft) = [0 : N1fft / 2 - 1, N1fft / 2 : -1 : 1] * input.dx;
    x1fft = [i * dx for i in (list(range(0, N1fft//2)) + list(range(N1fft//2, 0, -1)))]

    # x2(1:N2fft) = [0 : N2fft / 2 - 1, N2fft / 2 : -1 : 1] * input.dx;
    x2fft = [i * dx for i in (list(range(0, N2fft//2)) + list(range(N2fft//2, 0, -1)))]
    return np.meshgrid(x1fft, x2fft)


def IntG(dx, gamma_0, X1fft, X2fft):
    # Compute gam_0^2 * subdomain integrals of Green function
    import numpy as np
    from scipy.special import kv, iv
    DIS = np.sqrt(X1fft**2 + X2fft**2)

    # Avoid Green's singularity for DIS = 0
    DIS[0, 0] = 1.0
    # print(type(gamma_0*DIS))

    # G = 1 / (2 * pi) .* besselk(0, gamma_0*DIS);
    G = 1.0 / (2.0 * np.pi) * kv(0, gamma_0 * DIS)

    # # Radius circle with area of dx^2
    delta = (np.pi)**(-0.5) * dx

    # factor = 2 * besseli(1, gamma_0*delta) / (gamma_0 * delta)
    factor = 2 * iv(1, gamma_0 * delta) / (gamma_0 * delta)

    # Integral includes gamma_0**2
    # IntG = (gamma_0^2 * dx^2) * factor^2 * G;
    IntG = (gamma_0**2 * dx**2) * factor**2 * G

    # IntG(1, 1) = 1 - gamma_0 * delta * besselk(1, gamma_0*delta) * factor
    IntG[0, 0] = 1 - gamma_0 * delta * kv(1, gamma_0 * delta) * factor
    return IntG


def a():
    # Radius Circle Cylinder
    return 40


def contrast(c_0, c_sct):
    return 1 - c_0**2 / c_sct**2


def R(X1, X2):
    import numpy as np
    return np.sqrt(X1**2 + X2**2)


def CHI(contrast, a, R):
    return contrast * (R < a)


def Errcri():
    return 1e-3


def M():
    # Increase M for more accuracy
    return 100


def WavefieldSctCircle(c_0, c_sct, gamma_0, xR, xS, M):
    import numpy as np
    from scipy.special import iv, kv

    gam_sct = gamma_0 * c_0 / c_sct

    # Compute coefficients of series expansion
    arg0 = gamma_0 * a
    args = gam_sct * a

    A = np.zeros((1, M+1), dtype=np.complex_)
    for m in range(0, M+1):
        Ib0 = iv(m, arg0)
        dIb0 = iv(m+1, arg0) + m / arg0 * Ib0
        Ibs = iv(m, args)
        dIbs = iv(m+1, args) + m / args * Ibs
        Kb0 = kv(m, arg0)
        dKb0 = -kv(m+1, arg0) + m / arg0 * Kb0
        A[0, m] = -(gam_sct * dIbs * Ib0 - gamma_0 * dIb0 * Ibs) / (gam_sct * dIbs * Kb0 - gamma_0 * dKb0 * Ibs)

    # Compute reflected field at receivers (data)
    # rR = sqrt(xR(1, :).^2+xR(2, :).^2)
    # rR = np.sqrt(xR[0, :]**2 + xR[1, :]**2)
    rR = np.linalg.norm(xR, axis=0)

    # phiR = atan2(xR(2, :), xR(1, :))
    phiR = np.arctan2(xR[1, :], xR[0, :])

    # rS = sqrt(xS(1)^2+xS(2)^2)
    # rS = np.sqrt(xS[0]**2 + xS[1]**2)
    rS = np.linalg.norm(xS, axis=0)

    # phiS = atan2(xS(2), xS(1))
    phiS = np.arctan2(xS[1], xS[0])

    # data2D = A(1) * besselk(0, gam0*rS) .* besselk(0, gam0*rR);
    data2D = A[0, 1] * kv(0, gamma_0 * rS) * kv(0, gamma_0 * rR)

    for m in range(0, M+1):
        # factor = 2 * besselk(m, gam0*rS) .* cos(m*(phiS - phiR));
        factor = 2 * kv(m, gamma_0 * rS) * np.cos(m * (phiS - phiR))

        # data2D = data2D + A(m+1) * factor .* besselk(m, gamma_0*rR);
        data2D = data2D + A[0, m] * factor * kv(m, gamma_0 * rR)

    # data2D = 1 / (2 * pi) * data2D;
    data2D = 1 / (2 * np.pi) * data2D

    return data2D


def data_save(path, filename, data2D):
    # save DATA2D data2D;
    np.savetxt(path + filename + '.txt', data2D.view(float))


def data_load(path, filename):
    # load DATA2D data2D;
    return np.loadtxt(path + filename).view(complex)


def displayDataBesselApparoach(data, rcvr_phi):
    import numpy as np
    import matplotlib.pyplot as plt
    angle = rcvr_phi * 180 / np.pi

    # Plot data at a number of receivers
    # fig = plt.figure(figsize=(0.39, 0.39), dpi=100)
    plt.plot(angle, abs(data), label='Bessel-function method')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.title('scattered wave data in 2D', fontsize=12)
    plt.axis('tight')
    plt.xlabel('observation angle in degrees')
    plt.xlim([0, 360])
    plt.ylabel('abs(data) $\\rightarrow$')
    plt.show()


def u_inc(gamma_0, xS, dx, X1, X2):
    # incident wave on two-dimensional grid
    import numpy as np
    from scipy.special import kv, iv

    # DIS = sqrt((X1 - xS(1)).^2+(X2 - xS(2)).^2);
    DIS = np.sqrt((X1 - xS[0])**2 + (X2 - xS[1])**2)
    # DIS = np.sqrt(np.sum((np.array([X1, X2]) - np.array(xS))**2))

    # G = 1 / (2 * pi) .* besselk(0, gam0*DIS);
    G = 1 / (2 * np.pi) * kv(0, gamma_0 * DIS)

    # Radius Circle with Area of dx^2
    delta = (np.pi)**(-0.5) * dx
    factor = 2 * iv(1, gamma_0 * delta) / (gamma_0 * delta)

    # Factor for weak form if DIS > delta
    return factor * G


def itmax():
    return 1000


# MAIN DESCRIPTION
# Time factor = exp(-iwt)
# Spatial units is in m
# Source wavelet  Q = 1

c_0 = c_0()
c_sct = c_sct()
f = f()
wavelength = wavelength(c_0, f)
s = s(f)
gamma_0 = gamma_0(s, c_0)
NR = NR()
xS = xS()
rcvr_phi = rcvr_phi(NR)
xR = xR(NR, rcvr_phi)
N1 = N1()
N2 = N2()
dx = dx()
# THESE MAY BE TRANSPOSED, SO IF THERE'S A ROTATION OR DIMENSION ISSUE CHECK HERE
X1, X2 = initGrid(N1, N2, dx)
X1fft, X2fft = initFFTGreen(N1, N2, dx)
IntG = IntG(dx, gamma_0, X1fft, X2fft)
# Apply n-dimensional Fast Fourier transform
FFTG = np.fft.fftn(IntG)
a = a()
contrast = contrast(c_0, c_sct)
R = R(X1, X2)
CHI = CHI(contrast, a, R)
Errcri = Errcri()
M = M()
WavefieldSctCircle = WavefieldSctCircle(c_0, c_sct, gamma_0, xR, xS, M)
data_save('', 'data2D', WavefieldSctCircle)
data_load = data_load('', 'data2D.txt')
# ok so there is a trick here.
# when the function is initially called it assumes that the data doesn't exist.
# when the function runs and there already exists data, it compares this new data to the saved data.
# the initial run is the bessel-function approach while the second os the contrast source MoM
displayDataBesselApparoach(data_load, rcvr_phi)
u_inc = u_inc(gamma_0, xS, dx, X1, X2)
itmax = itmax()


def Aw(w, N1, N2, FFTG, CHI):
    # Define the matrix-vector multiplication function Aw
    # from scipy.sparse import csc_matrix
    w = w.reshape((N2, N1))
    y = w - CHI * Kop(w, FFTG)
    # return csc_matrix(y)
    y = y.flatten()
    return y


def ITERBiCGSTABw(CHI, u_inc, FFTG, N1, N2, Errcri, itmax):
    # BiCGSTAB_FFT scheme for contrast source integral equation Aw = b
    from scipy.sparse.linalg import bicgstab
    from scipy.sparse.linalg import LinearOperator
    import time

    def callback(xk):
        # Define the callback function
        callback.iter_count += 1
        residual_norm = np.linalg.norm(b - Aw_operator(xk))
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

    # Known 1D vector right-hand side
    # b = input.CHI(:) .* u_inc(:);
    b = CHI.flatten() * u_inc.flatten()

    # Call bicgstab with the LinearOperator instance and other inputs
    # w = bicgstab(@(w) Aw(w, input), b, Errcri, itmax);
    Aw_operator = LinearOperator((N1*N2, N1*N2), matvec=lambda w: Aw(w, N1, N2, FFTG, CHI))
    w, exit_code = bicgstab(Aw_operator, b.flatten(), tol=Errcri, maxiter=itmax, callback=callback)

    # Output Matrix
    w = vector2matrix(w, N1, N2)

    # Display the convergence information
    print("Convergence information:", exit_code)
    print(exit_code)

    print("Iteration:", callback.iter_count)
    print("time_total", callback.time_total)

    return w


def vector2matrix(w, N1, N2):
    # Modify vector output from 'bicgstab' to matrix for further computations
    # w = reshape(w, [input.N1, input.N2]);
    w = w.reshape((N1, N2))
    return w


def Kop(v, FFTG):
    import numpy as np
    # from numpy.fft import fftn, ifftn

    # Make fft grid
    Cv = np.zeros(FFTG.shape, dtype=np.complex128)
    N1, N2 = v.shape

    # Cv(1:N1, 1:N2) = v;
    Cv[0:N1, 0:N2] = v

    # Cv = fftn(Cv);
    Cv = np.fft.fftn(Cv)

    # Convolution by fft
    Cv = np.fft.ifftn(FFTG * Cv)

    # Kv = Cv(1:N1, 1:N2);
    Kv = Cv[0:N1, 0:N2]
    return Kv


w = ITERBiCGSTABw(CHI, u_inc, FFTG, N1, N2, Errcri, itmax)
# print("w: ", w)
print("w.shape: ", w.shape)

print("np.real(w).min()", np.real(w).min())
print("np.real(w).max()", np.real(w).max())
print("np.imag(w).min()", np.imag(w).min())
print("np.imag(w).max()", np.imag(w).max())








# # @jit(nopython=True, parallel=True)
# # def init():
# #     print()

# # # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
# # start = time.perf_counter()
# # wavelength(c_0, f)
# # end = time.perf_counter()
# # print("Elapsed (with compilation) = {}s".format((end - start)))

# # # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
# # start = time.perf_counter()
# # wavelength(c_0, f)
# # end = time.perf_counter()
# # print("Elapsed (after compilation) = {}s".format((end - start)))


# function plotContrastSource(w, input)
# CHI = input.CHI;

# % Plot 2D contrast/source distribution -------------
# x1 = input.X1(:, 1);
# x2 = input.X2(1, :);
# set(figure, 'Units', 'centimeters', 'Position', [5, 5, 18, 12]);
# subplot(1, 2, 1)
# IMAGESC(x1, x2, CHI);
# title('\fontsize{13} \chi = 1 - c_0^2 / c_{sct}^2');
# subplot(1, 2, 2)
# IMAGESC(x1, x2, abs(w))
# title('\fontsize{13} abs(w)');
