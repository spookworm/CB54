from IPython import get_ipython
import numpy as np
import sys
import time

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=20)


def mat_checker(variable, var_name):
    import scipy.io
    # mat_checker(x1fft, nameof(x1fft))
    var_name_matlab = scipy.io.loadmat(var_name + '.mat')
    var_name_m = np.array(var_name_matlab[var_name])
    are_equal = np.array_equal(variable, var_name_m)
    if are_equal is False:
        print(var_name, "are_equal:", are_equal)
    are_close = np.allclose(variable, var_name_m, atol=1e-15)
    if are_close is False:
        print(var_name, "are_close:", are_close)
    var_diff = variable - var_name_m
    max_test = np.max(var_diff)
    if max_test != 0.0:
        print(var_name, "max diff: ", max_test)
        return var_diff
    else:
        return None


def mat_loader(var_name):
    import scipy.io
    # x1fft_m = mat_loader(nameof(x1fft))
    var_name_matlab = scipy.io.loadmat(var_name + '.mat')
    var_name_m = np.array(var_name_matlab[var_name], dtype=np.complex_)
    return var_name_m


def init():
    # Time factor = exp(-iwt)
    # Spatial units is in m
    # Source wavelet  Q = 1

    # wave speed in embedding
    c_0 = 3e8
    # wave speed in scatterer
    c_sct = 1.75



    # temporal frequency
    f = 10e6
    # wavelength
    wavelength = c_0 / f
    # Laplace parameter
    s = 1e-16 - 1j*2*np.pi*f
    # propagation coefficient
    gamma_0 = s/c_0

    # add location of source/receiver
    xS, NR, rcvr_phi, xR = initSourceReceiver()

    # add grid in either 1D, 2D or 3D
    # initGrid() and initGridEM() are equivalent
    N1, N2, dx, X1, X2 = initGrid()

    # compute FFT of Green function
    FFTG = initFFTGreen(N1, N2, dx, gamma_0)

    def initContrast(X1, X2, c_sct):
        # half width slab / radius circle cylinder / radius sphere
        a = 40
        R = np.sqrt(X1**2 + X2**2)

        CHI = (1-c_sct) * (R < a)




        return a, CHI

    # add contrast distribution
    a, CHI = initContrast(X1, X2, c_sct)

    Errcri = 1e-18
    return c_0, c_sct, gamma_0, xS, NR, rcvr_phi, xR, N1, N2, dx, X1, X2, FFTG, a, CHI, Errcri


def initSourceReceiver():
    # Source Position
    xS = np.zeros((1, 2), dtype=np.float64, order='F')
    xS[0, 0] = -170.0
    xS[0, 1] = 0.0

    # Receiver Positions
    NR = 180
    rcvr_phi = np.zeros((1, NR), dtype=np.float64, order='F')
    # rcvr_phi[0, :] = np.linspace(1, NR, num=NR)*(2*np.pi)/NR
    rcvr_phi[0, 0:NR] = np.arange(1, NR+1, 1) * 2.0 * np.pi / NR

    xR = np.zeros((2, NR), dtype=np.float64, order='F')
    xR[0, 0:NR] = 150 * np.cos(rcvr_phi)
    xR[1, 0:NR] = 150 * np.sin(rcvr_phi)
    return xS, NR, rcvr_phi, xR


def initGrid():
    # number of samples in x_1
    N1 = 120
    # number of samples in x_2
    N2 = 100
    # with meshsize dx
    dx = 2

    x1 = np.zeros((1, N1), dtype=np.float64, order='F')
    x1[0, :] = -(N1+1)*dx/2 + np.linspace(1, N1, num=N1)*dx

    x2 = np.zeros((1, N2), dtype=np.float64, order='F')
    x2[0, :] = -(N2+1)*dx/2 + np.linspace(1, N2, num=N2)*dx

    # [X1,X2] = ndgrid(x1,x2)
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')

    # Now array subscripts are equivalent with Cartesian coordinates
    # x1 axis points downwards and x2 axis is in horizontal direction
    return N1, N2, dx, X1, X2


def initFFTGreen(N1, N2, dx, gamma_0):
    N1fft = np.int64(2**np.ceil(np.log2(2*N1)))
    N2fft = np.int64(2**np.ceil(np.log2(2*N2)))

    # x1(1:N1fft) = [0 : N1fft/2-1   N1fft/2 : -1 : 1] * input.dx;
    x1 = np.zeros((1, N1fft), dtype=np.float64, order='F')
    x1[0, :] = np.concatenate((np.arange(0, N1fft//2), np.arange(N1fft//2, 0, -1)))*dx

    # x2(1:N2fft) = [0 : N2fft/2-1   N2fft/2 : -1 : 1] * input.dx;
    x2 = np.zeros((1, N2fft), dtype=np.float64, order='F')
    x2[0, :] = np.concatenate((np.arange(0, N2fft//2), np.arange(N2fft//2, 0, -1)))*dx

    # [temp.X1fft,temp.X2fft] = ndgrid(x1,x2);
    X1fft, X2fft = np.meshgrid(x1, x2, indexing='ij')

    def Green(dx, gamma_0):
        from scipy.special import kv, iv
        gam0 = gamma_0
        X1 = X1fft
        X2 = X2fft
        DIS = np.sqrt(X1**2 + X2**2)
        # avoid Green's singularity for DIS = 0
        DIS[0, 0] = 1
        G = 1/(2*np.pi) * kv(0, gam0*DIS)
        # radius circle with area of dx^2
        delta = (np.pi)**(-1/2) * dx
        factor = 2 * iv(1, gam0*delta) / (gam0*delta)
        # integral includes gam0^2
        IntG = (gam0**2 * dx**2) * factor**2 * G
        IntG[0, 0] = 1 - gam0*delta * kv(1, gam0*delta) * factor
        return IntG

    # compute gam_0^2 * subdomain integrals  of Green function
    IntG = Green(dx, gamma_0)

    # apply n-dimensional Fast Fourier transform
    FFTG = np.fft.fftn(IntG)
    return FFTG



def WavefieldSctCircle():
    from scipy.special import kv, iv
    import os

    c_0, c_sct, gamma_0, xS, NR, rcvr_phi, xR, N1, N2, dx, X1, X2, FFTG, a, CHI, Errcri = init()

    gam_sct = gamma_0 * np.sqrt(c_sct)
    if os.path.exists('DATA2D.npy'):
        os.remove('DATA2D.npy')

    # (1) Compute reflected field at receivers (data)
    rR = np.zeros((1, xR.shape[1]), dtype=np.complex128, order='F')
    rR[0, :] = np.sqrt(xR[0, :]**2 + xR[1, :]**2)
    phiR = np.zeros((1, xR.shape[1]), dtype=np.complex128, order='F')
    phiR[0, :] = np.arctan2(xR[1, :], xR[0, :])
    rS = np.sqrt(xS[0, 0]**2 + xS[0, 1]**2)
    phiS = np.arctan2(xS[0, 1], xS[0, 0])

    # (2) Compute coefficients of series expansion
    arg0 = gamma_0 * a
    args = gam_sct*a
    # increase M for more accuracy
    M = 100

    A = np.zeros((1, M+1), dtype=np.complex128)
    for m in range(0, M+1):
        Ib0 = iv(m, arg0)
        dIb0 = iv(m+1, arg0) + m/arg0 * Ib0
        Ibs = iv(m, args)
        dIbs = iv(m+1, args) + m/args * Ibs
        Kb0 = kv(m, arg0)
        dKb0 = -kv(m+1, arg0) + m/arg0 * Kb0
        A[0, m] = - (gam_sct * dIbs*Ib0 - gamma_0 * dIb0*Ibs) / (gam_sct * dIbs*Kb0 - gamma_0 * dKb0*Ibs)

    data2D = A[0, 0] * kv(0, gamma_0*rS) * kv(0, gamma_0*rR)
    for m in range(1, M+1):
        factor = 2 * kv(m, gamma_0*rS) * np.cos(m*(phiS-phiR))
        data2D = data2D + A[0, m] * factor * kv(m, gamma_0*rR)

    data2D = 1/(2*np.pi) * data2D
    angle = rcvr_phi * 180 / np.pi
    displayDataBesselApproach(data2D, angle)
    np.savez('data2D.npz', data=data2D)
    return data2D


def displayDataBesselApproach(WavefieldSctCircle, angle):
    import matplotlib.pyplot as plt
    import numpy as np
    # Plot data at a number of receivers
    # fig = plt.figure(figsize=(0.39, 0.39), dpi=100)
    plt.plot(angle.T, np.abs(WavefieldSctCircle).T, label='Bessel-function method')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.title('scattered wave data in 2D', fontsize=12)
    plt.axis('tight')
    plt.xlabel('observation angle in degrees')
    plt.xlim([0, 360])
    plt.ylabel('abs(data) $\\rightarrow$')
    plt.show()


def IncWave(gamma_0, xS, dx, X1, X2):
    from scipy.special import kv, iv
    # incident wave on two-dimensional grid
    DIS = np.sqrt((X1-xS[0, 0])**2 + (X2-xS[0, 1])**2)
    G = 1/(2*np.pi) * kv(0, gamma_0*DIS)

    # radius circle with area of dx^2
    delta = (np.pi)**(-1/2) * dx
    factor = 2 * iv(1, gamma_0*delta) / (gamma_0*delta)

    # factor for weak form if DIS > delta
    u_inc = factor * G
    return u_inc


def ITERBiCGSTABw(u_inc, CHI, Errcri, x0=None):
    from scipy.sparse.linalg import bicgstab, LinearOperator

    # BiCGSTAB_FFT scheme for contrast source integral equation Aw = b
    itmax = 1000
    # Known 1D vector right-hand side
    # b = CHI(:) * u_inc(:)
    b = np.zeros((u_inc.flatten('F').shape[0], 1), dtype=np.complex128, order='F')
    b[:, 0] = CHI.flatten('F') * u_inc.flatten('F')

    if x0 is None:
        # Create an array of zeros
        x0 = np.zeros(b.shape, dtype=np.complex128, order='F')

    # Aw_operator = LinearOperator((b.shape[0], b.shape[0]), matvec=lambda w: Aw(w, N1, N2, FFTG, CHI))
    def custom_matvec(w):
        return Aw(w, N1, N2, FFTG, CHI)

    Aw_operator = LinearOperator((b.shape[0], b.shape[0]), matvec=custom_matvec)

    def callback(xk):
        callback.iter += 1
        resvec = np.linalg.norm(Aw_operator(xk).T - b.T)
        callback.time_total = time.time() - callback.start_time
        row = np.array([callback.iter, resvec, callback.time_total])
        callback.information = np.vstack((callback.information, row))

    # Initialise iteration count
    callback.start_time = time.time()
    callback.iter = 0
    # callback.information = np.array([[callback.iter, np.linalg.norm(b), time.time() - callback.start_time]])
    callback.information = np.array([[callback.iter, np.linalg.norm(Aw_operator(x0).T - b.T), time.time() - callback.start_time]])
    # callback.information = np.empty((1, 3))

    # Call bicgstab with the LinearOperator instance and other inputs
    w, exit_code = bicgstab(Aw_operator, b, x0=x0, tol=Errcri, maxiter=itmax, callback=callback)

    # Output Matrix
    # w = vector2matrix(w, N1, N2)
    w = w.reshape((N1, N2), order='F')
    return w, exit_code, callback.information


def Aw(w, N1, N2, FFTG, CHI):
    # Convert 1D vector to matrix
    w = vector2matrix(w, N1, N2)
    y = w - CHI * Kop(w, FFTG)
    # Convert matrix to 1D vector
    y = y.flatten('F')
    return y


def vector2matrix(w, N1, N2):
    # Modify vector output from 'bicgstab' to matrix for further computations
    w = w.reshape((N1, N2), order='F')
    return w


def Kop(v, FFTG):
    # Make FFT grid
    N1, N2 = v.shape
    Cv = np.zeros(FFTG.shape, dtype=np.complex128, order='F')
    Cv[0:N1, 0:N2] = v.copy()
    # Convolution by FFT
    Cv = np.fft.fftn(Cv)
    Cv = np.fft.ifftn(FFTG * Cv)
    Kv = np.zeros((N1, N2), dtype=np.complex128, order='F')
    Kv[0:N1, 0:N2] = Cv[0:N1, 0:N2]
    return Kv


def plotContrastSource(w, CHI, X1, X2):
    import matplotlib.pyplot as plt
    # Plot 2D contrast/source distribution
    # x1 = ForwardBiCGSTABFFT.input.X1(:, 1);
    x1 = X1[:, 0]
    # x2 = ForwardBiCGSTABFFT.input.X2(1, :);
    x2 = X2[0, :]
    fig = plt.figure(figsize=(7.09, 4.72))
    fig.subplots_adjust(wspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(CHI, extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax1.set_xlabel('x_2 \u2192')
    ax1.set_ylabel('\u2190 x_1')
    ax1.set_aspect('equal', adjustable='box')
    fig.colorbar(im1, ax=ax1, orientation='horizontal')
    ax1.set_title(r'$\chi =$1 - $c_0^2 / c_{sct}^2$', fontsize=13)
    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(abs(w), extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax2.set_xlabel('x_2 \u2192')
    ax2.set_ylabel('\u2190 x_1')
    ax2.set_aspect('equal', adjustable='box')
    fig.colorbar(im2, ax=ax2, orientation='horizontal')
    ax2.set_title(r'$|w|$', fontsize=13)
    plt.show()


def Dop(w, NR, N1, N2, xR, gamma_0, dx, X1, X2):
    # (4) Compute synthetic data and plot fields and data
    from scipy.special import kv, iv
    data = np.zeros((1, NR), dtype=np.complex128, order='F')
    delta = (np.pi)**(-1/2) * dx
    factor = 2 * iv(1, gamma_0*delta) / (gamma_0*delta)
    G = np.zeros((N1, N2), dtype=np.complex128, order='F')
    for p in range(0, NR):
        DIS = np.sqrt((xR[0, p-1] - X1)**2 + (xR[1, p-1] - X2)**2)
        G = 1.0 / (2.0 * np.pi) * kv(0, gamma_0*DIS)
        data[0, p-1] = (gamma_0**2 * dx**2) * factor * np.sum(G.flatten('F') * w.flatten('F'))
    return data


def displayDataCSIEApproach(Dop, angle):
    import matplotlib.pyplot as plt
    # Plot data at a number of receivers
    # fig = plt.figure(figsize=(0.39, 0.39), dpi=100)
    plt.plot(angle.T, np.abs(Dop).T, label='Integral-equation method')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.title('scattered wave data in 2D', fontsize=12)
    plt.axis('tight')
    plt.xlabel('observation angle in degrees')
    plt.xlim([0, 360])
    plt.ylabel('abs(data) $\\rightarrow$')
    plt.show()


def displayDataCompareApproachs(bessel_approach, CIS_approach, angle):
    import matplotlib.pyplot as plt
    error_num = np.linalg.norm(CIS_approach.flatten('F') - bessel_approach.flatten('F'), ord=1) / np.linalg.norm(bessel_approach.flatten('F'), ord=1)
    # error = format(error_num, 'f')
    error = str(error_num)
    print("error_num", error_num)
    # Plot data at a number of receivers
    # fig = plt.figure(figsize=(0.39, 0.39), dpi=100)
    plt.tight_layout()
    plt.plot(angle.T, np.abs(CIS_approach).T, '--r', angle.T, np.abs(bessel_approach).T, 'b')
    plt.legend(['Integral-equation method', 'Bessel-function method'], loc='upper center')
    plt.text(0.5*np.max(angle), 0.8*np.max(np.abs(CIS_approach)), 'Error$^{sct}$ = ' + error, color='red', ha='center', va='center')
    plt.title('scattered wave data in 2D', fontsize=12)
    plt.axis('tight')
    plt.xlabel('observation angle in degrees')
    plt.xlim([0, 360])
    plt.ylabel('abs(data) $\\rightarrow$')
    plt.show()


# START OF ForwardBiCGSTABFFTw
data2D = WavefieldSctCircle()
# from varname import nameof
# data2D_diff = mat_checker(data2D, nameof(data2D))
# data2D_m = mat_loader(nameof(data2D))
# np.linalg.norm(data2D - data2D_m, ord=1)
# np.max(np.real(data2D_diff))
# np.max(np.imag(data2D_diff))

c_0, c_sct, gamma_0, xS, NR, rcvr_phi, xR, N1, N2, dx, X1, X2, FFTG, a, CHI, Errcri = init()
# from varname import nameof
# FFTG_diff = mat_checker(FFTG, nameof(FFTG))
# FFTG_m = mat_loader(nameof(FFTG))
# np.linalg.norm(FFTG - FFTG_m, ord=1)
# from varname import nameof
# CHI_diff = mat_checker(CHI, nameof(CHI))
# CHI_m = mat_loader(nameof(CHI))
# np.linalg.norm(CHI - CHI_m, ord=1)


u_inc = IncWave(gamma_0, xS, dx, X1, X2)

# (3) Solve integral equation for contrast source with FFT
tic0 = time.time()
w, exit_code, information = ITERBiCGSTABw(u_inc, CHI, Errcri)
toc0 = time.time() - tic0

# from varname import nameof
# w_diff = mat_checker(w, nameof(w))
# w_m = mat_loader(nameof(w))
# np.linalg.norm(w - w_m, ord=1)

# tic = time.time()
# w, exit_code, information = ITERBiCGSTABw(u_inc, CHI, Errcri, x0=w.flatten('F'))
# toc = time.time() - tic

plotContrastSource(w, CHI, X1, X2)
data = Dop(w, NR, N1, N2, xR, gamma_0, dx, X1, X2)

# from varname import nameof
# data_diff = mat_checker(data, nameof(data))
# data_m = mat_loader(nameof(data))
# np.max(np.real(data_diff))
# np.max(np.imag(data_diff))

angle = rcvr_phi * 180 / np.pi
displayDataCSIEApproach(data, angle)
displayDataCompareApproachs(data2D, data, angle)
error = str(np.linalg.norm(data.flatten('F') - data2D.flatten('F'), ord=1)/np.linalg.norm(data2D.flatten('F'), ord=1))
error
