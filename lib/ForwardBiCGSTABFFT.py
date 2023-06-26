# from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv, iv
import time
from scipy.sparse.linalg import bicgstab, LinearOperator


def c_0():
    # wave speed in embedding
    return 1500.00


def c_sct():
    # wave speed in scatterer
    return 3000.00


def f():
    # temporal frequency
    return 50.00


def s(f):
    # Laplace Parameter
    return 1e-16 - 1j * 2 * np.pi * f


def gamma_0(s, c_0):
    # Propagation Co-efficient
    return s / c_0


def NR():
    # Reciever Count
    return 180


def wavelength(c_0, f):
    # wavelength
    return c_0 / f


def xS():
    # Source Position
    return [-170, 0]


def N1():
    # number of samples in x_1
    return 120


def N2():
    # number of samples in x_2
    return 100


def dx():
    # meshsize dx
    return 2.0


def a():
    # Radius Circle Cylinder
    return 40


def Errcri():
    return 1e-3


def M():
    # Increase M for more accuracy
    return 100


def itmax():
    return 1000


def rcvr_phi(NR):
    # input.rcvr_phi(1:input.NR) = (1:input.NR) * 2 * pi / input.NR;
    return np.transpose(np.arange(1, NR+1, 1, dtype=np.double) * 2 * np.pi / NR)


def xR(NR, rcvr_phi):
    # Receiver Positions
    xR = np.full((2, NR), np.inf)
    # input.xR(1, 1:input.NR) = 150 * cos(input.rcvr_phi);
    xR[0, :] = 150.00 * np.cos(rcvr_phi)
    # input.xR(2, 1:input.NR) = 150 * sin(input.rcvr_phi);
    xR[1, :] = 150.00 * np.sin(rcvr_phi)
    return xR


def initGrid(N1, N2, dx):
    # Grid in two-dimensional space
    # x1 = -(input.N1 + 1) * input.dx / 2 + (1:input.N1) * input.dx;
    x1 = -(N1 + 1) * dx / 2.0 + np.arange(1, N1+1, 1, dtype=np.double) * dx

    # x2 = -(input.N2 + 1) * input.dx / 2 + (1:input.N2) * input.dx;
    x2 = -(N2 + 1) * dx / 2.0 + np.arange(1, N2+1, 1, dtype=np.double) * dx

    # [input.X1, input.X2] = ndgrid(x1, x2);
    return np.meshgrid(x1, x2)


def X1(initGrid):
    return initGrid[0].transpose()


def X2(initGrid):
    return initGrid[1].transpose()


def FFTG(IntG):
    # Apply n-dimensional Fast Fourier transform
    return np.fft.fftn(IntG)


def x1fft(N1, dx):
    # Compute FFT of Green function
    # N1fft = 2^ceil(log2(2*input.N1));
    N1fft = (2**np.ceil(np.log2(2 * N1))).astype(int)
    # x1(1:N1fft) = [0 : N1fft / 2 - 1, N1fft / 2 : -1 : 1] * input.dx;
    x1fft = [i * dx for i in (list(range(0, N1fft//2)) + list(range(N1fft//2, 0, -1)))]
    return x1fft


def x2fft(N2, dx):
    # Compute FFT of Green function
    # N2fft = 2^ceil(log2(2*input.N2));
    N2fft = (2**np.ceil(np.log2(2 * N2))).astype(int)
    # x2(1:N2fft) = [0 : N2fft / 2 - 1, N2fft / 2 : -1 : 1] * input.dx;
    x2fft = [i * dx for i in (list(range(0, N2fft//2)) + list(range(N2fft//2, 0, -1)))]
    return x2fft


def initFFTGreen(x1fft, x2fft):
    return np.meshgrid(x1fft, x2fft)


def X1fft(initFFTGreen):
    return initFFTGreen[0]


def X2fft(initFFTGreen):
    return initFFTGreen[1]


def IntG(dx, gamma_0, X1fft, X2fft):
    # Compute gam_0^2 * subdomain integrals of Green function
    DIS = np.sqrt(X1fft**2 + X2fft**2)

    # Avoid Green's singularity for DIS = 0
    DIS[0, 0] = 1.0
    # print(type(gamma_0*DIS))

    # G = 1 / (2 * pi) .* besselk(0, gamma_0*DIS);
    G = 1.0 / (2.0 * np.pi) * kv(0, gamma_0 * DIS)

    # Radius circle with area of dx^2
    delta = np.pi**(-0.5) * dx

    # factor = 2 * besseli(1, gamma_0*delta) / (gamma_0 * delta)
    factor = 2 * iv(1, gamma_0 * delta) / (gamma_0 * delta)

    # Integral includes gamma_0**2
    # IntG = (gamma_0^2 * dx^2) * factor^2 * G;
    IntG = (gamma_0**2 * dx**2) * factor**2 * G

    # IntG(1, 1) = 1 - gamma_0 * delta * besselk(1, gamma_0*delta) * factor
    IntG[0, 0] = 1 - gamma_0 * delta * kv(1, gamma_0 * delta) * factor
    return IntG


def contrast(c_0, c_sct):
    return 1 - c_0**2 / c_sct**2


def R(X1, X2):
    return np.sqrt(X1**2 + X2**2)


def CHI(contrast, a, R):
    return contrast * (R < a)


def data_save(path, filename, data2D):
    # save DATA2D data2D;
    np.savetxt(path + filename + '.txt', data2D.view(float))


def data_load(path, filename):
    # load DATA2D data2D;
    return np.loadtxt(path + filename).view(complex)


def WavefieldSctCircle(c_0, c_sct, gamma_0, xR, xS, M, a):
    import numpy as np
    from scipy.special import kv, iv

    gam_sct = gamma_0 * c_0 / c_sct

    # Compute coefficients of series expansion
    arg0 = gamma_0 * a
    args = gam_sct * a
    A = np.zeros(M+1, dtype=np.complex_)
    for m in range(0, M+1):
        Ib0 = iv(m, arg0)
        dIb0 = iv(m+1, arg0) + m / arg0 * Ib0
        Ibs = iv(m, args)
        dIbs = iv(m+1, args) + m / args * Ibs
        Kb0 = kv(m, arg0)
        dKb0 = -kv(m+1, arg0) + m / arg0 * Kb0
        A[m] = -(gam_sct * dIbs * Ib0 - gamma_0 * dIb0 * Ibs) / (gam_sct * dIbs * Kb0 - gamma_0 * dKb0 * Ibs)

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
    data2D = A[0] * kv(0, gamma_0 * rS) * kv(0, gamma_0 * rR)

    for m in range(1, M+1):
        # factor = 2 * besselk(m, gam0*rS) .* cos(m*(phiS - phiR));
        factor = 2 * kv(m, gamma_0 * rS) * np.cos(m * (phiS - phiR))

        # data2D = data2D + A(m+1) * factor .* besselk(m, gamma_0*rR);
        data2D = data2D + A[m] * factor * kv(m, gamma_0 * rR)

    # data2D = 1 / (2 * pi) * data2D;
    data2D = 1.0 / (2.0 * np.pi) * data2D
    return data2D


def displayDataBesselApproach(WavefieldSctCircle, rcvr_phi):
    angle = rcvr_phi * 180 / np.pi
    # Plot data at a number of receivers
    # fig = plt.figure(figsize=(0.39, 0.39), dpi=100)
    plt.plot(angle, abs(WavefieldSctCircle), label='Bessel-function method')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.title('scattered wave data in 2D', fontsize=12)
    plt.axis('tight')
    plt.xlabel('observation angle in degrees')
    plt.xlim([0, 360])
    plt.ylabel('abs(data) $\\rightarrow$')
    plt.show()


def displayDataCSIEApproach(Dop, rcvr_phi):
    angle = rcvr_phi * 180 / np.pi
    # Plot data at a number of receivers
    # fig = plt.figure(figsize=(0.39, 0.39), dpi=100)
    plt.plot(angle, abs(Dop), label='Integral-equation method')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.title('scattered wave data in 2D', fontsize=12)
    plt.axis('tight')
    plt.xlabel('observation angle in degrees')
    plt.xlim([0, 360])
    plt.ylabel('abs(data) $\\rightarrow$')
    plt.show()


def displayDataCompareApproachs(WavefieldSctCircle, Dop, rcvr_phi):
    angle = rcvr_phi * 180 / np.pi
    error = str(np.linalg.norm(Dop - WavefieldSctCircle, ord=1)/np.linalg.norm(WavefieldSctCircle, ord=1))

    # Plot data at a number of receivers
    # fig = plt.figure(figsize=(0.39, 0.39), dpi=100)
    plt.tight_layout()
    plt.plot(angle, np.abs(Dop), '--r', angle, np.abs(WavefieldSctCircle), 'b')
    plt.legend(['Integral-equation method', 'Bessel-function method'], loc='upper center')

    plt.text(0.5*np.max(angle), 0.8*np.max(np.abs(Dop)), 'Error$^sct$ = ' + error, color='red', ha='center', va='center')
    plt.title('scattered wave data in 2D', fontsize=12)
    plt.axis('tight')
    plt.xlabel('observation angle in degrees')
    plt.xlim([0, 360])
    plt.ylabel('abs(data) $\\rightarrow$')
    plt.show()


def u_inc(gamma_0, xS, dx, X1, X2):
    # incident wave on two-dimensional grid
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


def Aw(w, N1, N2, FFTG, CHI):
    # Define the matrix-vector multiplication function Aw
    w = w.reshape((N1, N2))
    y = w - CHI * Kop(w, FFTG)
    # return csc_matrix(y)
    y = y.flatten()
    return y


def b(CHI, u_inc):
    # Known 1D vector right-hand side
    # b = input.CHI(:) .* u_inc(:);
    return CHI.flatten() * u_inc.flatten()


def ITERBiCGSTABw(b, CHI, u_inc, FFTG, N1, N2, Errcri, itmax):
    # BiCGSTAB_FFT scheme for contrast source integral equation Aw = b

    def callback(xk):
        # Define the callback function
        callback.iter_count += 1
        residual_norm = np.linalg.norm(b - Aw_operator(xk))
        # CHECK: Not sure that this time is correct
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


def Kop(w_E, FFTG):
    # Make FFT grid
    Cv = np.zeros(FFTG.shape, dtype=np.complex_)
    N1, N2 = w_E.shape
    Cv[0:N1, 0:N2] = w_E.copy()
    # Convolution by FFT
    Cv = np.fft.fftn(Cv)
    Cv = np.fft.ifftn(FFTG * Cv)
    Kv = Cv[0:N1, 0:N2]
    return Kv


def plotContrastSource(ITERBiCGSTABw, CHI, X1, X2):
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
    im2 = ax2.imshow(abs(ITERBiCGSTABw), extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax2.set_xlabel('x_2 \u2192')
    ax2.set_ylabel('\u2190 x_1')
    ax2.set_aspect('equal', adjustable='box')
    fig.colorbar(im2, ax=ax2, orientation='horizontal')
    ax2.set_title(r'$|w|$', fontsize=13)

    plt.show()


def Dop(ITERBiCGSTABw, gamma_0, dx, xR, NR, X1, X2):
    data = np.zeros(NR, dtype=np.complex_)
    # Radius circle with area of dx^2
    delta = np.pi**(-0.5) * dx

    print("data", data.shape)

    # factor = 2 * besseli(1, gamma_0*delta) / (gamma_0 * delta)
    factor = 2 * iv(1, gamma_0 * delta) / (gamma_0 * delta)

    # for p = 1:input.NR
    for p in range(0, NR):
        # DIS = sqrt((xR(1, p) - X1).^2+(xR(2, p) - X2).^2);
        DIS = np.sqrt((xR[0, p] - X1)**2 + (xR[1, p] - X2)**2)

        # G = 1 / (2 * pi) .* besselk(0, gamma_0*DIS);
        G = 1.0 / (2.0 * np.pi) * kv(0, gamma_0 * DIS)

        # data(1, p) = (gamma_0^2 * dx^2) * factor * sum(G(:).*w(:));
        data[p] = (gamma_0**2 * dx**2) * factor * np.sum(G * ITERBiCGSTABw)
    return data
