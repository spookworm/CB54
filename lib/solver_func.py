import numpy as np
from scipy.special import kv, iv
import time
from scipy.sparse.linalg import bicgstab, LinearOperator


def a(radius: float) -> float:
    """
    This takes the input for cylinder radius in meters and returns it.
    Args:
        input_ (float): The cylinder radius.

    Returns:
        float: The cylinder radius.
    """
    # help(solver_func.a)
    return radius


def angle(rcvr_phi):
    return rcvr_phi * 180 / np.pi


def arg0(gamma_0, a):
    return gamma_0 * a


def args(gam_sct, a):
    return gam_sct * a


def Aw(w, N1, N2, FFTG, CHI):
    # print("w.shape", w.shape)
    # # Define the matrix-vector multiplication function Aw
    w = np.reshape(w, (N1, N2), order='F')
    # print("w.shape reshaped", w.shape)
    y = np.zeros((N1, N2), dtype=np.complex128, order='F')
    # print("y.shape zeros", y.shape)
    y = w - CHI * Kop(w, FFTG)
    # print("y.shape", y.shape)
    y = y.flatten('F')
    # print("y.shape", y.shape)
    return y


def b(CHI, u_inc, N1, N2):
    # Known 1D vector right-hand side
    # b = CHI(:) * u_inc(:)
    b = np.zeros((N1*N2, 1), dtype=np.complex128, order='F')
    # b = np.multiply(CHI.flatten('F'), u_inc.flatten('F'))
    b[:, 0] = CHI.flatten('F') * u_inc.flatten('F')
    # b_diff = mat_checker(b, nameof(b))
    # b_m = mat_loader(nameof(b))
    return b


def CHI(c_0, c_sct, R, a):
    CHI = (1 - c_0**2 / c_sct**2) * (R < a)
    # CHI_diff = mat_checker(CHI, nameof(CHI))
    return CHI


def c_0(input_):
    """
    This takes the input for wave speed in embedding in meters per second and returns it.
    Args:
        input_ (float): wave speed in embedding.

    Returns:
        float: wave speed in embedding.
    """
    # help(solver_func.c_0)
    return input_


def contrast_sct(input_):
    """
    This takes the input for contrast of scatterer and returns it.
    Args:
        input_ (float): contrast of scatterer.

    Returns:
        float: contrast of scatterer.
    """
    # help(solver_func.contrast_sct)
    return input_


def c_sct(c_0, contrast_sct):
    """
    This takes the input for wave speed in embedding and the contrast of the scatterer in meters per second and returns the wave speed in the scatterer.
    Args:
        input_ (float): wave speed in scatterer.

    Returns:
        float: wave speed in scatterer.
    """
    # help(solver_func.c_sct)
    return c_0 * contrast_sct


def data_gen(b, CHI, FFTG, N1, N2, Errcri, itmax, x0):
    time_start = time.time()
    w, exit_code, iterative_info = ITERBiCGSTABw(b, CHI, FFTG, N1, N2, Errcri, itmax, x0)
    time_total = time.time() - time_start
    # Display the convergence information
    print("exit_code:", exit_code)
    # print("iter,\tresvec,\ttime_total")
    # for i, row in enumerate(iterative_info):
    #     print(f"{row[0]}\t{row[1]}\t{row[2]}")
    #     print()
    # relres = iterative_info[-1, 1]/np.linalg.norm(b)
    # print("relres", relres)
    return time_total, w, exit_code, iterative_info


def delta(dx):
    # Radius circle with area of dx^2
    # delta = (pi)^(-1 / 2) * dx;
    return (np.pi)**(-0.5) * dx


def Dop(w_out, gamma_0, dx, xR, NR, X1cap, X2cap, delta, factoru, N1, N2):
    # Dop_diff = mat_checker(Dop, nameof(Dop))
    # Dop_m = mat_loader(nameof(Dop))
    data = np.zeros((1, NR), dtype=np.complex128, order='F')
    G = np.zeros((N1, N2), dtype=np.complex128, order='F')
    for p in range(0, NR):
        DIS = np.sqrt((xR[0, p] - X1cap)**2 + (xR[1, p] - X2cap)**2)
        G = 1.0 / (2.0 * np.pi) * kv(0, gamma_0*DIS)
        data[0, p] = (gamma_0**2 * dx**2) * factoru * np.sum(G.flatten('F') * w_out.flatten('F'))
    return data


def dx(input_):
    # meshsize dx
    return input_


def Errcri(input_):
    """
    This takes the input for the tolerance of the BICGSTAB method.
    Args:
        input_ (float): tolerance

    Returns:
        float: tolerance
    """
    # help(solver_func.Errcri)
    return input_


def f(input_):
    """
    This takes the input for the carrier temporal frequency in Hz and returns the carrier temporal frequency in Hz.
    Args:
        input_ (float): the carrier temporal frequency in Hz.

    Returns:
        float: the carrier temporal frequency in Hz.
    """
    # help(solver_func.f)
    return input_


def factoru(gamma_0, delta):
    # Factor for weak form if DIS > delta
    return 2 * iv(1, gamma_0*delta) / (gamma_0*delta)


def FFTG(IntG):
    # Apply n-dimensional Fast Fourier transform
    # FFTG_diff = mat_checker(FFTG, nameof(FFTG))
    return np.fft.fftn(IntG)


def gamma_0(s, c_0):
    # Propagation Co-efficient
    return s / c_0


def gamma_sct(gamma_0, c_0, c_sct):
    return gamma_0 * c_0 / c_sct


def initFFTGreen(x1fft, x2fft):
    # [temp.X1fft, temp.X2fft] = ndgrid(x1, x2);
    X1fftcap, X2fftcap = np.meshgrid(x1fft, x2fft, indexing='ij')
    # X1fftcap_diff = mat_checker(X1fftcap, nameof(X1fftcap))
    # X2fftcap_diff = mat_checker(X2fftcap, nameof(X2fftcap))
    return np.meshgrid(x1fft, x2fft, indexing='ij')


def initGrid(N1, N2, dx):
    # Grid in two-dimensional space
    # x1 = -(N1 + 1) * dx / 2 + (1:N1) * dx
    x1 = np.zeros((1, N1), dtype=float, order='F')
    x1[0, :] = -(N1 + 1) * dx / 2 + np.arange(1, N1+1, 1) * dx
    # x1_diff = mat_checker(x1, nameof(x1))

    # x2 = -(N2 + 1) * dx / 2 + (1:N2) * dx
    x2 = np.zeros((1, N2), dtype=float, order='F')
    x2[0, :] = -(N2 + 1) * dx / 2 + np.arange(1, N2+1, 1) * dx
    # x2_diff = mat_checker(x2, nameof(x2))

    # [X1cap, X2cap] = ndgrid(x1, x2);
    X1cap, X2cap = np.meshgrid(x1, x2, indexing='ij')

    # X1cap_diff = mat_checker(X1cap, nameof(X1cap))
    # X2cap_diff = mat_checker(X2cap, nameof(X2cap))
    return X1cap, X2cap


def IntG(dx, gamma_0, X1fftcap, X2fftcap, N1, N2, delta):
    # Compute gam_0^2 * subdomain integrals of Green function
    DIS = np.sqrt(X1fftcap**2 + X2fftcap**2)
    # Avoid Green's singularity for DIS = 0
    DIS[0, 0] = 1.0
    # DIS_diff = mat_checker(DIS, nameof(DIS))

    # G = 1 / (2 * pi) .* besselk(0, gamma_0*DIS);
    G = np.zeros((N1, N2), dtype=np.complex128, order='F')
    G = 1 / (2 * np.pi) * kv(0, gamma_0*DIS)
    # G_diff = mat_checker(G, nameof(G))

    # factor = 2 * besseli(1, gamma_0*delta) / (gamma_0 * delta)
    factor = 2 * iv(1, gamma_0 * delta) / (gamma_0 * delta)

    # Integral includes gamma_0**2
    # IntG = (gamma_0^2 * dx^2) * factor^2 * G;
    IntG = (gamma_0**2 * dx**2) * factor**2 * G

    # IntG(1, 1) = 1 - gamma_0 * delta * besselk(1, gamma_0*delta) * factor
    IntG[0, 0] = 1 - gamma_0*delta * kv(1, gamma_0*delta) * factor
    # IntG_diff = mat_checker(IntG, nameof(IntG))
    return IntG


def ITERBiCGSTABw(b, CHI, FFTG, N1, N2, Errcri, itmax, x0):
    # BiCGSTAB_FFT scheme for contrast source integral equation Aw = b

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


def itmax(input_):
    return input_


def Kop(w_E, FFTG):
    # Make FFT grid
    N1, N2 = w_E.shape
    Cv = np.zeros(FFTG.shape, dtype=np.complex128, order='F')
    Cv[0:N1, 0:N2] = w_E.copy()
    # Convolution by FFT
    Cv = np.fft.fftn(Cv)
    Cv = np.fft.ifftn(FFTG * Cv)
    Kv = np.zeros((N1, N2), dtype=np.complex128, order='F')
    Kv[0:N1, 0:N2] = Cv[0:N1, 0:N2]
    return Kv


def M(input_):
    """
    Increase M for more accuracy
    """
    return input_


def N1(input_):
    # number of samples in x_1
    return input_


def N2(input_):
    # number of samples in x_2
    return input_


def NR(input_):
    # help(solver_func.NR)
    return input_


def phiR(xR):
    return np.arctan2(xR[1, :], xR[0, :])


def phiS(xS):
    return np.arctan2(xS[1], xS[0])


def R(X1cap, X2cap):
    R = np.sqrt(X1cap**2 + X2cap**2)
    # R_diff = mat_checker(R, nameof(R))
    return R


def rcvr_phi(NR):
    rcvr_phi = np.zeros((1, NR), dtype=np.float64, order='F')
    # rcvr_phi(1:input.NR) = (1:input.NR) * 2 * pi / input.NR;
    rcvr_phi[0, 0:NR] = np.arange(1, NR+1, 1) * 2.0 * np.pi / NR
    # mat_checker(rcvr_phi, nameof(rcvr_phi))
    return rcvr_phi


def rR(xR):
    return np.sqrt(xR[0, :]**2 + xR[1, :]**2)


def rS(xS):
    return np.sqrt(xS[0]**2 + xS[1]**2)


def s(f):
    # Laplace Parameter
    return 1e-16 - 1j * 2 * np.pi * f


def u_inc(gamma_0, xS, X1cap, X2cap, factoru):
    # incident wave on two-dimensional grid
    # DIS = sqrt((X1 - xS(1)).^2+(X2 - xS(2)).^2);
    DISu = np.sqrt((X1cap - xS[0])**2 + (X2cap - xS[1])**2)
    # DISu_diff = mat_checker(DISu, nameof(DISu))
    # G = 1 / (2 * pi) .* besselk(0, gam0*DIS);
    Gu = 1 / (2 * np.pi) * kv(0, gamma_0*DISu)
    # Gu_diff = mat_checker(Gu, nameof(Gu))
    u_inc = factoru * Gu
    # u_inc_diff = mat_checker(u_inc, nameof(u_inc))
    return u_inc


def WavefieldSctCircle(M, arg0, args, gamma_sct, gamma_0, xR, xS, rR, phiR, rS, phiS):
    # Compute coefficients of series expansion
    A = np.zeros((1, M+1), dtype=np.complex128)
    for m in range(0, M+1):
        Ib0 = iv(m, arg0)
        dIb0 = iv(m+1, arg0) + m / arg0 * Ib0
        Ibs = iv(m, args)
        dIbs = iv(m+1, args) + m / args * Ibs
        Kb0 = kv(m, arg0)
        dKb0 = -kv(m+1, arg0) + m / arg0 * Kb0

        A[0, m] = -(gamma_sct * dIbs * Ib0 - gamma_0 * dIb0 * Ibs) / (gamma_sct * dIbs * Kb0 - gamma_0 * dKb0 * Ibs)

    # data2D = A(1) * besselk(0, gam0*rS) .* besselk(0, gam0*rR);
    data2D = A[0, 0] * kv(0, gamma_0*rS) * kv(0, gamma_0*rR)
    for m in range(1, M+1):
        # factor = 2 * besselk(m, gam0*rS) .* cos(m*(phiS - phiR));
        factor = 2 * kv(m, gamma_0*rS) * np.cos(m*(phiS - phiR))
        # data2D = data2D + A(m+1) * factor .* besselk(m, gamma_0*rR);
        data2D = data2D + A[0, m] * factor * kv(m, gamma_0*rR)
    # data2D = 1 / (2 * pi) * data2D;
    data2D = 1 / (2 * np.pi) * data2D
    # data2D_diff = mat_checker(data2D, nameof(data2D))
    return data2D


def wavelength(c_0, f):
    # wavelength
    return c_0 / f


def x0_naive(N1, N2):
    # Initial Guess
    x0 = np.zeros((N1*N2, 1), dtype=np.complex128, order='F')
    return x0


def X1cap(initGrid):
    return initGrid[0]


def X1fft(initFFTGreen):
    return initFFTGreen[0].T


def x1fft(N1, dx):
    # Compute FFT of Green function
    # N1fft = 2^ceil(log2(2*input.N1));
    N1fft = int(2**np.ceil(np.log2(2*N1)))
    # x1(1:N1fft) = [0 : N1fft / 2 - 1, N1fft / 2 : -1 : 1] * input.dx;
    x1fft = np.concatenate((np.arange(0, N1fft/2), np.arange(N1fft/2, 0, -1))) * dx
    x1fft = np.reshape(x1fft, (1, N1fft), order='F')
    # x1fft_diff = mat_checker(x1fft, nameof(x1fft))
    return x1fft


def X2cap(initGrid):
    return initGrid[1]


def X2fft(initFFTGreen):
    return initFFTGreen[1].T


def x2fft(N2, dx):
    # Compute FFT of Green function
    # N2fft = 2^ceil(log2(2*input.N2));
    N2fft = int(2**np.ceil(np.log2(2*N2)))
    # x2(1:N2fft) = [0 : N2fft / 2 - 1, N2fft / 2 : -1 : 1] * input.dx;
    x2fft = np.concatenate((np.arange(0, N2fft/2), np.arange(N2fft/2, 0, -1))) * dx
    x2fft = np.reshape(x2fft, (1, N2fft), order='F')
    # x2fft_diff = mat_checker(x2fft, nameof(x2fft))
    return x2fft


def xR(NR, rcvr_phi):
    # Receiver Positions
    xR = np.zeros((2, NR), dtype=float, order='F')
    # input.xR(1, 1:input.NR) = 150 * cos(input.rcvr_phi);
    xR[0, 0:NR] = 150 * np.cos(rcvr_phi)
    # input.xR(2, 1:input.NR) = 150 * sin(input.rcvr_phi);
    xR[1, 0:NR] = 150 * np.sin(rcvr_phi)
    # xR_diff = mat_checker(xR, nameof(xR))
    return xR


def xS(x, y):
    # Source Position
    xS = np.zeros((1, 2), dtype=float)
    # xS = [-170, 0]
    xS = [x, y]
    return xS
