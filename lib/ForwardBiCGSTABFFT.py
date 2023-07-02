import numpy as np
import time
import scipy.io
from scipy.special import kv, iv
from scipy.sparse.linalg import bicgstab, LinearOperator
import matplotlib.pyplot as plt


def mat_loader(var_name):
    # x1fft_m = mat_loader(nameof(x1fft))
    var_name_matlab = scipy.io.loadmat(var_name + '.mat')
    var_name_m = np.array(var_name_matlab[var_name])
    return var_name_m


def mat_checker(variable, var_name):
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


def c_0(input_):
    # wave speed in embedding
    return input_


def c_sct(input_):
    # wave speed in scatterer
    return input_


def eps_sct(input_):
    # relative permittivity of scatterer
    return input_


def mu_sct(input_):
    # relative permeability of scatterer
    return input_


def f(input_):
    # temporal frequency
    return input_


def s(f):
    # Laplace Parameter
    return 1e-16 - 1j * 2 * np.pi * f


def gamma_0(s, c_0):
    # Propagation Co-efficient
    return s / c_0


def wavelength(c_0, f):
    # wavelength
    return c_0 / f


def Errcri(input_):
    return input_


def xS():
    # Source Position
    xS = np.zeros((2, 1), dtype=float)
    xS = [-170, 0]
    return xS


def NR(input_):
    # Reciever Count
    return input_


def rcvr_phi(NR):
    rcvr_phi = np.zeros((1, NR), dtype=np.float64, order='F')
    # rcvr_phi(1:input.NR) = (1:input.NR) * 2 * pi / input.NR;
    rcvr_phi[0, 0:NR] = np.arange(1, NR+1, 1) * 2.0 * np.pi / NR
    # mat_checker(rcvr_phi, nameof(rcvr_phi))
    return rcvr_phi


def xR(NR, rcvr_phi):
    # Receiver Positions
    xR = np.zeros((2, NR), dtype=float, order='F')
    # input.xR(1, 1:input.NR) = 150 * cos(input.rcvr_phi);
    xR[0, 0:NR] = 150 * np.cos(rcvr_phi)
    # input.xR(2, 1:input.NR) = 150 * sin(input.rcvr_phi);
    xR[1, 0:NR] = 150 * np.sin(rcvr_phi)
    # xR_diff = mat_checker(xR, nameof(xR))
    return xR


def N1(input_):
    # number of samples in x_1
    return input_


def N2(input_):
    # number of samples in x_2
    return input_


def dx(input_):
    # meshsize dx
    return input_


def a(input_):
    # Radius Cylinder
    return input_


def M(input_):
    # Increase M for more accuracy
    return input_


def itmax(input_):
    return input_


def delta(dx):
    # Radius circle with area of dx^2
    # delta = (pi)^(-1 / 2) * dx;
    return (np.pi)**(-0.5) * dx


def R(X1cap, X2cap):
    R = np.sqrt(X1cap**2 + X2cap**2)
    # R_diff = mat_checker(R, nameof(R))
    return R


def CHI(c_0, c_sct, R, a):
    CHI = (1 - c_0**2 / c_sct**2) * (R < a)
    # CHI_diff = mat_checker(CHI, nameof(CHI))
    return CHI


def CHI_eps(eps_sct, a, R):
    # Compute permittivity contrast
    return (1.0 - eps_sct) * (R < a)


def CHI_mu(mu_sct, a, R):
    # Compute permeability contrast
    return (1.0 - mu_sct) * (R < a)


def b(CHI, u_inc, N1, N2):
    # Known 1D vector right-hand side
    # b = CHI(:) * u_inc(:)
    b = np.zeros((N1*N2, 1), dtype=np.complex_, order='F')
    # b = np.multiply(CHI.flatten('F'), u_inc.flatten('F'))
    b[:, 0] = CHI.flatten('F') * u_inc.flatten('F')
    # b_diff = mat_checker(b, nameof(b))
    # b_m = mat_loader(nameof(b))
    return b


def N(CHI_eps):
    # [N, ~] = size(input.CHI_eps(:));
    N = CHI_eps.flatten('F').size
    return N


def b_E(CHI_eps, EM_inc, N1, N2, N):
    # Known 1D vector right-hand side
    b = np.zeros((2*N, 1), dtype=np.complex_, order='F')
    # b(1:N, 1) = input.CHI_eps(:) .* E_inc{1}(:);
    # b[0:N, 0] = CHI_eps.flatten('F') * EM_inc[0, :, :].flatten('F')
    b[0:N, 0] = CHI_eps.flatten('F') * EM_inc[0, :, :].flatten('F')
    # b(N+1:2*N, 1) = input.CHI_eps(:) .* E_inc{2}(:);
    # b[N, 2*N, 0] = CHI_eps.flatten('F') * EM_inc[1, :, :].flatten('F')
    b[N:2*N, 0] = CHI_eps.flatten('F') * EM_inc[1, :, :].flatten('F')
    # b_diff = mat_checker(b, nameof(b))
    # # b_m = mat_loader(nameof(b))
    return b


def vector2matrix(w, N1, N2):
    return w.reshape((N1, N2), order='F')


def angle(rcvr_phi):
    return rcvr_phi * 180 / np.pi


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


def x1fft(N1, dx):
    # Compute FFT of Green function
    # N1fft = 2^ceil(log2(2*input.N1));
    N1fft = int(2**np.ceil(np.log2(2*N1)))
    # x1(1:N1fft) = [0 : N1fft / 2 - 1, N1fft / 2 : -1 : 1] * input.dx;
    x1fft = np.concatenate((np.arange(0, N1fft/2), np.arange(N1fft/2, 0, -1))) * dx
    x1fft = np.reshape(x1fft, (1, N1fft), order='F')
    # x1fft_diff = mat_checker(x1fft, nameof(x1fft))
    return x1fft


def x2fft(N2, dx):
    # Compute FFT of Green function
    # N2fft = 2^ceil(log2(2*input.N2));
    N2fft = int(2**np.ceil(np.log2(2*N2)))
    # x2(1:N2fft) = [0 : N2fft / 2 - 1, N2fft / 2 : -1 : 1] * input.dx;
    x2fft = np.concatenate((np.arange(0, N2fft/2), np.arange(N2fft/2, 0, -1))) * dx
    x2fft = np.reshape(x2fft, (1, N2fft), order='F')
    # x2fft_diff = mat_checker(x2fft, nameof(x2fft))
    return x2fft


def IntG(dx, gamma_0, X1fftcap, X2fftcap, N1, N2, delta):
    # Compute gam_0^2 * subdomain integrals of Green function
    DIS = np.sqrt(X1fftcap**2 + X2fftcap**2)
    # Avoid Green's singularity for DIS = 0
    DIS[0, 0] = 1.0
    # DIS_diff = mat_checker(DIS, nameof(DIS))

    # G = 1 / (2 * pi) .* besselk(0, gamma_0*DIS);
    G = np.zeros((N1, N2), dtype=np.complex_, order='F')
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


def FFTG(IntG):
    # Apply n-dimensional Fast Fourier transform
    # FFTG_diff = mat_checker(FFTG, nameof(FFTG))
    return np.fft.fftn(IntG)


def displayDataBesselApproach(WavefieldSctCircle, angle):
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


def factoru(gamma_0, delta):
    # Factor for weak form if DIS > delta
    return 2 * iv(1, gamma_0*delta) / (gamma_0*delta)


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


def EZH_inc(gamma_0, xS, delta, X1cap, X2cap, factoru, N1, N2):
    # Compute Incident field
    # incident wave from electric dipole in negative x_1
    X1cap = X1cap - xS[0]
    X2cap = X2cap - xS[1]
    DIS = np.sqrt(X1cap**2 + X2cap**2)
    X1cap = X1cap / DIS
    X2cap = X2cap / DIS

    # G = factor * 1 / (2 * pi) .* besselk(0, gam0*DIS);
    G = factoru * 1.0 / (2.0 * np.pi) * kv(0, gamma_0*DIS)
    # dG = -factor * gam0 .* 1 / (2 * pi) .* besselk(1, gam0*DIS);
    dG = -factoru * gamma_0 * 1.0 / (2.0 * np.pi) * kv(1, gamma_0 * DIS)
    # dG11 = (2 * X1 .* X1 - 1) .* (-dG ./ DIS) + gam0^2 * X1 .* X1 .* G;
    dG11 = (2.0 * X1cap * X1cap - 1.0) * (-dG / DIS) + gamma_0**2 * X1cap * X1cap * G
    # dG21 = 2 * X2 .* X1 .* (-dG ./ DIS) + gam0^2 * X2 .* X1 .* G;
    dG21 = 2.0 * X2cap * X1cap * (-dG / DIS) + gamma_0**2 * X2cap * X1cap * G

    E_inc = np.zeros((3, N1, N2), dtype=np.complex_, order='F')
    # E_inc{1} = -(-gam0^2 * G + dG11);
    E_inc[0, :, :] = -(-gamma_0**2 * G + dG11)
    # E_inc{2} = -dG21;
    E_inc[1, :, :] = -dG21
    # E_inc{3} = 0;
    E_inc[2, :, :] = np.zeros((1, N1, N2), dtype=np.complex_, order='F')

    ZH_inc = np.zeros((3, N1, N2), dtype=np.complex_, order='F')
    # ZH_inc{1} = 0
    ZH_inc[0, :, :] = np.zeros((1, N1, N2), dtype=np.complex_, order='F')
    # ZH_inc{2} = 1
    ZH_inc[1, :, :] = np.zeros((1, N1, N2), dtype=np.complex_, order='F')
    # ZH_inc{3} = gam0 * X2 .* dG;
    ZH_inc[2, :, :] = gamma_0 * X2cap * dG
    return E_inc, ZH_inc


def E_inc(EZH_inc):
    return EZH_inc[0]


def ZH_inc(EZH_inc):
    return EZH_inc[1]


def x0(b):
    # Initial Guess
    x0 = np.zeros(b.shape, dtype=np.complex_)
    return x0


def ITERBiCGSTABw(b, CHI, u_inc, FFTG, N1, N2, Errcri, itmax, x0):
    # BiCGSTAB_FFT scheme for contrast source integral equation Aw = b
    norm_b = np.linalg.norm(b)

    def callback(xk):
        # print("Current solution:", xk)
        # relative residual norm(b-A*x)/norm(b)
        callback.iter_count += 1
        # residual = np.linalg.norm(b - Aw_operator.dot(xk))/norm_b
        residual = norm_b
        # residuals.append(residual)
        if residual < Errcri:
            return True
        else:
            return False
        print("Iteration:", "\t", "Residual norm:", "\t", "Time:")
        print(residual)

    # Initialise iteration count
    callback.iter_count = 0
    callback.start_time = time.time()

    # Call bicgstab with the LinearOperator instance and other inputs
    # w = bicgstab(@(w) Aw(w, input), b, Errcri, itmax);
    Aw_operator = LinearOperator((b.shape[0], b.shape[0]), matvec=lambda w: Aw(w, N1, N2, FFTG, CHI))
    start_time = time.time()
    w, exit_code = bicgstab(Aw_operator, b, x0=x0, tol=Errcri, maxiter=itmax, callback=callback)
    time_total = time.time() - start_time
    print("time_total", time_total)

    # Output Matrix
    w = vector2matrix(w, N1, N2)

    # # Display the convergence information
    # print("Convergence information:", exit_code)
    # print(exit_code)
    print("Iteration:", callback.iter_count)
    # print("time_total", callback.time_total)
    return w


def Aw(w, N1, N2, FFTG, CHI):
    # print("w.shape", w.shape)
    # # Define the matrix-vector multiplication function Aw
    w = np.reshape(w, (N1, N2), order='F')
    # print("w.shape reshaped", w.shape)
    y = np.zeros((N1, N2), dtype=np.complex_, order='F')
    # print("y.shape zeros", y.shape)
    y = w - CHI * Kop(w, FFTG)
    # print("y.shape", y.shape)
    y = y.flatten('F')
    # print("y.shape", y.shape)
    return y


def Kop(w_E, FFTG):
    # print("w_E.shape", w_E.shape)
    # Make FFT grid
    N1, N2 = w_E.shape
    Cv = np.zeros(FFTG.shape, dtype=np.complex_, order='F')
    Cv[0:N1, 0:N2] = w_E.copy()
    # Convolution by FFT
    Cv = np.fft.fftn(Cv)
    Cv = np.fft.ifftn(FFTG * Cv)
    Kv = np.zeros((N1, N2), dtype=np.complex_, order='F')
    Kv[0:N1, 0:N2] = Cv[0:N1, 0:N2]
    return Kv


def Dop(w_out, gamma_0, dx, xR, NR, X1cap, X2cap, delta, factoru, N1, N2):
    data = np.zeros((1, NR), dtype=np.complex_, order='F')
    G = np.zeros((N1, N2), dtype=np.complex_, order='F')
    for p in range(0, NR):
        DIS = np.sqrt((xR[0, p] - X1cap)**2 + (xR[1, p] - X2cap)**2)
        G = 1.0 / (2.0 * np.pi) * kv(0, gamma_0*DIS)
        data[0, p] = (gamma_0**2 * dx**2) * factoru * np.sum(G.flatten('F') * w_out.flatten('F'))
    # Dop_diff = mat_checker(Dop, nameof(Dop))
    # Dop_m = mat_loader(nameof(Dop))
    return data


def displayDataCSIEApproach(Dop, angle):
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


def displayDataCompareApproachs(WavefieldSctCircle, Dop, angle):
    error = str(100.00*np.linalg.norm(Dop - WavefieldSctCircle, ord=1)/np.linalg.norm(WavefieldSctCircle, ord=1))

    # Plot data at a number of receivers
    # fig = plt.figure(figsize=(0.39, 0.39), dpi=100)
    plt.tight_layout()
    plt.plot(angle.T, np.abs(Dop).T, '--r', angle.T, np.abs(WavefieldSctCircle).T, 'b')
    plt.legend(['Integral-equation method', 'Bessel-function method'], loc='upper center')

    plt.text(0.5*np.max(angle), 0.8*np.max(np.abs(Dop)), 'Error$^{sct}$ = ' + error, color='red', ha='center', va='center')
    plt.title('scattered wave data in 2D', fontsize=12)
    plt.axis('tight')
    plt.xlabel('observation angle in degrees')
    plt.xlim([0, 360])
    plt.ylabel('abs(data) $\\rightarrow$')
    plt.show()


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


def initFFTGreen(x1fft, x2fft):
    # [temp.X1fft, temp.X2fft] = ndgrid(x1, x2);
    X1fftcap, X2fftcap = np.meshgrid(x1fft, x2fft, indexing='ij')
    # X1fftcap_diff = mat_checker(X1fftcap, nameof(X1fftcap))
    # X2fftcap_diff = mat_checker(X2fftcap, nameof(X2fftcap))
    return np.meshgrid(x1fft, x2fft, indexing='ij')


def X1fft(initFFTGreen):
    return initFFTGreen[0].T


def X2fft(initFFTGreen):
    return initFFTGreen[1].T


def X1cap(initGrid):
    return initGrid[0]


def X2cap(initGrid):
    return initGrid[1]


def WavefieldSctCircle(M, arg0, args, gam_sct, gamma_0, xR, xS):
    A = np.zeros((1, M+1), dtype=np.complex_)
    # Compute coefficients of series expansion
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
    rR = np.sqrt(xR[0, :]**2 + xR[1, :]**2)
    # phiR = atan2(xR(2, :), xR(1, :))
    phiR = np.arctan2(xR[1, :], xR[0, :])
    # rS = sqrt(xS(1)^2+xS(2)^2)
    rS = np.sqrt(xS[0]**2 + xS[1]**2)
    # phiS = atan2(xS(2), xS(1))
    phiS = np.arctan2(xS[1], xS[0])
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


def gam_sct(gamma_0, c_0, c_sct):
    return gamma_0 * c_0 / c_sct


def arg0(gamma_0, a):
    return gamma_0 * a


def args(gam_sct, a):
    return gam_sct * a
