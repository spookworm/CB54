import numpy as np
import time
from scipy.special import kv, iv
from scipy.sparse.linalg import bicgstab, LinearOperator


def a(input_):
    # Radius Cylinder
    return input_


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
    y = np.zeros((N1, N2), dtype=np.complex_, order='F')
    # print("y.shape zeros", y.shape)
    y = w - CHI * Kop(w, FFTG)
    # print("y.shape", y.shape)
    y = y.flatten('F')
    # print("y.shape", y.shape)
    return y


def b(CHI, u_inc, N1, N2):
    # Known 1D vector right-hand side
    # b = CHI(:) * u_inc(:)
    b = np.zeros((N1*N2, 1), dtype=np.complex_, order='F')
    # b = np.multiply(CHI.flatten('F'), u_inc.flatten('F'))
    b[:, 0] = CHI.flatten('F') * u_inc.flatten('F')
    # b_diff = mat_checker(b, nameof(b))
    # b_m = mat_loader(nameof(b))
    return b


def b_E(CHI_eps, E_inc, N):
    # Known 1D vector right-hand side
    b = np.zeros((2*N, 1), dtype=np.complex_, order='F')
    # b(1:N, 1) = input.CHI_eps(:) .* E_inc{1}(:);
    # b[0:N, 0] = CHI_eps.flatten('F') * E_inc[0, :, :].flatten('F')
    b[0:N, 0] = CHI_eps.flatten('F') * E_inc[0, :, :].flatten('F')
    # b(N+1:2*N, 1) = input.CHI_eps(:) .* E_inc{2}(:);
    # b[N, 2*N, 0] = CHI_eps.flatten('F') * E_inc[1, :, :].flatten('F')
    b[N:2*N, 0] = CHI_eps.flatten('F') * E_inc[1, :, :].flatten('F')
    # b_diff = mat_checker(b, nameof(b))
    # # b_m = mat_loader(nameof(b))
    return b


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


def c_0(input_):
    # wave speed in embedding
    return input_


def c_sct(input_):
    # wave speed in scatterer
    return input_


def data_load(path, filename):
    # data_load = ForwardBiCGSTABFFT.data_load('', 'data2D.txt')
    # load DATA2D data2D;
    return np.loadtxt(path + filename).view(np.float_)


def data_save(path, filename, data2D):
    # ForwardBiCGSTABFFT.data_save('', 'data2D', WavefieldSctCircle)
    # save DATA2D data2D;
    np.savetxt(path + filename + '.txt', data2D.view(np.complex_))


def delta(dx):
    # Radius circle with area of dx^2
    # delta = (pi)^(-1 / 2) * dx;
    return (np.pi)**(-0.5) * dx


def Dop(w_out, gamma_0, dx, xR, NR, X1cap, X2cap, delta, factoru, N1, N2):
    # Dop_diff = mat_checker(Dop, nameof(Dop))
    # Dop_m = mat_loader(nameof(Dop))
    data = np.zeros((1, NR), dtype=np.complex_, order='F')
    G = np.zeros((N1, N2), dtype=np.complex_, order='F')
    for p in range(0, NR):
        DIS = np.sqrt((xR[0, p] - X1cap)**2 + (xR[1, p] - X2cap)**2)
        G = 1.0 / (2.0 * np.pi) * kv(0, gamma_0*DIS)
        data[0, p] = (gamma_0**2 * dx**2) * factoru * np.sum(G.flatten('F') * w_out.flatten('F'))
    return data


def DOPwE(w_E, gamma_0, dx, xR, NR, X1cap, X2cap, delta, factoru):
    Edata = np.zeros((1, NR), dtype=np.complex_, order='F')
    Hdata = np.zeros((1, NR), dtype=np.complex_, order='F')
    for p in range(1, NR+1):
        X1_ = xR[0, p-1] - X1cap
        X2_ = xR[1, p-1] - X1cap

        DIS = np.sqrt(X1_**2 + X2_**2)
        X1 = X1_ / DIS
        X2 = X2_ / DIS

        G = factoru * 1.0 / (2.0 * np.pi) * kv(0, gamma_0 * DIS)
        dG = -factoru * gamma_0 * 1.0 / (2.0 * np.pi) * kv(1, gamma_0 * DIS)
        d1_G = X1 * dG
        d2_G = X2 * dG

        dG11 = (2.0 * X1 * X1 - 1.0) * (-dG / DIS) + gamma_0**2 * X1 * X1 * G
        dG22 = (2.0 * X2 * X2 - 1.0) * (-dG / DIS) + gamma_0**2 * X2 * X2 * G
        dG21 = (2.0 * X2 * X1      ) * (-dG / DIS) + gamma_0**2 * X2 * X1 * G

        # E1rfl = dx^2 * sum((gam0^2 * G(:) - dG11(:)).*w_E{1}(:) - dG21(:).*w_E{2}(:));
        E1rfl = dx**2 * np.sum(gamma_0**2 * G.flatten('F') - dG11.flatten('F') * w_E[0].flatten('F') - dG21.flatten('F') * w_E[1].flatten('F'))
        E2rfl = dx**2 * np.sum(-dG21.flatten('F') * w_E[0].flatten('F') + (gamma_0**2 * G.flatten('F') - dG22.flatten('F')) * w_E[1].flatten('F'))
        ZH3rfl = gamma_0 * dx**2 * np.sum(d2_G.flatten('F') * w_E[0].flatten('F') - d1_G.flatten('F') * w_E[1].flatten('F'))

        Edata[0, p-1] = np.sqrt(np.abs(E1rfl)**2 + np.abs(E2rfl)**2)
        Hdata[0, p-1] = np.abs(ZH3rfl)
    return Edata, Hdata


def dx(input_):
    # meshsize dx
    return input_


def Edata2D(EMsctCircle):
    return EMsctCircle[0]


def EMsctCircle(c_0, eps_sct, mu_sct, gamma_0, xR, xS, M, a, gamma_sct, Z_sct, arg0, args, rR, phiR, rS, phiS):
    # Compute coefficients of series expansion
    A = np.zeros((1, M+1), dtype=np.complex_)
    for m in range(1, M+1):
        Ib0 = iv(m, arg0)
        dIb0 = iv(m+1, arg0) + m / arg0 * Ib0
        Ibs = iv(m, args)
        dIbs = iv(m+1, args) + m / args * Ibs
        Kb0 = kv(m, arg0)
        dKb0 = -kv(m+1, arg0) + m / arg0 * Kb0
        denominator = Z_sct * dIbs * Kb0 - dKb0 * Ibs
        A[0, m-1] = -(Z_sct * dIbs * Ib0 - dIb0 * Ibs) / denominator

    Er = np.zeros(rR.shape, dtype=np.complex_)
    Ephi = np.zeros(rR.shape, dtype=np.complex_)
    ZH3 = np.zeros(rR.shape, dtype=np.complex_)

    for m in range(1, M+1):
        arg0 = gamma_0 * rR
        Kb0 = kv(m, arg0)
        dKb0 = -kv(m+1, arg0) + m / arg0 * Kb0
        KbS = kv(m, gamma_0 * rS)
        Er = Er + A[0, m-1] * 2 * m**2 * Kb0 * KbS * np.cos(m*(phiR - phiS))
        Ephi = Ephi - A[0, m-1] * 2 * m * dKb0 * KbS * np.sin(m*(phiR - phiS))
        ZH3 = ZH3 - A[0, m-1] * 2 * m * Kb0 * KbS * np.sin(m*(phiR - phiS))

    Er = 1.0 / (2.0 * np.pi) * Er / rR / rS
    Ephi = gamma_0 * 1.0 / (2.0 * np.pi) * Ephi / rS
    ZH3 = -gamma_0 * 1.0 / (2.0 * np.pi) * ZH3 / rS

    E = {}
    E[1] = np.cos(phiR) * Er - np.sin(phiR) * Ephi
    E[2] = np.sin(phiR) * Er + np.cos(phiR) * Ephi

    Edata2D = np.zeros((1, E[1].size), dtype=np.complex_)
    Edata2D = np.sqrt(np.abs(E[1])**2 + np.abs(E[2])**2)
    Hdata2D = np.abs(ZH3)
    return Edata2D, Hdata2D


def eps_sct(input_):
    # relative permittivity of scatterer
    return input_


def Errcri(input_):
    return input_


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


def E_sct(w_E, FFTG, gamma_0, dx, N1, N2):
    return KwE(w_E, FFTG, gamma_0, dx, N1, N2)


def f(input_):
    # temporal frequency
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


def Hdata2D(EMsctCircle):
    return EMsctCircle[1]


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


def ITERBiCGSTABw(b, CHI, FFTG, N1, N2, Errcri, itmax, x0):
    # BiCGSTAB_FFT scheme for contrast source integral equation Aw = b
    norm_b = np.linalg.norm(b)

    def callback(xk):
        # Define the callback function
        # relative residual norm(b-A*x)/norm(b)
        callback.iter_count += 1
        # residual = np.linalg.norm(b - Aw_operator.dot(xk))/norm_b
        # residual_norm = np.linalg.norm(b - Aw_operator(xk))
        residual_norm = norm_b
        # residuals.append(residual)
        # CHECK: Not sure that this time is correct
        callback.time_total = time.time() - callback.start_time
        # print("Current solution:", xk)
        # print("Iteration:", callback.iter_count, "Residual norm:", residual_norm, "Time:", time.time() - callback.start_time)
        if residual_norm < Errcri:
            return True
        else:
            return False
        print(callback.iter_count, "\t", residual_norm, "\t", callback.time_total)
        print(residual_norm)

    # Initialise iteration count
    callback.iter_count = 0
    callback.start_time = time.time()

    # Call bicgstab with the LinearOperator instance and other inputs
    # w = bicgstab(@(w) Aw(w, input), b, Errcri, itmax);

    # Aw_operator = LinearOperator((b.shape[0], b.shape[0]), matvec=lambda w: Aw(w, N1, N2, FFTG, CHI))
    def custom_matvec(w):
        return Aw(w, N1, N2, FFTG, CHI)

    Aw_operator = LinearOperator((b.shape[0], b.shape[0]), matvec=custom_matvec)

    start_time = time.time()
    w, exit_code = bicgstab(Aw_operator, b, x0=x0, tol=Errcri, maxiter=itmax, callback=callback)
    time_total = time.time() - start_time
    print("time_total", time_total)

    # Output Matrix
    # w = vector2matrix(w, N1, N2)
    w = w.reshape((N1, N2), order='F')
    # # Display the convergence information
    # print("Convergence information:", exit_code)
    # print(exit_code)
    print("Iteration:", callback.iter_count)
    # print("time_total", callback.time_total)
    return w


def itmax(input_):
    return input_


def Kop(w_E, FFTG):
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


def KwE(w_E, FFTG, gamma_0, dx, N1, N2):
    return KopE(w_E, FFTG, gamma_0, dx, N1, N2)


def M(input_):
    # Increase M for more accuracy
    return input_


def mu_sct(input_):
    # relative permeability of scatterer
    return input_


def N(CHI_eps):
    # [N, ~] = size(input.CHI_eps(:));
    N = CHI_eps.flatten('F').size
    return N


def N1(input_):
    # number of samples in x_1
    return input_


def N2(input_):
    # number of samples in x_2
    return input_


def NR(input_):
    # Reciever Count
    return input_


def phi():
    # phi = 0:.01:2 * pi;
    return np.arange(0, 2.0*np.pi, 0.01)


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


def vector2matrix(w, N1, N2):
    return w.reshape((N1, N2), order='F')


def WavefieldSctCircle(M, arg0, args, gamma_sct, gamma_0, xR, xS, rR, phiR, rS, phiS):
    # Compute coefficients of series expansion
    A = np.zeros((1, M+1), dtype=np.complex_)
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


def w_E(w, N1, N2, N):
    # Output Matrix
    return vector2matrix(w, N1, N2, N)


def x0(b):
    # Initial Guess
    x0 = np.zeros(b.shape, dtype=np.complex_)
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


def xS():
    # Source Position
    xS = np.zeros((1, 2), dtype=float)
    xS = [-170, 0]
    return xS


def ZH_inc(EZH_inc):
    return EZH_inc[1]
