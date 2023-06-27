import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv, iv
from scipy.sparse.linalg import bicgstab, LinearOperator
import time


def c_0():
    # wave speed in embedding
    return 3e8


def eps_sct():
    # relative permittivity of scatterer
    return 1.75


def mu_sct():
    # relative permeability of scatterer
    return 1.0


def f():
    # temporal frequency
    return 10e6


def CHI_eps(eps_sct, a, R):
    # Compute permittivity contrast
    return (1.0 - eps_sct) * (R < a)


def CHI_mu(mu_sct, a, R):
    # Compute permeability contrast
    return (1.0 - mu_sct) * (R < a)


def Edata2D(EMsctCircle):
    return EMsctCircle[0]


def Hdata2D(EMsctCircle):
    return EMsctCircle[1]


def M():
    # Increase M for more accuracy
    return 20


def EMsctCircle(c_0, eps_sct, mu_sct, gamma_0, xR, xS, M, a):

    gam_sct = gamma_0 * (eps_sct * mu_sct)**(0.5)
    Z_sct = (mu_sct/eps_sct)**(0.5)

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

    # Compute coefficients of series expansion
    arg0 = gamma_0 * a
    args = gam_sct * a
    A = np.zeros(M, dtype=np.complex_)

    for m in range(1, M+1):
        Ib0 = iv(m, arg0)
        dIb0 = iv(m+1, arg0) + m / arg0 * Ib0
        Ibs = iv(m, args)
        dIbs = iv(m+1, args) + m / args * Ibs
        Kb0 = kv(m, arg0)
        dKb0 = -kv(m+1, arg0) + m / arg0 * Kb0
        denominator = Z_sct * dIbs * Kb0 - dKb0 * Ibs
        A[m-1] = -(Z_sct * dIbs * Ib0 - dIb0 * Ibs) / denominator

    Er = np.zeros(rR.shape, dtype=np.complex_)
    Ephi = np.zeros(rR.shape, dtype=np.complex_)
    ZH3 = np.zeros(rR.shape, dtype=np.complex_)

    for m in range(1, M+1):
        arg0 = gamma_0 * rR
        Kb0 = kv(m, arg0)
        dKb0 = -kv(m+1, arg0) + m / arg0 * Kb0
        KbS = kv(m, gamma_0 * rS)
        Er = Er + A[m-1] * 2 * m**2 * Kb0 * KbS * np.cos(m*(phiR - phiS))
        Ephi = Ephi - A[m-1] * 2 * m * dKb0 * KbS * np.sin(m*(phiR - phiS))
        ZH3 = ZH3 - A[m-1] * 2 * m * Kb0 * KbS * np.sin(m*(phiR - phiS))

    Er = 1.0 / (2.0 * np.pi) * Er / rR / rS
    Ephi = gamma_0 * 1.0 / (2.0 * np.pi) * Ephi / rS
    ZH3 = -gamma_0 * 1.0 / (2.0 * np.pi) * ZH3 / rS

    E = {}
    E[1] = np.cos(phiR) * Er - np.sin(phiR) * Ephi
    E[2] = np.sin(phiR) * Er + np.cos(phiR) * Ephi

    Edata2D = (np.abs(E[1])**2 + np.abs(E[2])**2)**(0.5)
    Hdata2D = np.abs(ZH3)
    return Edata2D, Hdata2D


def IncEMwave(gamma_0, xS, delta, X1, X2):
    # Compute Incident field
    # incident wave from electric dipole in negative x_1
    factor = 2.0 * iv(1, gamma_0 * delta) / (gamma_0 * delta)

    X1 = X1 - xS[0]
    X2 = X2 - xS[1]
    DIS = np.sqrt(X1**2 + X2**2)
    X1 = X1 / DIS
    X2 = X2 / DIS

    # G = factor * 1 / (2 * pi) .* besselk(0, gam0*DIS);
    G = factor * 1.0 / (2.0 * np.pi) * kv(0, gamma_0 * DIS)
    # dG = -factor * gam0 .* 1 / (2 * pi) .* besselk(1, gam0*DIS);
    dG = -factor * gamma_0 * 1.0 / (2.0 * np.pi) * kv(1, gamma_0 * DIS)
    # dG11 = (2 * X1 .* X1 - 1) .* (-dG ./ DIS) + gam0^2 * X1 .* X1 .* G;
    dG11 = (2.0 * X1 * X1 - 1.0) * (-dG / DIS) + gamma_0**2 * X1 * X1 * G
    # dG21 = 2 * X2 .* X1 .* (-dG ./ DIS) + gam0^2 * X2 .* X1 .* G;
    dG21 = 2.0 * X2 * X1 * (-dG / DIS) + gamma_0**2 * X2 * X1 * G

    E_inc = {}
    # E_inc{1} = -(-gam0^2 * G + dG11);
    E_inc[1] = -(-gamma_0**2 * G + dG11)
    # E_inc{2} = -dG21;
    E_inc[2] = -dG21
    # E_inc{3} = 0;
    E_inc[3] = 0.0

    ZH_inc = {}
    ZH_inc[1] = 0.0
    ZH_inc[2] = 0.0
    ZH_inc[3] = gamma_0 * X2 * dG

    return E_inc, ZH_inc


def E_inc(IncEMwave):
    return IncEMwave[0]


def ZH_inc(IncEMwave):
    return IncEMwave[1]


def E_sct(w_E, FFTG, gamma_0, dx, N1, N2):
    return KwE(w_E, FFTG, gamma_0, dx, N1, N2)


def N(CHI_eps):
    return CHI_eps.flatten().shape[0]


def b(CHI_eps, E_inc, N):
    # Known 1D vector right-hand side
    b = np.zeros(2*N, dtype=np.complex_)
    # b(1:N, 1) = input.CHI_eps(:) .* E_inc{1}(:);
    b[0:N] = CHI_eps.T.flatten() * E_inc[1].T.flatten()
    # b(N+1:2*N, 1) = input.CHI_eps(:) .* E_inc{2}(:);
    b[N:2*N] = CHI_eps.T.flatten() * E_inc[2].T.flatten()
    return b


def w(b, CHI_eps, E_inc, ZH_inc, FFTG, N1, N2, Errcri, itmax, gamma_0, dx, N):
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
    Aw_operator = LinearOperator((b.shape[0], b.shape[0]), matvec=lambda w: Aw(w, N1, N2, FFTG, CHI_eps, gamma_0, dx, N))
    w, exit_code = bicgstab(Aw_operator, b, tol=Errcri, maxiter=itmax, callback=callback)

    # Display the convergence information
    print("Convergence information:", exit_code)
    print(exit_code)
    print("Iteration:", callback.iter_count)
    print("time_total", callback.time_total)
    return w


def w_E(w, N1, N2, N):
    # Output Matrix
    return vector2matrix(w, N1, N2, N)


def Aw(w, N1, N2, FFTG, CHI_eps, gamma_0, dx, N):
    # print("HERE Aw")
    # [w_E] = vector2matrix(w, input);
    w_E = vector2matrix(w, N1, N2, N)

    # [Kw_E] = KopE(w_E, input);
    Kw_E = KwE(w_E, FFTG, gamma_0, dx, N1, N2)

    y = np.zeros(2*N, dtype=np.complex_)
    # y(1:N, 1) = w_E{1}(:) - CHI_eps(:) .* Kw_E{1}(:)
    y[0:N] = w_E[1].flatten() - CHI_eps.flatten() * Kw_E[1].flatten()
    # y(N+1:2*N, 1) = w_E{2}(:) - CHI_eps(:) .* Kw_E{2}(:)
    y[N:2*N] = w_E[2].flatten() - CHI_eps.flatten() * Kw_E[2].flatten()
    # print("THERE Aw")
    return y


def vector2matrix(w, N1, N2, N):
    # Modify vector output from 'bicgstab' to matrices for further computation
    DIM = [N1, N2]
    # w_E = cell(1, 2);
    w_E = {}
    w_E[1] = np.reshape(w[0:N], DIM)
    # w_E{1} = reshape(w(1:N, 1), DIM);
    w_E[2] = np.reshape(w[N:2*N], DIM)
    # w_E{2} = reshape(w(N+1:2*N, 1), DIM);
    return w_E


def KwE(w_E, FFTG, gamma_0, dx, N1, N2):
    return KopE(w_E, FFTG, gamma_0, dx, N1, N2)


def KopE(w_E, FFTG, gamma_0, dx, N1, N2):
    KwE = {}
    for n in range(1, 3):
        # KwE{n} = Kop(wE{n}, input.FFTG);
        KwE[n] = Kop(w_E[n], FFTG)

    # Dummy is temporary storage
    # dummy = graddiv(KwE, input);
    dummy = graddiv(KwE, dx, N1, N2)
    for n in range(1, 3):
        # KwE{n} = KwE{n} - dummy{n} / input.gamma_0^2;
        KwE[n] = KwE[n] - dummy[n] / gamma_0**2
    return KwE


def Kop(w_E, FFTG):
    # print("HERE Kop")
    # Make FFT grid
    Cv = np.zeros(FFTG.shape, dtype=np.complex_)
    N1, N2 = w_E.shape
    Cv[0:N1, 0:N2] = w_E.copy()
    # Convolution by FFT
    Cv = np.fft.fftn(Cv)
    Cv = np.fft.ifftn(FFTG * Cv)
    Kv = Cv[0:N1, 0:N2]
    # print("THERE Kop")
    return Kv


def graddiv(KwE, dx, N1, N2):
    # print("HERE graddiv")
    u = {}
    u[1] = np.zeros(np.shape(KwE[1]), dtype=np.complex_)
    u[2] = u[1].copy()

    # Compute d1d1_v1, d2d2_v2
    # u{1}(2:N1 - 1, :) = v{1}(1:N1 - 2, :) - 2 * v{1}(2:N1 - 1, :) + v{1}(3:N1, :);
    u[1][1:N1-1, :] = KwE[1][0:N1-2, :] - 2 * KwE[1][1:N1-1, :] + KwE[1][2:N1, :]

    # u{2}(:, 2:N2 - 1) = v{2}(:, 1:N2 - 2) - 2 * v{2}(:, 2:N2 - 1) + v{2}(:, 3:N2);
    u[2][:, 1:N2-2] = KwE[2][:, 0:N2-3] - 2 * KwE[2][:, 1:N2-2] + KwE[2][:, 2:N2-1]

    # Replace the input vector v1 by d1_v and v2 by d2_v2
    # d1_v1
    # v{1}(2:N1 - 1, :) = (v{1}(3:N1, :) - v{1}(1:N1 - 2, :)) / 2;
    KwE[1][1:N1-1, :] = (KwE[1][2:N1, :] - KwE[1][0:N1-2, :]) / 2
    # d2_v2
    # v{2}(:, 2:N2 - 1) = (v{2}(:, 3:N2) - v{2}(:, 1:N2 - 2)) / 2;
    KwE[2][:, 1:N2-2] = (KwE[2][:, 2:N2-1] - KwE[2][:, 0:N2-3]) / 2

    # Add d1_v2 = d1d2_v2 to output vector u1
    # u{1}(2:N1 - 1, :) = u{1}(2:N1 - 1, :) + (v{2}(3:N1, :) - v{2}(1:N1 - 2, :)) / 2;
    u[2][1:N1-1, :] = u[1][1:N1-1, :] + (KwE[2][2:N1, :] - KwE[2][0:N1-2, :]) / 2.0

    # Add d2_v1 = d2d1_v1 to output vector u2
    # u{2}(:, 2:N2 - 1) = u{2}(:, 2:N2 - 1) + (v{1}(:, 3:N2) - v{1}(:, 1:N2 - 2)) / 2;
    u[2][:, 1:N2-1] = u[2][:, 1:N2-1] + (KwE[1][:, 2:N2] - KwE[1][:, 0:N2-2]) / 2.0

    # divide by dx^2
    u[1] = u[1] / dx**2
    u[2] = u[2] / dx**2

    # print("THERE graddiv")
    return u


def E(E_inc, E_sct):
    E = {}
    for n in range(1, 3):
        E[n] = E_inc[n] + E_sct[n]
    return E


def phi():
    # phi = 0:.01:2 * pi;
    return np.arange(0, 2.0*np.pi, 0.01)


def DOPwE(w_E, gamma_0, dx, xR, NR, delta, X1, X2):
    Edata = np.zeros((NR), dtype=np.complex_)
    Hdata = np.zeros((NR), dtype=np.complex_)

    # Weak Form
    factor = 2 * iv(1, gamma_0 * delta) / (gamma_0 * delta)

    for p in range(1, NR+1):
        X1 = xR[0, p-1] - X1
        X2 = xR[1, p-1] - X2

        DIS = np.sqrt(X1**2 + X2**2)
        X1 = X1 / DIS
        X2 = X2 / DIS

        G = factor * 1.0 / (2.0 * np.pi) * kv(0, gamma_0 * DIS)
        dG = -factor * gamma_0 * 1.0 / (2.0 * np.pi) * kv(1, gamma_0 * DIS)
        d1_G = X1 * dG
        d2_G = X2 * dG

        dG11 = (2.0 * X1 * X1 - 1.0) * (-dG / DIS) + gamma_0**2 * X1 * X1 * G
        dG22 = (2.0 * X2 * X2 - 1.0) * (-dG / DIS) + gamma_0**2 * X2 * X2 * G
        dG21 = 2.0 * X2 * X1 * (-dG / DIS) + gamma_0**2 * X2 * X1 * G

        # E1rfl = dx^2 * sum((gam0^2 * G(:) - dG11(:)).*w_E{1}(:) - dG21(:).*w_E{2}(:));
        E1rfl = dx**2 * np.sum(gamma_0**2 * G.flatten() - dG11.flatten() * w_E[1].flatten() - dG21.flatten() * w_E[2].flatten())
        E2rfl = dx**2 * np.sum(-dG21.flatten() * w_E[1].flatten() + (gamma_0**2 * G.flatten() - dG22.flatten()) * w_E[2].flatten())
        ZH3rfl = gamma_0 * dx**2 * np.sum(d2_G.flatten() * w_E[1].flatten() - d1_G.flatten() * w_E[2].flatten())

        Edata[p-1] = np.sqrt(np.abs(E1rfl)**2 + np.abs(E2rfl)**2)
        Hdata[p-1] = np.abs(ZH3rfl)
    return Edata, Hdata


def Edata(DOPwE):
    return DOPwE[0]


def Hdata(DOPwE):
    return DOPwE[1]


def plotContrastSourcewE(w_E, X1, X2):
    # Plot 2D contrast/source distribution
    x1 = X1[:, 0]
    x2 = X2[0, :]

    fig = plt.figure(figsize=(7.09, 4.72))
    fig.subplots_adjust(wspace=0.3)

    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(np.abs(w_E[1]), extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax1.set_xlabel('x$_2$ \u2192')
    ax1.set_ylabel('\u2190 x_1')
    ax1.set_aspect('equal', adjustable='box')
    fig.colorbar(im1, ax=ax1, orientation='horizontal')
    ax1.set_title(r'abs(w$_1^E$)', fontsize=13)

    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(np.abs(w_E[2]), extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax2.set_xlabel('x$_2$ \u2192')
    ax2.set_ylabel('\u2190 x_1')
    ax2.set_aspect('equal', adjustable='box')
    fig.colorbar(im2, ax=ax2, orientation='horizontal')
    ax2.set_title(r'abs(w$_2^E$)', fontsize=13)

    plt.show()


def displayEdata(Edata2D, rcvr_phi):
    angle = rcvr_phi * 180 / np.pi
    # Plot data at a number of receivers
    # fig = plt.figure(figsize=(0.39, 0.39), dpi=100)
    plt.plot(angle, abs(Edata2D), label='Edata2D Bessel-function method')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.title('scattered E data in 2D', fontsize=12)
    plt.axis('tight')
    plt.xlabel('observation angle in degrees')
    plt.xlim([0, 360])
    plt.ylabel('abs(data) $\\rightarrow$')
    plt.show()


def displayHdata(Hdata2D, rcvr_phi):
    angle = rcvr_phi * 180 / np.pi
    # Plot data at a number of receivers
    # fig = plt.figure(figsize=(0.39, 0.39), dpi=100)
    plt.plot(angle, abs(Hdata2D), label='Hdata2D Bessel-function method')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.title('scattered H data in 2D', fontsize=12)
    plt.axis('tight')
    plt.xlabel('observation angle in degrees')
    plt.xlim([0, 360])
    plt.ylabel('abs(data) $\\rightarrow$')
    plt.show()


def plotEMcontrast(X1, X2, CHI_eps, CHI_mu):
    # Plot Permittivity / Permeability Contrast
    x1 = X1[:, 0]
    x2 = X2[0, :]

    fig = plt.figure(figsize=(7.09, 4.72))
    fig.subplots_adjust(wspace=0.3)

    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(CHI_eps, extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax1.set_xlabel('x$_{2}$ \u2192')
    ax1.set_ylabel('\u2190 x$_{1}$')
    ax1.set_aspect('equal', adjustable='box')
    fig.colorbar(im1, ax=ax1, orientation='horizontal')
    ax1.set_title(r'$\chi^\epsilon =$1 - $\epsilon_{sct} / \epsilon_0$', fontsize=13)

    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(CHI_mu, extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax2.set_xlabel('x$_{2}$ \u2192')
    ax2.set_ylabel('\u2190 x$_{1}$')
    ax2.set_aspect('equal', adjustable='box')
    fig.colorbar(im2, ax=ax2, orientation='horizontal')
    ax2.set_title(r'$\chi^\mu =$1 - $\mu_{sct} / \mu_0$', fontsize=13)

    plt.show()


def plotEtotalwavefield(E, a, X1, X2, N1, N2, phi):
    # Plot wave fields in two-dimensional space
    fig, axs = plt.subplots(1, 2, figsize=(18, 12))

    im1 = axs[0].imshow(abs(E[1]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet')
    axs[0].set_xlabel('x$_2$ $\\rightarrow$')
    axs[0].set_ylabel('$\\leftarrow$ x$_1$')
    axs[0].set_title('2D Electric field E1', fontsize=13)
    fig.colorbar(im1, ax=axs[0], orientation='horizontal')
    axs[0].plot(a*np.cos(phi), a*np.sin(phi), 'w')

    im2 = axs[1].imshow(abs(E[2]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet')
    axs[1].set_xlabel('x$_2$ $\\rightarrow$')
    axs[1].set_ylabel('$\\leftarrow$ x$_1$')
    axs[1].set_title('2D Electric field E2', fontsize=13)
    fig.colorbar(im2, ax=axs[1], orientation='horizontal')
    axs[1].plot(a*np.cos(phi), a*np.sin(phi), 'w')

    plt.show()
