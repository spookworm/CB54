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


def wavelength(c_0, f):
    # wavelength
    return c_0 / f


def CHI_eps(eps_sct, a, R):
    # Compute permittivity contrast
    return (1 - eps_sct) * (R < a)


def CHI_mu(mu_sct, a, R):
    # Compute permeability contrast
    return (1 - mu_sct) * (R < a)


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
    A = np.zeros(M+1, dtype=np.complex_)

    for m in range(1, M+1):
        Ib0 = iv(m, arg0)
        dIb0 = iv(m+1, arg0) + m / arg0 * Ib0
        Ibs = iv(m, args)
        dIbs = iv(m+1, args) + m / args * Ibs
        Kb0 = kv(m, arg0)
        dKb0 = -kv(m+1, arg0) + m / arg0 * Kb0
        denominator = Z_sct * dIbs * Kb0 - dKb0 * Ibs
        A[m] = -(Z_sct * dIbs * Ib0 - dIb0 * Ibs) / denominator

    Er = np.zeros(rR.shape, dtype=np.complex_)
    Ephi = np.zeros(rR.shape, dtype=np.complex_)
    ZH3 = np.zeros(rR.shape, dtype=np.complex_)

    for m in range(1, M+1):
        arg0 = gamma_0 * rR
        Kb0 = kv(m, arg0)
        dKb0 = -kv(m+1, arg0) + m / arg0 * Kb0
        KbS = kv(m, gamma_0 * rS)
        Er = Er + A[m] * 2 * m**2 * Kb0 * KbS * np.cos(m*(phiR - phiS))
        Ephi = Ephi - A[m] * 2 * m * dKb0 * KbS * np.sin(m*(phiR - phiS))
        ZH3 = ZH3 - A[m] * 2 * m * Kb0 * KbS * np.sin(m*(phiR - phiS))

    Er = 1.0 / (2.0 * np.pi) * Er / rR / rS
    Ephi = gamma_0 * 1.0 / (2.0 * np.pi) * Ephi / rS
    ZH3 = -gamma_0 * 1.0 / (2.0 * np.pi) * ZH3 / rS

    E = {}
    E[1] = np.cos(phiR) * Er - np.sin(phiR) * Ephi
    E[2] = np.sin(phiR) * Er + np.cos(phiR) * Ephi

    Edata2D = (np.abs(E[1])**2 + np.abs(E[2])**2)**(0.5)
    Hdata2D = np.abs(ZH3)
    return Edata2D, Hdata2D


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


def IncEMwave(gamma_0, xS, dx, X1, X2):
    # Compute Incident field

    # incident wave from electric dipole in negative x_1
    # radius circle with area of dx^2
    delta = (np.pi)**(-0.5) * dx
    factor = 2.0 * iv(1, gamma_0 * delta) / (gamma_0 * delta)
    X1 = X1 - xS[0]
    X2 = X2 - xS[1]
    DIS = np.sqrt(X1**2 + X2**2)
    X1 = X1 / DIS
    X2 = X2 / DIS

    G = factor * 1.0 / (2.0 * np.pi) * kv(0, gamma_0 * DIS)
    dG = -factor * gamma_0 * 1.0 / (2.0 * np.pi) * kv(1, gamma_0 * DIS)

    dG11 = (2.0 * X1 * X1 - 1.0) * (-dG / DIS) + gamma_0**2 * X1 * X1 * G
    dG21 = 2.0 * X2 * X1 * (-dG / DIS) + gamma_0**2 * X2 * X1 * G

    E_inc = {}
    E_inc[1] = -(-gamma_0**2 * G + dG11)
    E_inc[2] = -dG21
    E_inc[3] = 0

    ZH_inc = {}
    ZH_inc[1] = 0
    ZH_inc[2] = 0
    ZH_inc[3] = gamma_0 * X2 * dG

    return E_inc, ZH_inc


def E_inc(IncEMwave):
    return IncEMwave[0]


def ZH_inc(IncEMwave):
    return IncEMwave[1]
