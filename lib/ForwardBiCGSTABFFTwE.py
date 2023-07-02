import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv, iv
from scipy.sparse.linalg import bicgstab, LinearOperator
import time


# # Delete existing data if the file exists
# file_path = 'data2D.txt'
# if os.path.exists(file_path):
#     os.remove(file_path)
# ForwardBiCGSTABFFT.data_save('', 'data2D', WavefieldSctCircle)
# data_load = ForwardBiCGSTABFFT.data_load('', 'data2D.txt')


def w_E(w, N1, N2, N):
    # Output Matrix
    return vector2matrix(w, N1, N2, N)


def KwE(w_E, FFTG, gamma_0, dx, N1, N2):
    return KopE(w_E, FFTG, gamma_0, dx, N1, N2)


# def E_sct(w_E, FFTG, gamma_0, dx, N1, N2):
#     return KwE(w_E, FFTG, gamma_0, dx, N1, N2)


# def E(E_inc, E_sct):
#     E = {}
#     for n in range(1, 3):
#         E[n] = E_inc[n] + E_sct[n]
#     return E


# def phi():
#     # phi = 0:.01:2 * pi;
#     return np.arange(0, 2.0*np.pi, 0.01)


# def DOPwE(w_E, gamma_0, dx, xR, NR, delta, X1, X2):
#     Edata = np.zeros((NR), dtype=np.complex_)
#     Hdata = np.zeros((NR), dtype=np.complex_)

#     # Weak Form
#     factor = 2 * iv(1, gamma_0 * delta) / (gamma_0 * delta)

#     for p in range(1, NR+1):
#         X1 = xR[0, p-1] - X1
#         X2 = xR[1, p-1] - X2

#         DIS = np.sqrt(X1**2 + X2**2)
#         X1 = X1 / DIS
#         X2 = X2 / DIS

#         G = factor * 1.0 / (2.0 * np.pi) * kv(0, gamma_0 * DIS)
#         dG = -factor * gamma_0 * 1.0 / (2.0 * np.pi) * kv(1, gamma_0 * DIS)
#         d1_G = X1 * dG
#         d2_G = X2 * dG

#         dG11 = (2.0 * X1 * X1 - 1.0) * (-dG / DIS) + gamma_0**2 * X1 * X1 * G
#         dG22 = (2.0 * X2 * X2 - 1.0) * (-dG / DIS) + gamma_0**2 * X2 * X2 * G
#         dG21 = (2.0 * X2 * X1      ) * (-dG / DIS) + gamma_0**2 * X2 * X1 * G

#         # E1rfl = dx^2 * sum((gam0^2 * G(:) - dG11(:)).*w_E{1}(:) - dG21(:).*w_E{2}(:));
#         E1rfl = dx**2 * np.sum(gamma_0**2 * G.flatten() - dG11.flatten() * w_E[1].flatten() - dG21.flatten() * w_E[2].flatten())
#         E2rfl = dx**2 * np.sum(-dG21.flatten() * w_E[1].flatten() + (gamma_0**2 * G.flatten() - dG22.flatten()) * w_E[2].flatten())
#         ZH3rfl = gamma_0 * dx**2 * np.sum(d2_G.flatten() * w_E[1].flatten() - d1_G.flatten() * w_E[2].flatten())

#         Edata[p-1] = np.sqrt(np.abs(E1rfl)**2 + np.abs(E2rfl)**2)
#         Hdata[p-1] = np.abs(ZH3rfl)
#     return Edata, Hdata


# def Edata(DOPwE):
#     return DOPwE[0]


# def Hdata(DOPwE):
#     return DOPwE[1]


# def displayEdata(Edata2D, rcvr_phi):
#     angle = rcvr_phi * 180 / np.pi
#     # Plot data at a number of receivers
#     # fig = plt.figure(figsize=(0.39, 0.39), dpi=100)
#     plt.plot(angle, abs(Edata2D), label='Edata2D Bessel-function method')
#     plt.tight_layout()
#     plt.legend(loc='upper right')
#     plt.title('scattered E data in 2D', fontsize=12)
#     plt.axis('tight')
#     plt.xlabel('observation angle in degrees')
#     plt.xlim([0, 360])
#     plt.ylabel('abs(data) $\\rightarrow$')
#     plt.show()


# def displayHdata(Hdata2D, rcvr_phi):
#     angle = rcvr_phi * 180 / np.pi
#     # Plot data at a number of receivers
#     # fig = plt.figure(figsize=(0.39, 0.39), dpi=100)
#     plt.plot(angle, abs(Hdata2D), label='Hdata2D Bessel-function method')
#     plt.tight_layout()
#     plt.legend(loc='upper right')
#     plt.title('scattered H data in 2D', fontsize=12)
#     plt.axis('tight')
#     plt.xlabel('observation angle in degrees')
#     plt.xlim([0, 360])
#     plt.ylabel('abs(data) $\\rightarrow$')
#     plt.show()


# def plotEMcontrast(X1, X2, CHI_eps, CHI_mu):
#     # Plot Permittivity / Permeability Contrast
#     x1 = X1[:, 0]
#     x2 = X2[0, :]

#     fig = plt.figure(figsize=(7.09, 4.72))
#     fig.subplots_adjust(wspace=0.3)

#     ax1 = fig.add_subplot(1, 2, 1)
#     im1 = ax1.imshow(CHI_eps, extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
#     ax1.set_xlabel('x$_{2}$ \u2192')
#     ax1.set_ylabel('\u2190 x$_{1}$')
#     ax1.set_aspect('equal', adjustable='box')
#     fig.colorbar(im1, ax=ax1, orientation='horizontal')
#     ax1.set_title(r'$\chi^\epsilon =$1 - $\epsilon_{sct} / \epsilon_0$', fontsize=13)

#     ax2 = fig.add_subplot(1, 2, 2)
#     im2 = ax2.imshow(CHI_mu, extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
#     ax2.set_xlabel('x$_{2}$ \u2192')
#     ax2.set_ylabel('\u2190 x$_{1}$')
#     ax2.set_aspect('equal', adjustable='box')
#     fig.colorbar(im2, ax=ax2, orientation='horizontal')
#     ax2.set_title(r'$\chi^\mu =$1 - $\mu_{sct} / \mu_0$', fontsize=13)

#     plt.show()


# def plotEtotalwavefield(E, a, X1, X2, N1, N2, phi):
#     # Plot wave fields in two-dimensional space
#     fig, axs = plt.subplots(1, 2, figsize=(18, 12))

#     im1 = axs[0].imshow(abs(E[1]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet')
#     axs[0].set_xlabel('x$_2$ $\\rightarrow$')
#     axs[0].set_ylabel('$\\leftarrow$ x$_1$')
#     axs[0].set_title('2D Electric field E1', fontsize=13)
#     fig.colorbar(im1, ax=axs[0], orientation='horizontal')
#     axs[0].plot(a*np.cos(phi), a*np.sin(phi), 'w')

#     im2 = axs[1].imshow(abs(E[2]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet')
#     axs[1].set_xlabel('x$_2$ $\\rightarrow$')
#     axs[1].set_ylabel('$\\leftarrow$ x$_1$')
#     axs[1].set_title('2D Electric field E2', fontsize=13)
#     fig.colorbar(im2, ax=axs[1], orientation='horizontal')
#     axs[1].plot(a*np.cos(phi), a*np.sin(phi), 'w')

#     plt.show()

# def data_save(path, filename, data2D):
#     # save DATA2D data2D;
#     np.savetxt(path + filename + '.txt', data2D.view(np.float_))


# def data_load(path, filename):
#     # load DATA2D data2D;
#     return np.loadtxt(path + filename).view(np.complex_)


# def Edata2D(EMsctCircle):
#     return EMsctCircle[0]


# def Hdata2D(EMsctCircle):
#     return EMsctCircle[1]


# def EMsctCircle(c_0, eps_sct, mu_sct, gamma_0, xR, xS, M, a):

#     gam_sct = gamma_0 * (eps_sct * mu_sct)**(0.5)
#     Z_sct = (mu_sct/eps_sct)**(0.5)

#     # Compute reflected field at receivers (data)
#     # rR = sqrt(xR(1, :).^2+xR(2, :).^2)
#     # rR = np.sqrt(xR[0, :]**2 + xR[1, :]**2)
#     rR = np.linalg.norm(xR, axis=0)

#     # phiR = atan2(xR(2, :), xR(1, :))
#     phiR = np.arctan2(xR[1, :], xR[0, :])

#     # rS = sqrt(xS(1)^2+xS(2)^2)
#     # rS = np.sqrt(xS[0]**2 + xS[1]**2)
#     rS = np.linalg.norm(xS, axis=0)

#     # phiS = atan2(xS(2), xS(1))
#     phiS = np.arctan2(xS[1], xS[0])

#     # Compute coefficients of series expansion
#     arg0 = gamma_0 * a
#     args = gam_sct * a
#     A = np.zeros(M, dtype=np.complex_)

#     for m in range(1, M+1):
#         Ib0 = iv(m, arg0)
#         dIb0 = iv(m+1, arg0) + m / arg0 * Ib0
#         Ibs = iv(m, args)
#         dIbs = iv(m+1, args) + m / args * Ibs
#         Kb0 = kv(m, arg0)
#         dKb0 = -kv(m+1, arg0) + m / arg0 * Kb0
#         denominator = Z_sct * dIbs * Kb0 - dKb0 * Ibs
#         A[m-1] = -(Z_sct * dIbs * Ib0 - dIb0 * Ibs) / denominator

#     Er = np.zeros(rR.shape, dtype=np.complex_)
#     Ephi = np.zeros(rR.shape, dtype=np.complex_)
#     ZH3 = np.zeros(rR.shape, dtype=np.complex_)

#     for m in range(1, M+1):
#         arg0 = gamma_0 * rR
#         Kb0 = kv(m, arg0)
#         dKb0 = -kv(m+1, arg0) + m / arg0 * Kb0
#         KbS = kv(m, gamma_0 * rS)
#         Er = Er + A[m-1] * 2 * m**2 * Kb0 * KbS * np.cos(m*(phiR - phiS))
#         Ephi = Ephi - A[m-1] * 2 * m * dKb0 * KbS * np.sin(m*(phiR - phiS))
#         ZH3 = ZH3 - A[m-1] * 2 * m * Kb0 * KbS * np.sin(m*(phiR - phiS))

#     Er = 1.0 / (2.0 * np.pi) * Er / rR / rS
#     Ephi = gamma_0 * 1.0 / (2.0 * np.pi) * Ephi / rS
#     ZH3 = -gamma_0 * 1.0 / (2.0 * np.pi) * ZH3 / rS

#     E = {}
#     E[1] = np.cos(phiR) * Er - np.sin(phiR) * Ephi
#     E[2] = np.sin(phiR) * Er + np.cos(phiR) * Ephi

#     Edata2D = np.sqrt(np.abs(E[1])**2 + np.abs(E[2])**2)
#     Hdata2D = np.abs(ZH3)
#     return Edata2D, Hdata2D
