from IPython import get_ipython
import numpy as np
import sys
import time

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=20)


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


def DOPwE(w_E, gamma_0, dx, xR, NR, X1, X2):
    # (4) Compute synthetic data and plot fields and data --------------------
    from scipy.special import kv, iv

    # radius circle with area of dx^2
    delta = (np.pi)**(-1/2) * dx
    # Weak form
    factor = 2 * iv(1, gamma_0*delta) / (gamma_0*delta)
    Edata = np.zeros((1, NR), dtype=np.complex_, order='F')
    Hdata = np.zeros((1, NR), dtype=np.complex_, order='F')
    X1_ = np.zeros(X1.shape, dtype=np.complex_, order='F')
    X2_ = np.zeros(X2.shape, dtype=np.complex_, order='F')
    for p in range(1, NR+1):
        X1_ = xR[0, p-1] - X1
        X2_ = xR[1, p-1] - X2

        DIS = np.sqrt(X1_**2 + X2_**2)
        X1_ = X1_.copy() / DIS
        X2_ = X2_.copy() / DIS

        G = factor * 1.0 / (2.0 * np.pi) * kv(0, gamma_0*DIS)
        dG = -factor * gamma_0 * 1.0 / (2.0 * np.pi) * kv(1, gamma_0*DIS)
        d1_G = X1_ * dG
        d2_G = X2_ * dG

        dG11 = (2.0 * X1_ * X1_ - 1.0) * (-dG / DIS) + gamma_0**2 * X1_ * X1_ * G
        dG22 = (2.0 * X2_ * X2_ - 1.0) * (-dG / DIS) + gamma_0**2 * X2_ * X2_ * G
        dG21 = (2.0 * X2_ * X1_      ) * (-dG / DIS) + gamma_0**2 * X2_ * X1_ * G

        # E1rfl = dx^2 * sum((gam0^2 * G(:) - dG11(:)).*w_E{1}(:) - dG21(:).*w_E{2}(:));
        E1rfl = dx**2 * np.sum((gamma_0**2 * G.flatten('F') - dG11.flatten('F')) * w_E[0, :, :].flatten('F') - dG21.flatten('F') * w_E[1, :, :].flatten('F'))
        E2rfl = dx**2 * np.sum(-dG21.flatten('F') * w_E[0, :, :].flatten('F') + (gamma_0**2 * G.flatten('F') - dG22.flatten('F')) * w_E[1, :, :].flatten('F'))
        ZH3rfl = gamma_0 * dx**2 * np.sum(d2_G.flatten('F') * w_E[0, :, :].flatten('F') - d1_G.flatten('F') * w_E[1, :, :].flatten('F'))

        Edata[0, p-1] = np.sqrt(np.abs(E1rfl)**2 + np.abs(E2rfl)**2)
        Hdata[0, p-1] = np.abs(ZH3rfl)
    return Edata, Hdata


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


# # START OF ForwardBiCGSTABFFTwE
# Edata2D, Hdata2D = EMsctCircle()
plotEMContrast(CHI_eps, CHI_mu, X1, X2)
E_inc, ZH_inc = IncEMwave(gamma_0, xS, dx, X1, X2)

tic0 = time.time()
w_E, exit_code, information = ITERBiCGSTABwE(E_inc, CHI_eps, Errcri, x0=None)
toc0 = time.time() - tic0
print("toc", toc0)
tic1 = time.time()
w_E, exit_code, information = ITERBiCGSTABwE(E_inc, CHI_eps, Errcri, x0=w_E.copy().flatten('F'))
toc1 = time.time() - tic1
print("toc", toc1)

plotContrastSourcewE(w_E, X1, X2)
E_sct = KopE(w_E, gamma_0)
E_val = E(E_inc, E_sct)
plotEtotalwavefield(E_val, a, X1, X2, N1, N2)


def plotML(E, X1, X2, N1, N2):
    # Plot wave fields in two-dimensional space
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(18, 12))
    im1 = axs[0].imshow(abs(E[0]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet')
    axs[0].set_xlabel('x$_2$ $\\rightarrow$')
    axs[0].set_ylabel('$\\leftarrow$ x$_1$')
    axs[0].set_title('2D Electric field E1', fontsize=13)
    fig.colorbar(im1, ax=axs[0], orientation='horizontal')
    im2 = axs[1].imshow(abs(E[1]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet')
    axs[1].set_xlabel('x$_2$ $\\rightarrow$')
    axs[1].set_ylabel('$\\leftarrow$ x$_1$')
    axs[1].set_title('2D Electric field E2', fontsize=13)
    fig.colorbar(im2, ax=axs[1], orientation='horizontal')
    plt.show()


def plotComplexArray(cmd, array, cmap_name, cmap_min, cmap_max, title):
    import numpy as np
    from skimage import io
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if cmd == 'real':
        array = np.real(array)
    elif cmd == 'imag':
        array = np.imag(array)
    elif cmd == 'abs':
        array = np.abs(array)
    else:
        print("error input cmd string")
    normalized_array = (array - cmap_min) / (cmap_max - cmap_min)
    # colormap = cm.get_cmap('jet')
    colormap = mpl.colormaps['jet']
    colored_array = colormap(normalized_array)
    colored_array = (colored_array * 255).astype(np.uint8)
    io.imsave('output.png', colored_array)

    # Plot the image
    plt.imshow(colored_array)

    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('Value')

    # Display the plot
    plt.show()


# useless, must export as array as visualisation clipping etc and range setting is very difficult. so just export as a numerical complex array

plotComplexArray(cmd='abs', array=E_inc[0], cmap_name='jet', cmap_min=0, cmap_max=np.max(np.abs(E_inc)), title='real_part')
plotComplexArray(cmd='abs', array=E_sct[0], cmap_name='jet', cmap_min=0, cmap_max=np.max(np.abs(E_sct)), title='real_part')
plotComplexArray(cmd='abs', array=E_val[0], cmap_name='jet', cmap_min=0, cmap_max=np.max(np.abs(E_val)), title='real_part')

plotComplexArray(cmd='abs', array=E_inc[1], cmap_name='jet', cmap_min=0, cmap_max=np.max(np.abs(E_inc)), title='real_part')
plotComplexArray(cmd='abs', array=E_sct[1], cmap_name='jet', cmap_min=0, cmap_max=np.max(np.abs(E_sct)), title='real_part')
plotComplexArray(cmd='abs', array=E_val[1], cmap_name='jet', cmap_min=0, cmap_max=np.max(np.abs(E_val)), title='real_part')

# plotComplexArray(cmd='abs', array=E_inc[2], cmap_name='jet', cmap_min=0, cmap_max=np.max(np.abs(E_inc)), title='real_part')
# plotComplexArray(cmd='abs', array=E_sct[2], cmap_name='jet', cmap_min=0, cmap_max=np.max(np.abs(E_sct)), title='real_part')
# plotComplexArray(cmd='abs', array=E_val[2], cmap_name='jet', cmap_min=0, cmap_max=np.max(np.abs(E_val)), title='real_part')


# E_inc
# E_sct
# E_val
plotEtotalwavefield(E_val, a, X1, X2, N1, N2)
plotML(E_val, X1, X2, N1, N2)

# Edata, Hdata = DOPwE(w_E, gamma_0, dx, xR, NR, X1, X2)
# angle = rcvr_phi * 180 / np.pi
# displayDataBesselApproach(Edata, angle)
# displayDataBesselApproach(Hdata, angle)
# displayDataCompareApproachs(Edata2D, Edata, angle)
# displayDataCompareApproachs(Hdata2D, Hdata, angle)


def plotComplexContrast(CHI_eps, X1, X2):
    import matplotlib.pyplot as plt
    # Plot 2D contrast/source distribution
    # x1 = ForwardBiCGSTABFFT.input.X1(:, 1);
    x1 = X1[:, 0]
    # x2 = ForwardBiCGSTABFFT.input.X2(1, :);
    x2 = X2[0, :]

    fig = plt.figure(figsize=(7.09, 4.72))
    fig.subplots_adjust(wspace=0.3)

    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(CHI_eps, extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax1.set_xlabel('x_2 \u2192')
    ax1.set_ylabel('\u2190 x_1')
    ax1.set_aspect('equal', adjustable='box')
    fig.colorbar(im1, ax=ax1, orientation='horizontal')
    ax1.set_title(r'$\chi^\epsilon = 1 - \epsilon_{sct}/\epsilon_{0}$', fontsize=13)

    plt.show()
    from skimage import io
    io.imsave('output.png', CHI_eps)


plotComplexContrast(CHI_eps, X1, X2)
