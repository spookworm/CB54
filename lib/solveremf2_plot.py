import matplotlib.pyplot as plt
import numpy as np


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


def graph_resivec_iter(iterative_information):
    """
    Plot the resvec at each iteration
    iter
    tresvec
    time_total
    """
    x1 = iterative_information[:, 0]
    y = iterative_information[:, 1]
    x2 = iterative_information[:, 2]

    fig, axs = plt.subplots(2, dpi=300)
    fig.suptitle('Initial Error: ' + str(y[0]))
    plt.subplots_adjust(top=0.85)

    axs[0].plot(x1, y, 'r-o', linewidth=1, markersize=2)
    axs[0].set_title('Residual Norm versus Iteration Count')
    axs[0].set_xlim(np.min(x1), np.max(x1))

    axs[1].plot(x2, y, 'b-o', linewidth=1, markersize=2)
    axs[1].set_title('Residual Norm versus Computation Time')
    axs[1].set_xlim(np.min(x2), np.max(x2))

    # plt.subplots_adjust(wspace=1.5)
    plt.subplots_adjust(hspace=0.75)

    area1 = np.trapz(y, x1)
    axs[0].text(0.5*np.max(x1), 0.8*np.max(np.abs(y)), 'Area: ' + str(area1) + ';\nIter Count: ' + str(int(np.max(x1))), color='red', ha='center', va='center', fontsize=12)

    area2 = np.trapz(y, x2)
    axs[1].text(0.5*np.max(x2), 0.8*np.max(np.abs(y)), 'Area: ' + str(area2) + ';\nTime (sec): ' + str(np.max(x2)), color='red', ha='center', va='center', fontsize=12)

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


def plotContrastSourcewE(w_E, X1, X2):
    # Plot 2D contrast/source distribution
    x1 = X1[:, 0]
    x2 = X2[0, :]
    fig = plt.figure(figsize=(7.09, 4.72))
    fig.subplots_adjust(wspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(np.abs(w_E[0]), extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax1.set_xlabel('x$_2$ \u2192')
    ax1.set_ylabel('\u2190 x_1')
    ax1.set_aspect('equal', adjustable='box')
    fig.colorbar(im1, ax=ax1, orientation='horizontal')
    ax1.set_title(r'abs(w$_1^E$)', fontsize=13)
    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(np.abs(w_E[1]), extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax2.set_xlabel('x$_2$ \u2192')
    ax2.set_ylabel('\u2190 x_1')
    ax2.set_aspect('equal', adjustable='box')
    fig.colorbar(im2, ax=ax2, orientation='horizontal')
    ax2.set_title(r'abs(w$_2^E$)', fontsize=13)
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
    im1 = axs[0].imshow(abs(E[0]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet')
    axs[0].set_xlabel('x$_2$ $\\rightarrow$')
    axs[0].set_ylabel('$\\leftarrow$ x$_1$')
    axs[0].set_title('2D Electric field E1', fontsize=13)
    fig.colorbar(im1, ax=axs[0], orientation='horizontal')
    axs[0].plot(a*np.cos(phi), a*np.sin(phi), 'w')
    im2 = axs[1].imshow(abs(E[1]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet')
    axs[1].set_xlabel('x$_2$ $\\rightarrow$')
    axs[1].set_ylabel('$\\leftarrow$ x$_1$')
    axs[1].set_title('2D Electric field E2', fontsize=13)
    fig.colorbar(im2, ax=axs[1], orientation='horizontal')
    axs[1].plot(a*np.cos(phi), a*np.sin(phi), 'w')
    plt.show()


def subplotsComplexArray(u_inc, cmap_min, cmap_max):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3)

    # Plot the real part
    axes[0].imshow(np.real(u_inc), cmap='jet', vmin=cmap_min, vmax=cmap_max)
    axes[0].set_title('Real Part')

    # Plot the imaginary part
    axes[1].imshow(np.imag(u_inc), cmap='jet', vmin=cmap_min, vmax=cmap_max)
    axes[1].set_title('Imaginary Part')

    # Plot the absolute part
    axes[2].imshow(np.abs(u_inc), cmap='jet', vmin=cmap_min, vmax=cmap_max)
    axes[2].set_title('Absolute Part')

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()

