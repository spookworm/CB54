def init():
    # Time factor = exp(-iwt)
    # Spatial units is in m
    # Source wavelet  Q = 1

    # wave speed in embedding
    c_0 = 1500
    # wave speed in scatterer
    c_sct = 3000
	
	
	
    # temporal frequency
    f = 50
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

    def initContrast(c_0, c_sct, X1, X2):
        # half width slab / radius circle cylinder / radius sphere
        a = 40
        R = np.sqrt(X1**2 + X2**2)
        contrast = 1 - c_0**2/c_sct**2
        CHI = contrast * (R < a)






        return a, CHI

    # add contrast distribution
    a, CHI = initContrast(c_0, c_sct, X1, X2)

    Errcri = 1e-18
    return c_0, c_sct, gamma_0, xS, NR, rcvr_phi, xR, N1, N2, dx, X1, X2, FFTG, a, CHI, Errcri


def WavefieldSctCircle():
    from scipy.special import kv, iv
    import os

    c_0, c_sct, gamma_0, xS, NR, rcvr_phi, xR, N1, N2, dx, X1, X2, FFTG, a, CHI, Errcri = init()

    gam_sct = gamma_0 * c_0/c_sct

    if os.path.exists('DATA2D.npy'):
        os.remove('DATA2D.npy')

    # (1) Compute coefficients of series expansion
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

    # (2) Compute reflected field at receivers (data)
    rR = np.zeros((1, xR.shape[1]), dtype=np.complex128, order='F')
    rR[0, :] = np.sqrt(xR[0, :]**2 + xR[1, :]**2)
    phiR = np.zeros((1, xR.shape[1]), dtype=np.complex128, order='F')
    phiR[0, :] = np.arctan2(xR[1, :], xR[0, :])
    rS = np.sqrt(xS[0, 0]**2 + xS[0, 1]**2)
    phiS = np.arctan2(xS[0, 1], xS[0, 0])
    data2D = A[0, 0] * kv(0, gamma_0*rS) * kv(0, gamma_0*rR)

    for m in range(1, M+1):
        factor = 2 * kv(m, gamma_0*rS) * np.cos(m*(phiS-phiR))
        data2D = data2D + A[0, m] * factor * kv(m, gamma_0*rR)

    data2D = 1/(2*np.pi) * data2D
    angle = rcvr_phi * 180 / np.pi
    displayDataBesselApproach(data2D, angle)
    np.savez('data2D.npz', data=data2D)
    return data2D


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


def plotContrastSource(w, CHI, X1, X2):
    import matplotlib.pyplot as plt
    # Plot 2D contrast/source distribution
    x1 = X1[:, 0]
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
