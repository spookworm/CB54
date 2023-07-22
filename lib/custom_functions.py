# import graphviz
import sys
import os
import numpy as np
import time


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
    """
    CHECK
    """
    return rcvr_phi * 180 / np.pi


def angular_frequency(f):
    """
    # angular frequency (rad/s)
    """
    return 2.0 * np.pi * f


def Aw(w, N1, N2, FFTG, CHI):
    """
    CHECK
    Parameters
    ----------
    w : TYPE
        DESCRIPTION.
    N1 : TYPE
        DESCRIPTION.
    N2 : TYPE
        DESCRIPTION.
    FFTG : TYPE
        DESCRIPTION.
    CHI : TYPE
        DESCRIPTION.

    Returns
    -------
    y : TYPE
        DESCRIPTION.
    """
    # Convert 1D vector to matrix
    w = w.reshape((N1, N2), order='F')
    y = np.zeros((N1, N2), dtype=np.complex128, order='F')
    y = w - CHI * Kop(w, FFTG)
    # Convert matrix to 1D vector
    y = y.flatten('F')
    return y


def b(CHI, u_inc):
    """
    Known 1D vector right-hand side
    b = CHI(:) * u_inc(:)
    """
    b = np.zeros((u_inc.flatten('F').shape[0], 1), dtype=np.complex128, order='F')
    b[:, 0] = CHI.flatten('F') * u_inc.flatten('F')
    return b


def c_0(epsilon0, mu0):
    """
    This takes the permittivity and permiability in vacuum and returns the wave speed in embedding in meters per second.
    Args:
        input_ (float): wave speed in embedding.

    Returns:
        float: wave speed in embedding.
    """
    # help(solver_func.c_0)
    return np.power(epsilon0*mu0, -0.5)


def c_sct(c_0, contrast_sct):
    """
    This takes the input for wave speed in embedding and the contrast of the scatterer in meters per second and returns the wave speed in the scatterer.
    Args:
        input_ (float): wave speed in scatterer.

    Returns:
        float: wave speed in scatterer.
    """
    # help(solver_func.c_sct)
    return c_0 * np.sqrt(1/contrast_sct)


def CHI(CHI_array=None):
    """
    CHECK
    # add contrast distribution
    """
    if CHI_array is None:
        print("IMPLEMENT GEOMETRY")
        sys.exit()
    else:
        return CHI_array


def CHI_Bessel(c_0, c_sct, R, a):
    """
    CHECK
    # add contrast distribution
    """
    CHI = (1 - c_0**2 / c_sct**2) * (R < a)
    return CHI


def composer_render(composer_call, path_doc, filename):
    """
    Parameters
    ----------
    composer_call : TYPE
        DESCRIPTION.
    path_doc : TYPE
        DESCRIPTION.
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    composer_call.graphviz().render(directory=path_doc, filename=filename, format='png')
    os.remove(path_doc + filename)
    return None


def contrast_sct(relative_constrast_scatter):
    """
    This takes the input for contrast of scatterer and returns it.
    Args:
        input_ (float): contrast of scatterer.

    Returns:
        float: contrast of scatterer.
    """
    # help(solver_func.contrast_sct)
    return relative_constrast_scatter


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
    plt.legend(['Integral-equation method', 'Bessel-function method'], loc='center')
    plt.text(0.5*np.max(angle), np.max(np.abs(CIS_approach)), 'Error$^{sct}$ = ' + error, color='red', ha='center', va='center')
    plt.title('scattered wave data in 2D', fontsize=12)
    plt.axis('tight')
    plt.xlabel('observation angle in degrees')
    plt.xlim([0, 360])
    plt.ylabel('abs(data) $\\rightarrow$')
    plt.show()


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


def dx(mesh_sample_length):
    """
    Parameters
    ----------
    input_ : float
        Meshsize

    Returns
    -------
    input_ : float
        Meshsize
    """
    return mesh_sample_length


def epsilon0():
    return 8.854187817e-12


def Errcri(krylov_error_tolerance):
    """
    This takes the input for the tolerance of the BICGSTAB method.
    Args:
        input_ (float): tolerance

    Returns:
        float: tolerance
    """
    # help(solver_func.Errcri)
    return krylov_error_tolerance


def exit_code(ITERBiCGSTABw):
    return ITERBiCGSTABw[1]


def f(carrier_temporal_frequency):
    """
    This takes the input for the carrier temporal frequency in Hz and returns the carrier temporal frequency in Hz.
    Args:
        input_ (float): the carrier temporal frequency in Hz.

    Returns:
        float: the carrier temporal frequency in Hz.
    """
    # help(solver_func.f)
    return carrier_temporal_frequency


def FFTG(Green):
    """
    # Apply n-dimensional Fast Fourier transform
    """
    return np.fft.fftn(Green)


def gamma_0(s, c_0):
    """
    This takes the Laplace Parameter and the wave speed in embedding and returns the Propagation Co-efficient.
    Args:
        s (complex): the Laplace Parameter.
        c_0 (float): the wave speed in embedding in meters per second .

    Returns:
        complex: Propagation Co-efficient.
    """
    return s / c_0


def Green(dx, gamma_0, X1fft, X2fft):
    # compute gam_0^2 * subdomain integrals  of Green function
    from scipy.special import kv, iv
    DIS = np.sqrt(X1fft**2 + X2fft**2)
    # avoid Green's singularity for DIS = 0
    DIS[0, 0] = 1
    G = 1/(2*np.pi) * kv(0, gamma_0*DIS)
    # radius circle with area of dx^2
    delta = (np.pi)**(-1/2) * dx
    factor = 2 * iv(1, gamma_0*delta) / (gamma_0*delta)
    # integral includes gam0^2
    IntG = (gamma_0**2 * dx**2) * factor**2 * G
    IntG[0, 0] = 1 - gamma_0*delta * kv(1, gamma_0*delta) * factor
    return IntG


def information(ITERBiCGSTABw):
    return ITERBiCGSTABw[2]


# def init(c_0, c_sct, gamma_0, xS, NR, rcvr_phi, xR, N1, N2, dx, X1, X2, FFTG, a, CHI, Errcri):
#     """
#     Wrapper Function
#     """
#     return c_0, c_sct, gamma_0, xS, NR, rcvr_phi, xR, N1, N2, dx, X1, X2, FFTG, a, CHI, Errcri


def initFFTGreenGrid(N1, N2, dx, gamma_0):
    """
    Compute FFT of Green function
    Parameters
    ----------
    N1 : TYPE
        DESCRIPTION.
    N2 : TYPE
        DESCRIPTION.
    dx : TYPE
        DESCRIPTION.
    gamma_0 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    N1fft = np.int64(2**np.ceil(np.log2(2*N1)))
    N2fft = np.int64(2**np.ceil(np.log2(2*N2)))

    # x1(1:N1fft) = [0 : N1fft/2-1   N1fft/2 : -1 : 1] * input.dx;
    x1 = np.zeros((1, N1fft), dtype=np.float64, order='F')
    x1[0, :] = np.concatenate((np.arange(0, N1fft//2), np.arange(N1fft//2, 0, -1)))*dx

    # x2(1:N2fft) = [0 : N2fft/2-1   N2fft/2 : -1 : 1] * input.dx;
    x2 = np.zeros((1, N2fft), dtype=np.float64, order='F')
    x2[0, :] = np.concatenate((np.arange(0, N2fft//2), np.arange(N2fft//2, 0, -1)))*dx

    # [temp.X1fft,temp.X2fft] = ndgrid(x1,x2);
    X1fft, X2fft = np.meshgrid(x1, x2, indexing='ij')
    return X1fft, X2fft


def initGrid(N1, N2, dx):
    """
    Add grid in either 2D

    Parameters
    ----------
    N1 : TYPE
        DESCRIPTION.
    N2 : TYPE
        DESCRIPTION.
    dx : TYPE
        DESCRIPTION.

    Returns
    -------
    N1 : TYPE
        DESCRIPTION.
    N2 : TYPE
        DESCRIPTION.
    dx : TYPE
        DESCRIPTION.
    X1 : TYPE
        DESCRIPTION.
    X2 : TYPE
        DESCRIPTION.
    """
    x1 = np.zeros((1, N1), dtype=np.float64, order='F')
    x1[0, :] = -(N1+1)*dx/2 + np.linspace(1, N1, num=N1)*dx

    x2 = np.zeros((1, N2), dtype=np.float64, order='F')
    x2[0, :] = -(N2+1)*dx/2 + np.linspace(1, N2, num=N2)*dx

    # [X1,X2] = ndgrid(x1,x2)
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')

    # Now array subscripts are equivalent with Cartesian coordinates
    # x1 axis points downwards and x2 axis is in horizontal direction
    return X1, X2


def input_disc_per_lambda(sample_count):
    """
    CHECK
    """
    # help(solver_func.NR)
    return sample_count


def ITERBiCGSTABw(u_inc, CHI, Errcri, N1, N2, b, FFTG, itmax, x0=None):
    """
    Solve integral equation for contrast source with FFT
    BiCGSTAB_FFT scheme for contrast source integral equation Aw = b
    """
    from scipy.sparse.linalg import bicgstab, LinearOperator
    from keras.metrics import MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError

    N = CHI.flatten('F').shape[0]

    if x0 is None:
        # Create an array of zeros
        x0 = np.zeros(b.shape, dtype=np.complex128, order='F')
    # else:
    #     from keras.models import load_model
    #     model = load_model(x0)
    #     model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()])

    #     x0 = np.zeros(b.shape, dtype=np.complex128, order='F')
    #     x0_2D = np.squeeze(model.predict(np.abs(CHI.reshape(-1, 1, N1, N2))))
    #     x0[0:N, 0] = x0_2D.copy().flatten('F')

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
        if callback.iter % 50 == 0:
            print("iter: ", callback.iter)

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


def itmax(CHI):
    """
    CHECK
    Parameters
    ----------
    CHI : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    return CHI.flatten('F').shape[0]


def Kop(v, FFTG):
    """
    CHECK
    Parameters
    ----------
    v : TYPE
        DESCRIPTION.
    FFTG : TYPE
        DESCRIPTION.

    Returns
    -------
    Kv : TYPE
        DESCRIPTION.
    """
    # Make FFT grid
    N1, N2 = v.shape
    Cv = np.zeros(FFTG.shape, dtype=np.complex128, order='F')
    Cv[0:N1, 0:N2] = v.copy()
    # Convolution by FFT
    Cv = np.fft.fftn(Cv)
    Cv = np.fft.ifftn(FFTG * Cv)
    Kv = np.zeros((N1, N2), dtype=np.complex128, order='F')
    Kv[0:N1, 0:N2] = Cv[0:N1, 0:N2]
    return Kv


def lambda_smallest(materials_master, f):
    return np.min(materials_master['cr']) / f


def mu0():
    return 4.0 * np.pi * 1.0e-7


def N1(sample_count_x1):
    """
    Parameters
    ----------
    input_ : integer
        Number of samples in x_1

    Returns
    -------
    input_ : integer
        Number of samples in x_1
    """
    return sample_count_x1


def N2(sample_count_x2):
    """
    Parameters
    ----------
    input_ : integer
        Number of samples in x_2

    Returns
    -------
    input_ : integer
        Number of samples in x_2
    """
    return sample_count_x2


def NR(recievers_count):
    """
    CHECK
    """
    # help(solver_func.NR)
    return recievers_count


def plotContrastSource(w, CHI, X1, X2):
    import matplotlib.pyplot as plt
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
    im2 = ax2.imshow(abs(w), extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax2.set_xlabel('x_2 \u2192')
    ax2.set_ylabel('\u2190 x_1')
    ax2.set_aspect('equal', adjustable='box')
    fig.colorbar(im2, ax=ax2, orientation='horizontal')
    ax2.set_title(r'$|w|$', fontsize=13)
    plt.show()


def prescient2DL_data(data_folder, sample_list, N1, N2):
    # u_inc layer, CHI layer, w_o layer
    x_list = []
    y_list = []
    for file in sample_list:
        data = np.load(os.path.join(data_folder, file))
        # input_data = data[0:6, :, :]
        input_data = data[5, :, :]
        # print("input_data.shape", input_data.shape)
        # output_data = data[6:9, :, :]
        output_data = data[8, :, :]
        # print("output_data.shape", output_data.shape)
        x_list.append(input_data)
        y_list.append(output_data)

        # CANNOT AUGMENT WITHOUT SUPPLYING THE INCIDENT WAVE
        # for k in range(1, 4):
        #     x_list.append(np.rot90(input_data, k))
        #     y_list.append(np.rot90(output_data, k))
        # x_list.append(np.fliplr(input_data))
        # y_list.append(np.fliplr(output_data))
        # x_list.append(np.flipud(input_data))
        # y_list.append(np.flipud(output_data))

    x_list = np.array(x_list)
    y_list = np.array(y_list)
    # Step 2: Reshape the data
    x_list = np.reshape(x_list, (x_list.shape[0], 1, N1, N2))
    y_list = np.reshape(y_list, (y_list.shape[0], 1, N1, N2))

    return x_list, y_list


def R(X1, X2):
    """
    CHECK
    Parameters
    ----------
    X1 : TYPE
        DESCRIPTION.
    X2 : TYPE
        DESCRIPTION.

    Returns
    -------
    R : TYPE
        DESCRIPTION.
    """
    R = np.sqrt(X1**2 + X2**2)
    return R


def radius_receiver(origin_displacement_circle):
    """
    CHECK
    """
    return origin_displacement_circle


def radius_source(negative_x1_distance):
    """
    CHECK
    """
    return negative_x1_distance


def rcvr_phi(NR):
    """
    CHECK
    """
    rcvr_phi = np.zeros((1, NR), dtype=np.float64, order='F')
    rcvr_phi[0, 0:NR] = np.arange(1, NR+1, 1) * 2.0 * np.pi / NR
    return rcvr_phi


def s(f, angular_frequency):
    """
    This takes the input for the carrier temporal frequency in Hz and returns the wavelength the complex Laplace Parameter.
    Args:
        f (float): the carrier temporal frequency in Hz.

    Returns:
        complex: the Laplace Parameter.
    """
    # help(solver_func.s)
    return 1e-16 - 1j * angular_frequency


def tissuePermittivity(mtls, f, epsilon0):
    # This is based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5879051/ @0.5e9 Hz only.
    # Only the absolute values were available so these were used as the real parts.
    mtlLib = {
        # Name                  a          b   c        d
        'vacuum':               [1.0,      0,  0.0,     0.0],
        'normal tissue':        [9.070,    0,  0.245,   0.0],
        'benign tumor':         [24.842,   0,  0.279,   0.0],
        'cancer':               [66.696,   0,  1.697,   0.0],
    }
    # fcGHz = f/1e9
    value = mtlLib.get(mtls)
    if value:
        # epsilon = [mtlParams{libIdx, 1}] .* (fcGHz.^[mtlParams{libIdx, 2}]);
        # epsilon = value[0] * fcGHz**value[1]
        epsilon = value[0]
        # sigma = [mtlParams{libIdx, 3}] .* (fcGHz.^[mtlParams{libIdx, 4}]);
        # sigma = value[2] * fcGHz**value[3]
        sigma = value[2]
        # complexEpsilon = epsilon - 1i*sigma/(2*pi*fc*epsilon0);
        complexEpsilon = epsilon - 1j*sigma/(2*np.pi*f*epsilon0)
        return epsilon, sigma, complexEpsilon
    return None


def u_inc(gamma_0, xS, dx, X1, X2):
    """
    CHECK
    Parameters
    ----------
    gamma_0 : TYPE
        DESCRIPTION.
    xS : TYPE
        DESCRIPTION.
    dx : TYPE
        DESCRIPTION.
    X1 : TYPE
        DESCRIPTION.
    X2 : TYPE
        DESCRIPTION.
    Returns
    -------
    u_inc : TYPE
        DESCRIPTION.
    """
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


def unet(input_shape):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate

    # Input layer
    inputs = Input(input_shape)

    # Contracting path
    conv0 = Conv2D(128, 3, activation='relu', padding='same')(inputs)
    conv0 = Conv2D(128, 3, activation='relu', padding='same')(conv0)
    pool0 = MaxPooling2D(pool_size=(1, 1))(conv0)

    conv1 = Conv2D(64, 3, activation='relu', padding='same')(pool0)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(1, 1))(conv1)

    conv2 = Conv2D(32, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(1, 1))(conv2)

    conv3 = Conv2D(16, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(16, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(1, 1))(conv3)

    # Bottom layer
    conv4 = Conv2D(8, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(8, 3, activation='relu', padding='same')(conv4)

    # Expanding path
    up5 = UpSampling2D(size=(1, 1))(conv4)
    up5 = Conv2D(16, 2, activation='relu', padding='same')(up5)
    merge5 = Concatenate(axis=-1)([conv3, up5])
    conv5 = Conv2D(16, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(16, 3, activation='relu', padding='same')(conv5)

    up6 = UpSampling2D(size=(1, 1))(conv5)
    up6 = Conv2D(32, 2, activation='relu', padding='same')(up6)
    merge6 = Concatenate(axis=-1)([conv2, up6])
    conv6 = Conv2D(32, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(32, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(1, 1))(conv6)
    up7 = Conv2D(64, 2, activation='relu', padding='same')(up7)
    merge7 = Concatenate(axis=-1)([conv1, up7])
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    up8 = UpSampling2D(size=(1, 1))(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same')(up8)
    merge8 = Concatenate(axis=-1)([conv1, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    # Output layer
    # outputs = Conv2D(60, 1)(conv7)
    outputs = Conv2D(128, 1)(conv8)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model


def unet_elu(input_shape):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate

    # Input layer
    inputs = Input(input_shape)

    # Contracting path
    conv0 = Conv2D(128, 3, activation='elu', padding='same')(inputs)
    conv0 = Conv2D(128, 3, activation='elu', padding='same')(conv0)
    pool0 = MaxPooling2D(pool_size=(1, 1))(conv0)

    conv1 = Conv2D(64, 3, activation='elu', padding='same')(pool0)
    conv1 = Conv2D(64, 3, activation='elu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(1, 1))(conv1)

    conv2 = Conv2D(32, 3, activation='elu', padding='same')(pool1)
    conv2 = Conv2D(32, 3, activation='elu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(1, 1))(conv2)

    conv3 = Conv2D(16, 3, activation='elu', padding='same')(pool2)
    conv3 = Conv2D(16, 3, activation='elu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(1, 1))(conv3)

    # Bottom layer
    conv4 = Conv2D(8, 3, activation='elu', padding='same')(pool3)
    conv4 = Conv2D(8, 3, activation='elu', padding='same')(conv4)

    # Expanding path
    up5 = UpSampling2D(size=(1, 1))(conv4)
    up5 = Conv2D(16, 2, activation='elu', padding='same')(up5)
    merge5 = Concatenate(axis=-1)([conv3, up5])
    conv5 = Conv2D(16, 3, activation='elu', padding='same')(merge5)
    conv5 = Conv2D(16, 3, activation='elu', padding='same')(conv5)

    up6 = UpSampling2D(size=(1, 1))(conv5)
    up6 = Conv2D(32, 2, activation='elu', padding='same')(up6)
    merge6 = Concatenate(axis=-1)([conv2, up6])
    conv6 = Conv2D(32, 3, activation='elu', padding='same')(merge6)
    conv6 = Conv2D(32, 3, activation='elu', padding='same')(conv6)

    up7 = UpSampling2D(size=(1, 1))(conv6)
    up7 = Conv2D(64, 2, activation='elu', padding='same')(up7)
    merge7 = Concatenate(axis=-1)([conv1, up7])
    conv7 = Conv2D(64, 3, activation='elu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='elu', padding='same')(conv7)

    up8 = UpSampling2D(size=(1, 1))(conv7)
    up8 = Conv2D(128, 2, activation='elu', padding='same')(up8)
    merge8 = Concatenate(axis=-1)([conv1, up8])
    conv8 = Conv2D(128, 3, activation='elu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='elu', padding='same')(conv8)

    # Output layer
    # outputs = Conv2D(60, 1)(conv7)
    outputs = Conv2D(128, 1)(conv8)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model


def w(ITERBiCGSTABw):
    return ITERBiCGSTABw[0]


def wavelength(c_0, f):
    """
    This takes the input for the carrier temporal frequency in Hz and wave speed in embedding in meters per second and returns the wavelength.
    Args:
        c_0 (float): the wave speed in embedding in meters per second .
        f (float): the carrier temporal frequency in Hz.

    Returns:
        float: the wavelength in meters.
    """
    # help(solver_func.wavelength)
    return c_0 / f


def WavefieldSctCircle(c_0, c_sct, gamma_0, xS, NR, rcvr_phi, xR, N1, N2, dx, X1, X2, FFTG, a, CHI, Errcri):
    """
    CHECK

    Parameters
    ----------
    c_0 : TYPE
        DESCRIPTION.
    c_sct : TYPE
        DESCRIPTION.
    gamma_0 : TYPE
        DESCRIPTION.
    xS : TYPE
        DESCRIPTION.
    NR : TYPE
        DESCRIPTION.
    rcvr_phi : TYPE
        DESCRIPTION.
    xR : TYPE
        DESCRIPTION.
    N1 : TYPE
        DESCRIPTION.
    N2 : TYPE
        DESCRIPTION.
    dx : TYPE
        DESCRIPTION.
    X1 : TYPE
        DESCRIPTION.
    X2 : TYPE
        DESCRIPTION.
    FFTG : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    CHI : TYPE
        DESCRIPTION.
    Errcri : TYPE
        DESCRIPTION.

    Returns
    -------
    data2D : TYPE
        DESCRIPTION.

    """
    from scipy.special import kv, iv
    import os

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
    # angle = rcvr_phi * 180 / np.pi
    # displayDataBesselApproach(data2D, angle)
    np.savez('data2D.npz', data=data2D)
    return data2D


def X1(initGrid):
    """
    CHECK

    Parameters
    ----------
    initGrid : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return initGrid[0]


def X2(initGrid):
    """
    CHECK

    Parameters
    ----------
    initGrid : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return initGrid[1]


def X1fft(initFFTGreenGrid):
    """
    CHECK

    Parameters
    ----------
    initGrid : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return initFFTGreenGrid[0]


def X2fft(initFFTGreenGrid):
    """
    CHECK

    Parameters
    ----------
    initGrid : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return initFFTGreenGrid[1]


def xR(radius_receiver, rcvr_phi, NR):
    """
    # Receiver Positions
    # Add location of receiver
    """
    xR = np.zeros((2, NR), dtype=np.float64, order='F')
    xR[0, 0:NR] = radius_receiver * np.cos(rcvr_phi)
    xR[1, 0:NR] = radius_receiver * np.sin(rcvr_phi)
    return xR


def xS(radius_source):
    """
    Source Position
    Add location of source
    """
    xS = np.zeros((1, 2), dtype=np.float64, order='F')
    xS[0, 0] = -radius_source
    xS[0, 1] = 0.0
    return xS
