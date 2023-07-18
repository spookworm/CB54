import matplotlib.pyplot as plt
import numpy as np
import random
import os
from scipy.sparse.linalg import bicgstab, LinearOperator
import time
from scipy.special import kv, iv


def unet(input_shape):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate

    # Input layer
    inputs = Input(input_shape)

    # Contracting path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
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

    # Output layer
    # outputs = Conv2D(60, 1)(conv7)
    outputs = Conv2D(60, 1)(conv7)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model


def prescient2DL_data(data_folder, field, train_list, val_list, test_list):
    x_train = []
    y_train = []
    for file in train_list:
        data = np.load(os.path.join(data_folder, file))
        input_data = np.abs(data[0, :, :])
        if field == "real":
            # output_data = np.stack((np.real(data[2, :, :]), np.imag(data[2, :, :])), axis=0)
            output_data = np.real(data[2, :, :])
        elif field == "imag":
            # output_data = np.stack((np.real(data[2, :, :]), np.imag(data[2, :, :])), axis=0)
            output_data = np.imag(data[2, :, :])
        elif field == "abs":
            # output_data = np.stack((np.real(data[2, :, :]), np.imag(data[2, :, :])), axis=0)
            output_data = np.abs(data[2, :, :])
        x_train.append(input_data)
        y_train.append(output_data)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    # Step 2: Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], 1, 60, 60))
    y_train = np.reshape(y_train, (y_train.shape[0], 1, 60, 60))

    x_test = []
    y_test = []
    for file in test_list:
        data = np.load(os.path.join(data_folder, file))
        input_data = np.abs(data[0, :, :])
        if field == "real":
            # output_data = np.stack((np.real(data[2, :, :]), np.imag(data[2, :, :])), axis=0)
            output_data = np.real(data[2, :, :])
        elif field == "imag":
            # output_data = np.stack((np.real(data[2, :, :]), np.imag(data[2, :, :])), axis=0)
            output_data = np.imag(data[2, :, :])
        elif field == "abs":
            # output_data = np.stack((np.real(data[2, :, :]), np.imag(data[2, :, :])), axis=0)
            output_data = np.abs(data[2, :, :])
        x_test.append(input_data)
        y_test.append(output_data)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    # Step 2: Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], 1, 60, 60))
    y_test = np.reshape(y_test, (y_test.shape[0], 1, 60, 60))

    x_val = []
    y_val = []
    for file in val_list:
        data = np.load(os.path.join(data_folder, file))
        input_data = np.abs(data[0, :, :])
        if field == "real":
            # output_data = np.stack((np.real(data[2, :, :]), np.imag(data[2, :, :])), axis=0)
            output_data = np.real(data[2, :, :])
        elif field == "imag":
            # output_data = np.stack((np.real(data[2, :, :]), np.imag(data[2, :, :])), axis=0)
            output_data = np.imag(data[2, :, :])
        elif field == "abs":
            # output_data = np.stack((np.real(data[2, :, :]), np.imag(data[2, :, :])), axis=0)
            output_data = np.abs(data[2, :, :])
        x_val.append(input_data)
        y_val.append(output_data)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    # Step 2: Reshape the data
    x_val = np.reshape(x_val, (x_val.shape[0], 1, 60, 60))
    y_val = np.reshape(y_val, (y_val.shape[0], 1, 60, 60))

    return x_train, y_train, x_test, y_test, x_val, y_val


def tissuePermittivity(mtls, fc):
    # This is based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5879051/ @0.5e9 Hz only.
    # Only the absolute values were available so these were used as the real parts.
    epsilon0 = 8.854187817e-12
    mtlLib = {
        # Name                  a          b   c        d
        'vacuum':               [1.0,      0,  0.0,     0.0],
        'normal tissue':        [9.070,    0,  0.245,   0.0],
        'benign tumor':         [24.842,   0,  0.279,   0.0],
        'cancer':               [66.696,   0,  1.697,   0.0],
    }
    fcGHz = fc/1e9
    value = mtlLib.get(mtls)
    if value:
        # epsilon = [mtlParams{libIdx, 1}] .* (fcGHz.^[mtlParams{libIdx, 2}]);
        epsilon = value[0] * fcGHz**value[1]
        # sigma = [mtlParams{libIdx, 3}] .* (fcGHz.^[mtlParams{libIdx, 4}]);
        sigma = value[2] * fcGHz**value[3]
        # complexEpsilon = epsilon - 1i*sigma/(2*pi*fc*epsilon0);
        complexEpsilon = epsilon - 1j*sigma/(2*np.pi*fc*epsilon0)
        return epsilon, sigma, complexEpsilon
    return None


def generate_random_circles(N1, N2, radius_min_pix, radius_max_pix, seedling, seed_count, subfolder):
    from skimage import io
    import matplotlib.cm as cm
    for seed in range(seedling, seedling+seed_count):

        shape_array = np.zeros((N2, N1))
        radius = random.uniform(radius_min_pix, radius_max_pix)
        center_x = random.uniform(radius, N1 - radius)
        center_y = random.uniform(radius, N2 - radius)

        for i in range(N2):
            for j in range(N1):
                if np.sqrt((j - center_x) ** 2 + (i - center_y) ** 2) <= radius:
                    shape_array[i, j] = 1

        # Only use as visualisation, not input data. Use the npy array as input data to avoid clipping etc.
        colored_image = cm.gray(shape_array.T)
        image_uint8 = (colored_image * 255).astype(np.uint8)
        io.imsave(os.path.join(subfolder, f"instance_{str(seed).zfill(10)}.png"), image_uint8, check_contrast=False)
        np.save(os.path.join(subfolder, f"instance_{str(seed).zfill(10)}.npy"), shape_array)
        # print("np.shape(shape_array)", np.shape(shape_array))


def initEM(c_0, eps_sct, mu_sct, gamma_0, N1, N2, dx):
    # add location of source/receiver
    xS, NR, rcvr_phi, xR = initSourceReceiver(N1, dx)

    # add grid in either 1D, 2D or 3D
    # initGrid() and initGridEM() are equivalent
    X1, X2 = initGrid(N1, N2, dx)

    # compute FFT of Green function
    FFTG = initFFTGreen(N1, N2, dx, gamma_0)

    Errcri = 1e-10
    return xS, NR, rcvr_phi, xR, X1, X2, FFTG, Errcri


def initSourceReceiver(N1, dx):
    # Source Position
    xS = np.zeros((1, 2), dtype=np.float64, order='F')
    # xS[0, 0] = -170.0
    xS[0, 0] = -N1*dx
    xS[0, 1] = 0.0

    # Receiver Positions
    NR = 180
    rcvr_phi = np.zeros((1, NR), dtype=np.float64, order='F')
    # rcvr_phi[0, :] = np.linspace(1, NR, num=NR)*(2*np.pi)/NR
    rcvr_phi[0, 0:NR] = np.arange(1, NR+1, 1) * 2.0 * np.pi / NR

    xR = np.zeros((2, NR), dtype=np.float64, order='F')
    xR[0, 0:NR] = (N1*dx) * np.cos(rcvr_phi)
    xR[1, 0:NR] = (N1*dx) * np.sin(rcvr_phi)
    return xS, NR, rcvr_phi, xR


def initGrid(N1, N2, dx):
    x1 = np.zeros((1, N1), dtype=np.float64, order='F')
    x1[0, :] = -(N1+1)*dx/2 + np.linspace(1, N1, num=N1)*dx

    x2 = np.zeros((1, N2), dtype=np.float64, order='F')
    x2[0, :] = -(N2+1)*dx/2 + np.linspace(1, N2, num=N2)*dx

    # [X1,X2] = ndgrid(x1,x2)
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')

    # Now array subscripts are equivalent with Cartesian coordinates
    # x1 axis points downwards and x2 axis is in horizontal direction
    return X1, X2


def initFFTGreen(N1, N2, dx, gamma_0):
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

    def Green(dx, gamma_0):
        from scipy.special import kv, iv
        gam0 = gamma_0
        X1 = X1fft
        X2 = X2fft
        DIS = np.sqrt(X1**2 + X2**2)
        # avoid Green's singularity for DIS = 0
        DIS[0, 0] = 1
        G = 1/(2*np.pi) * kv(0, gam0*DIS)
        # radius circle with area of dx^2
        delta = (np.pi)**(-1/2) * dx
        factor = 2 * iv(1, gam0*delta) / (gam0*delta)
        # integral includes gam0^2
        IntG = (gam0**2 * dx**2) * factor**2 * G
        IntG[0, 0] = 1 - gam0*delta * kv(1, gam0*delta) * factor
        return IntG

    # compute gam_0^2 * subdomain integrals  of Green function
    IntG = Green(dx, gamma_0)

    # apply n-dimensional Fast Fourier transform
    FFTG = np.fft.fftn(IntG)
    return FFTG


def initEMContrast(eps_sct, mu_sct, X1, X2, geometry_file=None):
    if geometry_file is None:
        # half width slab / radius circle cylinder / radius sphere
        a = 40
        R = np.sqrt(X1**2 + X2**2)
        # (1) Compute permittivity contrast
        CHI_eps = (1-eps_sct) * (R < a)
        # (2) Compute permeability contrast
        CHI_mu = (1-mu_sct) * (R < a)
    else:
        array = np.load(geometry_file, mmap_mode='r')
        # (1) Compute permittivity contrast
        CHI_eps = (1-eps_sct) * array.T

        # (2) Compute permeability contrast
        CHI_mu = (1-mu_sct) * array.T
        a = None
    return a, CHI_eps, CHI_mu


def plotEMContrast(CHI_eps, CHI_mu, X1, X2):
    # Plot 2D contrast/source distribution
    # x1 = ForwardBiCGSTABFFT.input.X1(:, 1);
    x1 = X1[:, 0]
    # x2 = ForwardBiCGSTABFFT.input.X2(1, :);
    x2 = X2[0, :]

    fig = plt.figure(figsize=(7.09, 4.72))
    fig.subplots_adjust(wspace=0.3)

    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(np.fliplr(CHI_eps), extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax1.set_xlabel('x_2 \u2192')
    ax1.set_ylabel('\u2190 x_1')
    ax1.set_aspect('equal', adjustable='box')
    fig.colorbar(im1, ax=ax1, orientation='horizontal')
    ax1.set_title(r'$\chi^\epsilon = 1 - \epsilon_{sct}/\epsilon_{0}$', fontsize=13)

    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(np.fliplr(CHI_mu), extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax2.set_xlabel('x_2 \u2192')
    ax2.set_ylabel('\u2190 x_1')
    ax2.set_aspect('equal', adjustable='box')
    fig.colorbar(im2, ax=ax2, orientation='horizontal')
    ax2.set_title(r'$\chi^\mu = 1 - \mu_{sct}/\mu_{0}$', fontsize=13)
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
    # io.imsave('output.png', colored_array)

    # Plot the image
    plt.imshow(colored_array)

    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('Value')

    # Display the plot
    plt.show()


def IncEMwave(gamma_0, xS, dx, X1, X2):
    # incident wave from electric dipole in negative x_1

    # radius circle with area of dx^2
    delta = (np.pi)**(-1/2) * dx
    factor = 2 * iv(1, gamma_0*delta) / (gamma_0*delta)

    X1 = X1-xS[0, 0]
    X2 = X2-xS[0, 1]
    DIS = np.sqrt(X1**2 + X2**2)
    X2 = X2/DIS
    X1 = X1/DIS

    G = factor * 1/(2*np.pi) * kv(0, gamma_0*DIS)
    dG = - factor * gamma_0 * 1/(2*np.pi) * kv(1, gamma_0*DIS)
    dG11 = (2 * X1 * X1 - 1) * (-dG/DIS) + gamma_0**2 * X1 * X1 * G
    dG21 = (2 * X2 * X1 - 0) * (-dG/DIS) + gamma_0**2 * X2 * X1 * G

    E_inc = np.zeros((3, X1.shape[0], X1.shape[1]), dtype=np.complex128, order='F')
    E_inc[0, :] = -(-gamma_0**2 * G + dG11)
    E_inc[1, :] = - dG21
    E_inc[2, :] = 0

    ZH_inc = np.zeros((3, X1.shape[0], X1.shape[1]), dtype=np.complex128, order='F')
    ZH_inc[0, :] = 0
    ZH_inc[1, :] = 0
    ZH_inc[2, :] = gamma_0 * X2 * dG
    return E_inc, ZH_inc


def ITERBiCGSTABwE(E_inc, CHI_eps, Errcri, N1, N2, dx, FFTG, gamma_0, x0=None):
    # BiCGSTAB_FFT scheme for contrast source integral equation Aw = b
    # Known 1D vector right-hand side
    N = CHI_eps.flatten('F').shape[0]

    itmax = N

    b = np.zeros((2*N, 1), dtype=np.complex128, order='F')
    b[0:N, 0] = CHI_eps.flatten('F') * E_inc[0, :].flatten('F')
    b[N:2*N, 0] = CHI_eps.flatten('F') * E_inc[1, :].flatten('F')

    if x0 == 0:
        # Create an array of zeros
        x0 = np.zeros(b.shape, dtype=np.complex128, order='F')
    else:
        from keras.models import load_model
        model_re = load_model('model_re.keras')
        model_im = load_model('model_im.keras')
        # x0 = np.concatenate([w_E_o[0, :, :].flatten('F'), w_E_o[1, :, :].flatten('F')], axis=0)
        x0 = np.zeros(b.shape, dtype=np.complex128, order='F')
        x0_2D = np.squeeze(model_re.predict(np.real(CHI_eps.reshape(-1, 1, N1, N2))) + 1j*model_im.predict(np.imag(CHI_eps.reshape(-1, 1, N1, N2))))
        x0[0:N, 0] = x0_2D.copy().flatten('F')

    def custom_matvec(w):
        return Aw(w, N1, N2, dx, FFTG, CHI_eps, gamma_0)

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
    callback.iter = 0.0
    # callback.information = np.array([[callback.iter, np.linalg.norm(b), time.time() - callback.start_time]])
    callback.information = np.array([[callback.iter, np.linalg.norm(Aw_operator(x0).T - b.T), time.time() - callback.start_time]])
    # callback.information = np.empty((1, 3))

    # Call bicgstab with the LinearOperator instance and other inputs
    w, exit_code = bicgstab(Aw_operator, b, x0, tol=Errcri, maxiter=itmax, callback=callback)

    # Output Matrix UPDATE
    # w = w.reshape((N1, N2), order='F')
    w_E = vector2matrix(w, N1, N2)
    return w_E, exit_code, callback.information


def Aw(w, N1, N2, dx, FFTG, CHI_eps, gamma_0):
    N = CHI_eps.flatten('F').shape[0]
    # Convert 1D vector to matrix
    w_E = vector2matrix(w, N1, N2)
    Kw_E = KopE(w_E, gamma_0, N1, N2, dx, FFTG)
    # Convert matrix to 1D vector
    y = np.zeros((2*N1*N2, 1), dtype=np.complex_, order='F')
    y[0:N, 0] = w_E[0, :, :].flatten('F') - (CHI_eps.flatten('F') * Kw_E[0, :, :].flatten('F'))
    y[N:2*N, 0] = w_E[1, :, :].flatten('F') - (CHI_eps.flatten('F') * Kw_E[1, :, :].flatten('F'))
    return y


def vector2matrix(w, N1, N2):
    # Modify vector output from 'bicgstab' to matrix for further computations
    # N = CHI_eps.flatten('F').shape[0]
    N = N1 * N2
    DIM = [N1, N2]
    w_E = np.zeros((2, N1, N2), dtype=np.complex128, order='F')
    w_E[0, :, :] = np.reshape(w[0:N], DIM, order='F')
    w_E[1, :, :] = np.reshape(w[N:2*N], DIM, order='F')
    return w_E


def KopE(wE, gamma_0, N1, N2, dx, FFTG):
    wE = wE.copy()
    KwE = np.zeros((2, N1, N2), dtype=np.complex128, order='F')
    for n in range(0, 2):
        KwE[n, :, :] = Kop(wE[n, :, :], FFTG)
    # dummy is temporary storage
    dummy = np.zeros((2, N1, N2), dtype=np.complex128, order='F')
    dummy[:, :, :] = graddiv(KwE, dx, N1, N2)
    # print((graddiv(KwE, dx, N1, N2)).shape)
    for n in range(0, 2):
        KwE[n, :, :] = KwE[n, :, :] - dummy[n, :, :] / gamma_0**2
    return KwE


def Kop(v, FFTG):
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


def graddiv(v, dx, N1, N2):
    # Anywhere where there is swapping there could be inheritance issues so use copy.
    v = v.copy()
    u = np.zeros((v.shape), dtype=np.complex_, order='F')

    # % Compute d1d1_v1, d2d2_v2
    # u{1}(2:N1 - 1, :) = v{1}(1:N1 - 2, :) - 2 * v{1}(2:N1 - 1, :) + v{1}(3:N1, :);
    u[0, 1:N1 - 1, :] = v[0, 0:N1 - 2, :] - 2 * v[0, 1:N1 - 1, :] + v[0, 2:N1, :]

    # u{2}(:, 2:N2 - 1) = v{2}(:, 1:N2 - 2) - 2 * v{2}(:, 2:N2 - 1) + v{2}(:, 3:N2);
    u[1, :, 1:N2 - 1] = v[1, :, 0:N2 - 2] - 2 * v[1, :, 1:N2 - 1] + v[1, :, 2:N2]

    # % Replace the input vector v1 by d1_v and v2 by d2_v2 ---------------------
    # v{1}(2:N1 - 1, :) = (v{1}(3:N1, :) - v{1}(1:N1 - 2, :)) / 2; % d1_v1
    v[0, 1:N1 - 1, :] = (v[0, 2:N1, :] - v[0, 0:N1 - 2, :]) / 2.0

    # v{2}(:, 2:N2 - 1) = (v{2}(:, 3:N2) - v{2}(:, 1:N2 - 2)) / 2; % d2_v2
    v[1, :, 1:N2 - 1] = (v[1, :, 2:N2] - v[1, :, 0:N2 - 2]) / 2.0

    # % Add d1_v2 = d1d2_v2 to output vector u1 ---------------------------------
    # u{1}(2:N1 - 1, :) = u{1}(2:N1 - 1, :) + (v{2}(3:N1, :) - v{2}(1:N1 - 2, :)) / 2;
    u[0, 1:N1 - 1, :] = u[0, 1:N1 - 1, :] + (v[1, 2:N1, :] - v[1, 0:N1 - 2, :]) / 2.0

    # % Add d2_v1 = d2d1_v1 to output vector u2 ---------------------------------
    # u{2}(:, 2:N2 - 1) = u{2}(:, 2:N2 - 1) + (v{1}(:, 3:N2) - v{1}(:, 1:N2 - 2)) / 2;
    u[1, :, 1:N2 - 1] = u[1, :, 1:N2 - 1] + (v[0, :, 2:N2] - v[0, :, 0:N2 - 2]) / 2.0

    # % divide by dx^2
    u[0, :, :] = u[0, :, :] / dx**2
    u[1, :, :] = u[1, :, :] / dx**2
    return u


def plotContrastSourcewE(w_E, X1, X2):
    import matplotlib.pyplot as plt
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


def E(E_inc, E_sct):
    E = np.zeros((E_inc.shape), dtype=np.complex_, order='F')
    for n in range(0, 2):
        E[n, :, :] = E_inc[n, :, :] + E_sct[n, :, :]
    return E


def plotEtotalwavefield(E, a, X1, X2, N1, N2):
    phi = np.arange(0, 2*np.pi, 0.01)
    import matplotlib.pyplot as plt
    # Plot wave fields in two-dimensional space
    fig, axs = plt.subplots(1, 2, figsize=(18, 12))
    im1 = axs[0].imshow(abs(E[0]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet')
    axs[0].set_xlabel('x$_2$ $\\rightarrow$')
    axs[0].set_ylabel('$\\leftarrow$ x$_1$')
    axs[0].set_title('2D Electric field E1', fontsize=13)
    fig.colorbar(im1, ax=axs[0], orientation='horizontal')
    if a is not None:
        axs[0].plot(a*np.cos(phi), a*np.sin(phi), 'w')
    im2 = axs[1].imshow(abs(E[1]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet')
    axs[1].set_xlabel('x$_2$ $\\rightarrow$')
    axs[1].set_ylabel('$\\leftarrow$ x$_1$')
    axs[1].set_title('2D Electric field E2', fontsize=13)
    fig.colorbar(im2, ax=axs[1], orientation='horizontal')
    if a is not None:
        axs[1].plot(a*np.cos(phi), a*np.sin(phi), 'w')
    plt.show()
