import os
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

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


def complex_separation(complex_array):
    # Separate real and imaginary components
    real_array = np.real(complex_array)
    imaginary_array = np.imag(complex_array)

    # Compute absolute array
    absolute_array = np.abs(complex_array)

    # Stack the arrays together
    result_array = np.stack([real_array, imaginary_array, absolute_array])
    return result_array


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
    plt.legend(['Integral-equation method', 'Bessel-function method'], loc='upper center')
    plt.text(0.5*np.max(angle), 0.8*np.max(np.abs(CIS_approach)), 'Error$^{sct}$ = ' + error, color='red', ha='center', va='center')
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


def E(E_inc, E_sct):
    E = np.zeros((E_inc.shape), dtype=np.complex_, order='F')
    for n in range(0, 2):
        E[n, :, :] = E_inc[n, :, :] + E_sct[n, :, :]
    return E


def EMsctCircle():
    from scipy.special import kv, iv
    import os

    c_0, eps_sct, mu_sct, gamma_0, xS, NR, rcvr_phi, xR, N1, N2, dx, X1, X2, FFTG, a, CHI_eps, CHI_mu, Errcri = initEM(bessel=1)

    gam0 = gamma_0
    gam_sct = gam0 * np.sqrt(eps_sct*mu_sct)
    Z_sct = np.sqrt(mu_sct/eps_sct)

    # (1) Transform Cartesian coordinates to polar coordinates
    rR = np.zeros((1, xR.shape[1]), dtype=np.complex128, order='F')
    rR[0, :] = np.sqrt(xR[0, :]**2 + xR[1, :]**2)
    phiR = np.zeros((1, xR.shape[1]), dtype=np.complex128, order='F')
    phiR[0, :] = np.arctan2(xR[1, :], xR[0, :])
    rS = np.sqrt(xS[0, 0]**2 + xS[0, 1]**2)
    phiS = np.arctan2(xS[0, 1], xS[0, 0])

    # (2) Compute coefficients of Bessel series expansion
    arg0 = gamma_0 * a
    args = gam_sct*a
    # increase M for more accuracy
    M = 100

    A = np.zeros((1, M), dtype=np.complex128)
    for m in range(1, M+1):
        Ib0 = iv(m, arg0)
        # print("Ib0", Ib0)
        dIb0 = iv(m+1, arg0) + m/arg0 * Ib0
        # print("dIb0", dIb0)
        Ibs = iv(m, args)
        # print("Ibs", Ibs)
        dIbs = iv(m+1, args) + m/args * Ibs
        # print("dIbs", dIbs)
        Kb0 = kv(m, arg0)
        # print("Kb0", Kb0)
        dKb0 = -kv(m+1, arg0) + m/arg0 * Kb0
        # print("dKb0", dKb0)
        A[0, m-1] = - (Z_sct * dIbs*Ib0 - dIb0*Ibs) / (Z_sct * dIbs*Kb0 - dKb0*Ibs)

    # (3) Compute reflected Er field at receivers (data)
    Er = np.zeros(rR.shape, dtype=np.complex128, order='F')
    Ephi = np.zeros(rR.shape, dtype=np.complex128, order='F')
    ZH3 = np.zeros(rR.shape, dtype=np.complex128, order='F')
    for m in range(1, M+1):
        arg0 = gam0*rR
        Kb0 = kv(m, arg0)
        dKb0 = -kv(m+1, arg0) + m/arg0 * Kb0
        KbS = kv(m, gam0*rS)
        Er = Er + A[0, m-1]*2*m**2 * Kb0 * KbS * np.cos(m*(phiR-phiS))
        Ephi = Ephi - A[0, m-1]*2*m * dKb0 * KbS * np.sin(m*(phiR-phiS))
        ZH3 = ZH3 - A[0, m-1]*2*m * Kb0 * KbS * np.sin(m*(phiR-phiS))

    Er = 1/(2*np.pi) * Er / rR / rS
    Ephi = gam0 * 1/(2*np.pi) * Ephi / rS
    ZH3 = -gam0 * 1/(2*np.pi) * ZH3 / rS

    E = np.zeros((2, NR), dtype=np.complex128, order='F')
    E[0, :] = np.cos(phiR) * Er - np.sin(phiR) * Ephi
    E[1, :] = np.sin(phiR) * Er + np.cos(phiR) * Ephi

    Edata2D = np.zeros((1, NR), dtype=np.complex128, order='F')
    Edata2D[0, :] = np.sqrt(abs(E[0, :])**2 + abs(E[1, :])**2)
    Hdata2D = np.zeros((1, NR), dtype=np.complex128, order='F')
    Hdata2D[0, :] = abs(ZH3)

    if os.path.exists('EDATA2D.npy'):
        os.remove('EDATA2D.npy')
    np.savez('EDATA2D.npz', data=Edata2D)

    if os.path.exists('HDATA2D.npy'):
        os.remove('HDATA2D.npy')
    np.savez('HDATA2D.npz', data=Hdata2D)

    # angle = rcvr_phi * 180 / np.pi
    # displayDataBesselApproach(Edata2D, angle)
    # displayDataBesselApproach(Hdata2D, angle)
    return Edata2D, Hdata2D


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


def initEM(bessel=None):
    # Time factor = exp(-iwt)
    # Spatial units is in m
    # Source wavelet M Z_0 / gamma_0  = 1   (Z_0 M = gamma_0)

    # c_0 = 3e8
    eps_0 = 8.854e-12
    mu_0 = 4e-7*np.pi
    c_0 = 1/np.sqrt(eps_0*mu_0)
    # wave speed in embedding
    # relative permittivity of scatterer
    eps_sct = 1.75
    # eps_sct = 50
    # print("eps_sct: ", eps_sct)
    eps_d = eps_sct*eps_0
    # relative permeability of scatterer
    mu_sct = 1.0
    mu_d = mu_sct*mu_0

    c_d = 1/np.sqrt(eps_d*mu_d)
    (c_0/c_d)**2

    # temporal frequency
    f = 10e6
    # f = 10e6*50
    # wavelength
    wavelength_0 = c_0 / f
    print("wavelength_0", wavelength_0)
    wavelength_d = c_d / f
    print("wavelength_d", wavelength_d)
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

    # add contrast distribution
    # print("bessel", bessel)
    if bessel is None:
        a, CHI_eps, CHI_mu = initEMContrast(eps_sct, mu_sct, X1, X2, dx)
        # Run if wanting to create base scatter
        # a, CHI_eps, CHI_mu = initEMContrastBase(eps_sct, mu_sct, X1, X2, dx)
    else:
        a, CHI_eps, CHI_mu = initEMContrastOrig(eps_sct, mu_sct, X1, X2)

    Errcri = 1e-18
    return c_0, eps_sct, mu_sct, gamma_0, xS, NR, rcvr_phi, xR, N1, N2, dx, X1, X2, FFTG, a, CHI_eps, CHI_mu, Errcri


def initEMContrastOrig(eps_sct, mu_sct, X1, X2):
    # half width slab / radius circle cylinder / radius sphere
    a = 40
    # a = 40*(0.5**10)

    R = np.sqrt(X1**2 + X2**2)
    # Bessel-Approach Validation Cancer only
    CHI_eps = (1-eps_sct) * (R < a)

    # (2) Compute permeability contrast
    CHI_mu = (1-mu_sct) * (R < a)

    # return a, CHI_eps, CHI_mu
    return a, CHI_eps, CHI_mu


def initEMContrastBase(eps_sct, mu_sct, X1, X2, dx):
    import random
    R = np.sqrt(X1**2 + X2**2)

    # half width slab / radius circle cylinder / radius sphere
    a = 40
    a = 8.94427
    # a = 40*(0.5**10)
    a_normal = a*np.sqrt(1/0.05)
    a_cancer = a

    # (1) Compute permittivity contrast
    eps_normal = 8
    eps_normal = 1.25
    eps_cancer = eps_sct

    CHI_eps_0 = (1-eps_normal) * (R < a_normal)
    CHI_eps_1 = (1-eps_cancer+(eps_normal-1)) * (R < a_cancer)
    radius_max_pix_b = int(np.floor(a_normal/dx - 0.5*a_cancer/dx))
    CHI_eps_1 = np.roll(CHI_eps_1, shift=int(random.uniform(-radius_max_pix_b, radius_max_pix_b)), axis=0)
    CHI_eps_1 = np.roll(CHI_eps_1, shift=int(random.uniform(-radius_max_pix_b, radius_max_pix_b)), axis=1)
    CHI_eps = CHI_eps_0
    CHI_eps[R > a_normal] = 0.0

    # # Bessel-Approach Validation Cancer only
    # CHI_eps = (1-eps_sct) * (R < a)

    # (2) Compute permeability contrast
    CHI_mu = (1-mu_sct) * (R < a)
    return a, CHI_eps, CHI_mu


def initEMContrast(eps_sct, mu_sct, X1, X2, dx):
    import random
    R = np.sqrt(X1**2 + X2**2)

    # half width slab / radius circle cylinder / radius sphere
    a = 40
    a = 8.94427
    # a = 40*(0.5**10)
    a_normal = a*np.sqrt(1/0.05)
    a_cancer = a

    # (1) Compute permittivity contrast
    eps_normal = 8
    eps_normal = 1.25
    eps_cancer = eps_sct

    CHI_eps_0 = (1-eps_normal) * (R < a_normal)
    CHI_eps_1 = (1-eps_cancer+(eps_normal-1)) * (R < a_cancer)
    radius_max_pix_b = int(np.floor(a_normal/dx - 0.5*a_cancer/dx))
    CHI_eps_1 = np.roll(CHI_eps_1, shift=int(random.uniform(-radius_max_pix_b, radius_max_pix_b)), axis=0)
    CHI_eps_1 = np.roll(CHI_eps_1, shift=int(random.uniform(-radius_max_pix_b, radius_max_pix_b)), axis=1)
    CHI_eps = CHI_eps_0 + CHI_eps_1
    CHI_eps[R > a_normal] = 0.0

    # # Bessel-Approach Validation Cancer only
    # CHI_eps = (1-eps_sct) * (R < a)

    # (2) Compute permeability contrast
    CHI_mu = (1-mu_sct) * (R < a)
    return a, CHI_eps, CHI_mu


def IncEMwave(gamma_0, xS, dx, X1, X2):
    from scipy.special import kv, iv
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
    dG21 =  2 * X2 * X1      * (-dG/DIS) + gamma_0**2 * X2 * X1 * G

    E_inc = np.zeros((3, X1.shape[0], X1.shape[1]), dtype=np.complex128, order='F')
    E_inc[0, :] = -(-gamma_0**2 * G + dG11)
    E_inc[1, :] = - dG21
    E_inc[2, :] = 0

    ZH_inc = np.zeros((3, X1.shape[0], X1.shape[1]), dtype=np.complex128, order='F')
    ZH_inc[0, :] = 0
    ZH_inc[1, :] = 0
    ZH_inc[2, :] = gamma_0 * X2 * dG
    return E_inc, ZH_inc


def info_data_harvest(input_folder):
    # Initialize an empty list to store the loaded arrays
    final_rows = []
    # Add headers to the new array
    headers = ["Name", "Iteration_Count", "Error_Final", "Duration", "Error_Initial", "Model Flag", "Duration_Log"]
    # final_rows.append(headers)

    # Iterate through each file in the folder
    # info_files = [f for f in os.listdir(input_folder) if "_info_" in f]
    info_files_o = [f for f in os.listdir(input_folder) if "_info_" in f and "_o.n" in f]
    info_files_m = [f for f in os.listdir(input_folder) if "_info_" in f and "_m.n" in f]
    info_files_o_m = []
    for f in info_files_m:
        new_f = f.replace("_m.n", "_o.n")
        info_files_o_m.append(new_f)
    info_files_O_m = list(set(info_files_o) & set(info_files_o_m))
    info_files_o_M = []
    for f in info_files_O_m:
        new_f = f.replace("_o.n", "_m.n")
        info_files_o_M.append(new_f)
    info_files_O_M = info_files_O_m + info_files_o_M
    for filename in info_files_O_M:
        # Construct the full file path
        file_path = os.path.join(input_folder, filename)

        # Load the array from the file
        array = np.load(file_path)
        # Extract the final row
        final_row = array[-1]

        # Extract the second column of the first row
        second_column = array[0][1]

        # Add another column with the name of the original array
        final_row_with_name = [f'{filename}'] + list(map(str, final_row))

        # Add the second column to the final row
        final_row_with_second_column = final_row_with_name + [str(second_column)]

        # Indicator model flag
        final_row_with_flag = final_row_with_second_column + [str(filename[-5])]

        # Calculate the Duration_Log
        final_row_complete = final_row_with_flag + [str(np.log(final_row[2]))]

        # Append the final row to the new array
        final_rows.append(final_row_complete)

    # Specify the output file path
    # last_part = os.path.basename(os.path.normpath(input_folder))
    output_file = '.\\doc\\_stats\\dataset_instances_output.csv'
    final_rows = pd.DataFrame(final_rows, columns=headers)
    final_rows.to_csv(output_file, index=False)
    return final_rows


def info_data_paired(input_folder):
    df = pd.read_csv(input_folder, header=0)
    df['Short_Name'] = df['Name'].str[:19]
    df = df.drop('Name', axis=1)

    # Create a DataFrame with distinct values of "Short_Name"
    df_paired = pd.DataFrame(df["Short_Name"].unique(), columns=["Short_Name"])

    for flag in ["o", "m"]:
        filtered_df = df[df["Model Flag"] == flag]
        result = df_paired.merge(filtered_df, on="Short_Name", how="left")
        new_column_names = {column: column + '_' + flag for column in result.columns if column != "Short_Name" and "_o" not in column}
        result = result.rename(columns=new_column_names)
        df_paired = result.drop('Model Flag_' + flag, axis=1)

    # for column in df.columns:
    #     # Selecting the desired columns from base1
    #     base1 = df[df['Model Flag'] == 'o'][['Short_Name', column]].rename(columns={column: column + '_o'})
    #     # Selecting the desired columns from base2
    #     base2 = df[df['Model Flag'] == 'm'][['Short_Name', column]].rename(columns={column: column + '_m'})
    #     # Merging base1 and base2 on the 'Name' column
    #     result = base1.merge(base2, on='Short_Name', how='left')

    # Open the file in write mode
    last_part = os.path.basename(os.path.normpath(input_folder))
    output_file = os.path.dirname(input_folder) + '\\paired_' + str(last_part)
    # result.to_csv(output_file, index=False)
    df_paired.to_csv(output_file, index=False)
    # return result


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


def initGrid():
    # number of samples in x_1
    N1 = 120
    N1 = 128
    # number of samples in x_2
    N2 = 100
    N2 = 128
    # with meshsize dx
    dx = 2
    # dx = 2*(0.5**8.38)

    # print("lamda_0", wavelength_0/dx)
    # print("lamda_d", wavelength_d/dx)

    x1 = np.zeros((1, N1), dtype=np.float64, order='F')
    x1[0, :] = -(N1+1)*dx/2 + np.linspace(1, N1, num=N1)*dx

    x2 = np.zeros((1, N2), dtype=np.float64, order='F')
    x2[0, :] = -(N2+1)*dx/2 + np.linspace(1, N2, num=N2)*dx

    # [X1,X2] = ndgrid(x1,x2)
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')

    # Now array subscripts are equivalent with Cartesian coordinates
    # x1 axis points downwards and x2 axis is in horizontal direction
    return N1, N2, dx, X1, X2


def initSourceReceiver():
    source_distance = 170

    # Source Position
    xS = np.zeros((1, 2), dtype=np.float64, order='F')
    xS[0, 0] = -source_distance
    xS[0, 1] = 0.0

    # Receiver Positions
    NR = 180
    reciever_source_coefficient = (15/17)
    # reciever_source_coefficient = (0.25/17)
    rcvr_phi = np.zeros((1, NR), dtype=np.float64, order='F')
    # rcvr_phi[0, :] = np.linspace(1, NR, num=NR)*(2*np.pi)/NR
    rcvr_phi[0, 0:NR] = np.arange(1, NR+1, 1) * 2.0 * np.pi / NR
    xR = np.zeros((2, NR), dtype=np.float64, order='F')
    xR[0, 0:NR] = source_distance * reciever_source_coefficient * np.cos(rcvr_phi)
    xR[1, 0:NR] = source_distance * reciever_source_coefficient * np.sin(rcvr_phi)
    return xS, NR, rcvr_phi, xR


def ITERBiCGSTABwE(E_inc, CHI_eps, Errcri, N1, N2, dx, FFTG, gamma_0, x0=None):
    from scipy.sparse.linalg import bicgstab, LinearOperator

    # BiCGSTAB_FFT scheme for contrast source integral equation Aw = b
    itmax = 1000
    # Known 1D vector right-hand side
    N = CHI_eps.flatten('F').shape[0]

    b = np.zeros((2*N, 1), dtype=np.complex128, order='F')
    b[0:N, 0] = CHI_eps.flatten('F') * E_inc[0, :].flatten('F')
    b[N:2*N, 0] = CHI_eps.flatten('F') * E_inc[1, :].flatten('F')

    if x0 is None:
        # Create an array of zeros
        x0 = np.zeros(b.shape, dtype=np.complex128, order='F')
        x0[0:N, 0] = np.multiply(CHI_eps, E_inc[0, :, :]).flatten('F')
        x0[N:2*N, 0] = np.multiply(CHI_eps, E_inc[1, :, :]).flatten('F')

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
    callback.iter = 0
    # callback.information = np.array([[callback.iter, np.linalg.norm(b), time.time() - callback.start_time]])
    callback.information = np.array([[callback.iter, np.linalg.norm(Aw_operator(x0).T - b.T), time.time() - callback.start_time]])
    # callback.information = np.empty((1, 3))

    # Call bicgstab with the LinearOperator instance and other inputs
    w, exit_code = bicgstab(Aw_operator, b, x0=x0, tol=Errcri, maxiter=itmax, callback=callback)
    print("iter max: ", callback.iter)

    # Output Matrix UPDATE
    # w = w.reshape((N1, N2), order='F')
    w_E = vector2matrix(w, N1, N2)
    return w_E, exit_code, callback.information


def keras_format(original_array):
    original_array = np.transpose(original_array, (1, 2, 0))
    keras_array = np.expand_dims(original_array, axis=0)
    return keras_array


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


def mat_checker(variable, var_name):
    import scipy.io
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


def mat_loader(var_name):
    import scipy.io
    # x1fft_m = mat_loader(nameof(x1fft))
    var_name_matlab = scipy.io.loadmat(var_name + '.mat')
    var_name_m = np.array(var_name_matlab[var_name], dtype=np.complex_)
    return var_name_m


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


def plotEMContrast(CHI_eps, CHI_mu, X1, X2):
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

    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(CHI_mu, extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax2.set_xlabel('x_2 \u2192')
    ax2.set_ylabel('\u2190 x_1')
    ax2.set_aspect('equal', adjustable='box')
    fig.colorbar(im2, ax=ax2, orientation='horizontal')
    ax2.set_title(r'$\chi^\mu = 1 - \mu_{sct}/\mu_{0}$', fontsize=13)
    plt.show()


def plotEtotalwavefield(field_name, field, a, X1, X2, N1, N2):
    # phi = np.arange(0, 2*np.pi, 0.01)
    import matplotlib.pyplot as plt
    # Plot wave fields in two-dimensional space
    fig, axs = plt.subplots(1, 2, figsize=(18, 12))
    im1 = axs[0].imshow(np.abs(field[0]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet', interpolation='none')
    axs[0].set_xlabel('x$_2$ $\\rightarrow$')
    axs[0].set_ylabel('$\\leftarrow$ x$_1$')
    axs[0].set_title('2D Electric field E1\n' + field_name, fontsize=13)
    fig.colorbar(im1, ax=axs[0], orientation='horizontal')
    # axs[0].plot(a*np.cos(phi), a*np.sin(phi), 'w')
    im2 = axs[1].imshow(np.abs(field[1]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet', interpolation='none')
    axs[1].set_xlabel('x$_2$ $\\rightarrow$')
    axs[1].set_ylabel('$\\leftarrow$ x$_1$')
    axs[1].set_title('2D Electric field E2\n' + field_name, fontsize=13)
    fig.colorbar(im2, ax=axs[1], orientation='horizontal')
    # axs[1].plot(a*np.cos(phi), a*np.sin(phi), 'w')
    plt.show()


def plotFieldComparisons(field_name_0, field_0, field_1, X1, X2, N1, N2):
    from matplotlib.ticker import ScalarFormatter
    # Plot wave fields in two-dimensional space
    fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)

    # Calculate the minimum and maximum values for each row
    row0_min = np.min([np.abs(field_0[0]), np.abs(field_1[0])])
    row0_max = np.max([np.abs(field_0[0]), np.abs(field_1[0])])

    # Plot each row and add a colorbar
    im0 = axs[0].imshow(np.abs(field_0[0]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet', interpolation='none', vmin=row0_min, vmax=row0_max)
    axs[0].set_xlabel('x$_2$ $\\rightarrow$')
    axs[0].set_ylabel('$\\leftarrow$ x$_1$')
    axs[0].set_title('2D Electric field E1\n' + field_name_0 + " Truth", fontsize=13)

    im1 = axs[1].imshow(np.abs(field_1[0]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet', interpolation='none', vmin=row0_min, vmax=row0_max)
    axs[1].set_xlabel('x$_2$ $\\rightarrow$')
    axs[1].set_ylabel('$\\leftarrow$ x$_1$')
    axs[1].set_title('2D Electric field E1\n' + field_name_0 + " Predicted", fontsize=13)

    # Add colorbars for each row
    cax0 = fig.add_axes([0.2, 0.1, 0.6, 0.03])  # Position of the colorbar for the first row
    cbar0 = fig.colorbar(im1, cax=cax0, orientation='horizontal', format=ScalarFormatter(useMathText=True))
    # plt.subplots_adjust(wspace=0.01)
    plt.show()

    ####
    # Plot wave fields in two-dimensional space
    fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)
    # Plot each row and add a colorbar
    im0 = axs[0].imshow(np.abs(field_0[0]-field_1[0]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet', interpolation='none')
    axs[0].set_xlabel('x$_2$ $\\rightarrow$')
    axs[0].set_ylabel('$\\leftarrow$ x$_1$')
    axs[0].set_title('2D Electric field E1\n' + "|" + field_name_0 + " - " + field_name_0 + "_pred|", fontsize=13)
    cbar0 = fig.colorbar(im0, ax=axs[0], orientation='horizontal', format=ScalarFormatter(useMathText=True))
    # cbar0.formatter.set_powerlimits((-3, 3))  # Adjust the power limits for scientific notation
    # cbar0.update_ticks()

    im1 = axs[1].imshow(np.abs(field_0[1]-field_1[1]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet', interpolation='none')
    axs[1].set_xlabel('x$_2$ $\\rightarrow$')
    axs[1].set_ylabel('$\\leftarrow$ x$_1$')
    axs[1].set_title('2D Electric field E2\n' + "|" + field_name_0 + " - " + field_name_0 + "|", fontsize=13)
    cbar1 = fig.colorbar(im1, ax=axs[1], orientation='horizontal', format=ScalarFormatter(useMathText=True))
    # cbar1.formatter.set_powerlimits((-3, 3))  # Adjust the power limits for scientific notation
    # cbar1.update_ticks()

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()
    ####

    fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)

    # Calculate the minimum and maximum values for each row
    row0_min = np.min([np.abs(field_0[1]), np.abs(field_1[1])])
    row0_max = np.max([np.abs(field_0[1]), np.abs(field_1[1])])

    # Plot each row and add a colorbar
    im0 = axs[0].imshow(np.abs(field_0[1]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet', interpolation='none', vmin=row0_min, vmax=row0_max)
    axs[0].set_xlabel('x$_2$ $\\rightarrow$')
    axs[0].set_ylabel('$\\leftarrow$ x$_1$')
    axs[0].set_title('2D Electric field E2\n' + field_name_0 + " Truth", fontsize=13)

    im1 = axs[1].imshow(np.abs(field_1[1]), extent=[X2.min(), X2.max(), X1.min(), X1.max()], cmap='jet', interpolation='none', vmin=row0_min, vmax=row0_max)
    axs[1].set_xlabel('x$_2$ $\\rightarrow$')
    axs[1].set_ylabel('$\\leftarrow$ x$_1$')
    axs[1].set_title('2D Electric field E2\n' + field_name_0 + " Predicted", fontsize=13)

    # Add colorbars for each row
    cax0 = fig.add_axes([0.2, 0.1, 0.6, 0.03])  # Position of the colorbar for the first row
    cbar0 = fig.colorbar(im1, cax=cax0, orientation='horizontal', format=ScalarFormatter(useMathText=True))
    # plt.subplots_adjust(wspace=0.01)
    plt.show()


def plot_history_ignore(history, ignore_entries):
    # Ignore the first ignore_entries entries
    result_dict = {}
    for key, value in history.items():
        result_dict[key] = value[ignore_entries:]
    return result_dict


def plot_loss(history):
    # Plot the loss
    plt.semilogy(history['loss'], label='Training Loss')
    plt.semilogy(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_red_square():
    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Create a red square patch
    square = plt.Rectangle((0, 0), 1, 1, fc='red')

    # Add the square patch to the axis
    ax.add_patch(square)

    # Set the axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Remove the axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Display the plot
    plt.show()


def vector2matrix(w, N1, N2):
    # Modify vector output from 'bicgstab' to matrix for further computations
    # N = CHI_eps.flatten('F').shape[0]
    N = N1 * N2
    DIM = [N1, N2]
    w_E = np.zeros((2, N1, N2), dtype=np.complex128, order='F')
    w_E[0, :, :] = np.reshape(w[0:N], DIM, order='F')
    w_E[1, :, :] = np.reshape(w[N:2*N], DIM, order='F')
    return w_E
