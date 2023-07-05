import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def tidy_workspace():
    # Iterate over local variables and delete the types that are None to keep workspace tidy
    local_variables = list(locals().keys())
    for var_name in local_variables:
        var_value = locals()[var_name]
        if var_value is None:
            del locals()[var_name]
            # print(f"Deleted local variable: {var_name}")


def mat_checker(variable, var_name):
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
    # x1fft_m = mat_loader(nameof(x1fft))
    var_name_matlab = scipy.io.loadmat(var_name + '.mat')
    var_name_m = np.array(var_name_matlab[var_name], dtype=np.complex_)
    return var_name_m


def plotDiff(ITERBiCGSTABw, X1, X2):
    x1 = X1[:, 0]
    # x2 = ForwardBiCGSTABFFT.input.X2(1, :);
    x2 = X2[0, :]
    fig = plt.figure(figsize=(7.09, 4.72))
    # fig.subplots_adjust(wspace=0.3)
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(np.abs(ITERBiCGSTABw), extent=[x2[0], x2[-1], x1[-1], x1[0]], cmap='jet', interpolation='none')
    ax1.set_xlabel('x_2 \u2192')
    ax1.set_ylabel('\u2190 x_1')
    ax1.set_aspect('equal', adjustable='box')
    fig.colorbar(im1, ax=ax1, orientation='horizontal')
    plt.show()
