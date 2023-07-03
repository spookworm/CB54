import scipy.io
import numpy as np


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
    var_name_m = np.array(var_name_matlab[var_name])
    return var_name_m
