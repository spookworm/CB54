# from numba import jit
# import numpy as np
# import time

# x = np.arange(100).reshape(10, 10)


# @jit(nopython=True, parallel=True)
# def go_fast(a):  # Function is compiled and runs in machine code
#     trace = 0.0
#     for i in range(a.shape[0]):
#         trace += np.tanh(a[i, i])
#     return a + trace


# # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
# start = time.perf_counter()
# go_fast(x)
# end = time.perf_counter()
# print("Elapsed (with compilation) = {}s".format((end - start)))

# # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
# start = time.perf_counter()
# go_fast(x)
# end = time.perf_counter()
# print("Elapsed (after compilation) = {}s".format((end - start)))

import numpy as np
from numba import jit
from scipy.sparse.linalg import LinearOperator, bicgstab
import time


# @jit(nopython=True)
def Aw_operator(x):
    return A.dot(x)


A = np.random.rand(1000, 1000)
b = np.random.rand(1000)

A_operator = LinearOperator((1000, 1000), matvec=Aw_operator)
x, info = bicgstab(A_operator, b, tol=1e-5)

# print(x)


start = time.perf_counter()
Aw_operator(x)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.perf_counter()
Aw_operator(x)
end = time.perf_counter()
print("Elapsed (after compilation) = {}s".format((end - start)))
