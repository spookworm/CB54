import cupy as cp
import cupyx.scipy.sparse as sparse

# Create a sparse matrix
A = sparse.random(1000, 1000, density=0.1, format='csr')

# Create a vector
b = cp.random.rand(1000)

# Define a lambda function for matrix-vector multiplication
matvec = lambda x: A @ x

# Use the cg function to solve the linear system Ax = b
x, info = sparse.linalg.cg(matvec, b)

print(x)
print(info)