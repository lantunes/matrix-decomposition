import numpy as np

# Eigendecomposition applies only to square matrices
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(A)

# Eigendecomposition with numpy
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)
print(eigenvectors)

# the eigenvector matrix has the same dimensions as A (i.e. 3x3)
first_eigenvector = eigenvectors[:,0]  # i.e. first column
print(first_eigenvector)
# eigenvectors are normalized to unit length
print(np.linalg.norm(first_eigenvector))

# the eigenvalues are a list, where the i'th eigenvalue corresponds with the i'th eigenvector
first_eigenvalue = eigenvalues[0]

# Confirm that A * x = c * x, where c is an eigenvalue, and x is an eigenvector
print(A.dot(first_eigenvector))
print(first_eigenvalue * first_eigenvector)

# reconstruct A using its eigenvalues and eigenvectors
# A = PDP^-1, where P are the eigenvectors, D is a diagonal matrix with the eigenvalues
P = eigenvectors
P_inv = np.linalg.inv(P)
D = np.diag(eigenvalues)
print(P.dot(D).dot(P_inv))
