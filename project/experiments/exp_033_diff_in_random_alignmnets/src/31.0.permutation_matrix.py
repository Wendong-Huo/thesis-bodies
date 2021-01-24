# Let A be an arbitrary matirx
# Find a permutation matrix B that is close to matrix A
import numpy as np 
import matplotlib.pyplot as plt

def permutation_matrix(A):
    """input a square matrix A, to get a close permutation matrix B."""
    assert A.shape[0]==A.shape[1], "matrix A is not square"
    B = np.zeros_like(A)
    A_length = A.shape[0]
    A_flatten = A.flatten()
    A_argsort = np.argsort(A_flatten)[::-1]
    visited_x = {}
    visited_y = {}
    for i in A_argsort:
        x =  i//A_length
        y = i%A_length
        if x in visited_x or y in visited_y:
            continue
        visited_x[x] = True
        visited_y[y] = True
        B[x,y]=1
    return B

np.random.seed(0)
A = np.random.random(size=[40,40])
B = permutation_matrix(A)
order = np.arange(start=0,stop=40)
c = np.dot(B,order)

print(A)
print(B)
print(c)