import math
import numpy as np

def my_dot(A, B):
    if len(A) != len(B):
        raise ValueError("Vectors must have same length")
    result = 0
    for i in range(len(A)):
        result += A[i] * B[i]
    return result

def my_norm(A):
    sum_sq = 0
    for x in A:
        sum_sq += x ** 2
    return math.sqrt(sum_sq)

# Take vector input
A = list(map(float, input("Enter values of vector A (space separated): ").split()))
B = list(map(float, input("Enter values of vector B (space separated): ").split()))

print("My Dot Product:", my_dot(A, B))
print("My Norm of A  :", my_norm(A))
print("My Norm of B  :", my_norm(B))

print("\nNumPy Dot Product:", np.dot(A, B))
print("NumPy Norm of A  :", np.linalg.norm(A))
print("NumPy Norm of B  :", np.linalg.norm(B))
