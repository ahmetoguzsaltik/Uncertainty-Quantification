# should read from file but this is faster
import math
from scipy.stats import multivariate_normal
import numpy as np
#q= h*width*velocity

np.random.seed(123)

h = np.array([46.8, 45.3, 40.1, 40.1, 56.9, 29.9, 29.9, 32.3, 32.3, 23.7, 23.7]).reshape((11, 1)) # define h array as matrix 1*1
q = np.array([2039, 1557, 1291.7, 1298.2, 2639.5, 420.7, 433.5, 559, 501.6, 266.6, 265.8]).reshape((11, 1)) # define q array as matrix 1*1
print("h matrix shape", h.shape)
print("q matrix shape", q.shape)

n = 2 # two parameters h and q
x = np.random.rand(n,1)
print("x matrix shape", x.shape)

nd = h.shape[0]

A = np.zeros((nd, n))


for i in range(nd):
    A[i][0] = 1 # from the pdf Ai1 = 1
    A[i][1] = math.log(h[i]) #Ai2 = log(hi)

Ax = np.matmul(A, x)

print("AX matrix shape", Ax.shape)

log_q = np.log(q)
print("q_ax matrix shape", log_q.shape)


error_array=np.subtract(log_q, Ax)
print("error_array matrix shape", error_array.shape)
print(error_array)  # Task 1 Answer


##Task 2

m0 = np.array([0, 0])
E0 = np.array([[10, 0], [0, 10]])

#Noise Part
covariance_matrix = np.zeros((nd, nd), float)
for i in range(nd):
    covariance_matrix[i][i] = 0.001 * log_q[i] # diagonal matrix


A_transpose = A.T

# common part between equation6 and equation7
common_part1 = np.matmul(A, np.matmul(E0, A_transpose))
common_part2 = np.power(np.add(covariance_matrix, common_part1), -1)
common_part3 = np.matmul(np.matmul(E0, A_transpose), common_part2)

# Equation 6
m_part1 = np.subtract(q, m0)
m_part2 = np.matmul(common_part3, m_part1)
m = np.add(m0, m_part2)
print(m)
m = np.array([m[0][1], m[1][0]]) # It must be a vector

# Equation 7
e_part1 = np.matmul(A, E0)
e_part2 = np.matmul(common_part3, e_part1)
E = np.subtract(E0, e_part2)
print(E)


### Task 3

def compute_prior(parameter_array, mean, covariance):
    result = multivariate_normal.pdf(parameter_array, mean, covariance)
    
    return result
    

# Test the function
compute_prior(x, m0, E0)



def compute_likelihood(q, h):
    
    np.random.seed(123)
    n = 2 # two parameters h and q
    x = np.random.rand(n,1)
    nd = h.shape[0]
    A = np.zeros((nd, n))
    
    for i in range(nd):
        A[i][0] = 1 # from the pdf Ai1 = 1
        A[i][1] = math.log(h[i]) #Ai2 = log(hi)

    Ax = np.matmul(A, x)
    
    result = np.exp(-0.5 * (np.linalg.norm((np.log(q) - Ax)) ** 2))
    
    return result


# Test the function  
compute_likelihood(q,h)

  

def likelihood_prior_product(q, h, parameter_array, mean, covariance):
    
    result1 = multivariate_normal.pdf(parameter_array, mean, covariance)
    
    
    np.random.seed(123)
    n = 2 # two parameters h and q
    x = np.random.rand(n,1)
    nd = h.shape[0]
    A = np.zeros((nd, n))
    
    for i in range(nd):
        A[i][0] = 1 # from the pdf Ai1 = 1
        A[i][1] = math.log(h[i]) #Ai2 = log(hi)

    Ax = np.matmul(A, x)
    
    result2 = np.exp(-0.5 * (np.linalg.norm((np.log(q) - Ax)) ** 2))
    
    return result1 * result2


# Test the function 
likelihood_prior_product(q, h, x, m0, E0)

