import numpy as np
def posDef(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.allclose(matrix, matrix.T):
        return False
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues > 0)

def diagDom(matrix):
    n = matrix.shape[0]
    for i in range(0, n) :  
        sum = 0
        for j in range(0, n) :
            sum = sum + abs(matrix[i][j]) 
    sum = sum - abs(matrix[i][i])
    if (abs(matrix[i][i]) < sum) :
        return False
    return True

def LUFactorization(array):
    n = array.shape[0] 
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for k in range(n):
        L[k, k] = 1
        for j in range(k, n):
            U[k, j] = array[k, j] - np.dot(L[k, :k], U[:k, j])
        for i in range(k + 1, n):
            L[i, k] = (array[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]

    return L, U




def rungeKutta(function, rangeTuple, iterations, initial):

    h = (rangeTuple[1] - rangeTuple[0]) / iterations
    tPrev = rangeTuple[0]
    yPrev = initial

    for _ in range(iterations):
        k1 = h * function(tPrev, yPrev)
        k2 = h* function(tPrev + h/ 2, yPrev + k1 / 2)
        k3 = h* function(tPrev + h/ 2, yPrev + k2 / 2)
        k4 = h* function(tPrev + h, yPrev + k3)

        t = tPrev + h
        y= yPrev + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        tPrev= t
        yPrev = y


    return y



def euler(function, rangeTuple, iterations, initial):
    h = (rangeTuple[1] - rangeTuple[0]) / iterations
    tPrev = rangeTuple[0]
    yPrev = initial

    for i in range(iterations):
        t = tPrev + h
        y = yPrev + h * function(tPrev, yPrev)
        tPrev = t
        yPrev = y

    return y

def gaussianElimination(array):
    for i in range(array.shape[0]):
        maxRow = i + np.argmax(np.abs(array[i:, i]))
        array[[i, maxRow]] = array[[maxRow, i]]
        for j in range(i + 1, array.shape[0]):
            factor = array[j, i] / array[i, i]
            array[j, i:] = array[j, i:] - factor * array[i, i:]

    x = np.zeros(array.shape[0])
    for i in range(array.shape[0] - 1, -1, -1):
        x[i] = (array[i, -1] - np.dot(array[i, :-1], x)) / array[i, i]
    x = x.astype(dtype=np.double)
    return x

def main():
    np.set_printoptions(precision=7, suppress=True, linewidth=100)
    # Parameters for Question 1 and 2
    def functionTY(t, y):
     return t - y ** 2
    range = (0, 2)
    iterations = 10
    initialPoint = 1

    # Question 1 Euler's Method
    print('%.5f \n' % euler(functionTY, range, iterations, initialPoint))

    # Question 2 Runge-Kutta
    print('%.5f \n' % rungeKutta(functionTY, range, iterations, initialPoint))

    # Question 3 Gaussian Elimination
    gausianArray = np.array([[2, -1, 1, 6],
                    [1, 3, 1, 0],
                    [-1, 5, 4, -3]])

    
    print(gaussianElimination(gausianArray),"\n")
    # Question 4 LU Factorization

    LUArray = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
    L, U = LUFactorization(LUArray)
    determinant = np.linalg.det(U)

    print('%.5f \n' %determinant)
    print(L,"\n")
    print(U,"\n")
    # Question 5 Diagonally Dominate
    domMatrix = np.array([
    [9, 0, 5, 2, 1],
    [3, 9, 1, 2, 1],
    [0, 1, 7, 2, 3],
    [4, 2, 3, 12, 2],
    [3, 2, 4, 0, 8],
    ])

    print(diagDom(domMatrix),"\n")
    # Question 6 Positive Definite
    posDefMatrix = np.array([
    [2, 2, 1],
    [2, 3, 0],
    [1, 0, 2],
    ])
    print(posDef(posDefMatrix))





main()


