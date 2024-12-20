import random
import time
import matplotlib.pyplot as plt
import numpy as np


SIZE = 5
TOLERANCE = 10**(-10)

def transposeMatrix(A):
    n = len(A)
    B = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            B[j][i] = A[i][j]
    return B

def transposeVector(A):
    n = len(A)
    B = [[0] for _ in range(n)]
    for i in range(n):
        B[i][0] = A[i]
    return B

def transposeColumn(A):
    n = len(A)
    B = [0 for _ in range(n)]
    for i in range(n):
        B[i] = A[i][0]
    return B

def matrixMultiplication(A, B):
    if len(A[0]) != len(B):
        print("Error: Size of matrices is different")
        return
    else:
        C = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                s = 0
                for k in range(len(A[0])):
                    s += A[i][k]*B[k][j]
                C[i][j] = s
        return C

def vectorsMultiplication(A, B):
    C = [[0 for _ in range(len(B))] for _ in range(len(B))]
    for i in range(len(B)):
        for j in range(len(B)):
            C[i][j] = A[i][0]*B[j]
    return C

def matrixAddition(A, B):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        print("Error: Size of matrices is different")
        return
    else:
        n = len(A)
        m = len(A[0])
        for i in range(n):
            for j in range(m):
                A[i][j] += B[i][j]
        return A

def numberMultiplication(A, B):
    n = len(A)
    for i in range(n):
        for j in range(n):
            A[i][j] *= B
    return A

def numberAddition(A, B):
    n = len(A)
    for i in range(n):
        for j in range(n):
            A[i][j] += B
    return A

def signF(A):
    if A > 0:
        return 1
    elif A < 0:
        return -1
    else:
        return 0

def vectorNorm(A):
    n = len(A)
    s = 0
    for i in range(n):
        s += A[i]**2
    return s**(0.5)

def columnNorm(A):
    n = len(A)
    s = 0
    for i in range(n):
        s += A[i][0]**2
    return s**(0.5)

def vectorSubtract(A, B):
    n = len(A)
    for i in range(n):
        A[i] = A[i] - B[i]
    return A

def numberVectorMulti(A, B):
    n = len(A)
    for i in range(n):
        A[i] *= B
    return A

def numberColumnMultiply(A, B):
    n = len(A)
    for i in range(n):
        A[i][0] *= B
    return A

def stringColumnMultip(A, B):
    s = 0
    n = len(A)
    for i in range(n):
        s += A[i]*B[i][0]
    return s

def convergenceChecking(A):
    s = 0
    n = len(A)
    for i in range(n):
        for j in range(i+1,n):
            s += A[i][j]**2
    s = s**(0.5)
    if s <= TOLERANCE:
        return True
    else:
        return False

def householder_method(A):
    n = len(A)
    R = A.copy()
    Q = [[0 if i!=j else 1 for i in range(n)] for j in range(n)]
    for k in range(n):
        x = [R[i+k][k] if i+k < n else 0 for i in range(n-k)]
        x = transposeVector(x)
        sign = signF(x[0][0])
        x_norm = columnNorm(x)
        e = transposeVector([1 if i == 0 else 0 for i in range(n - k)])
        e[0][0] = x_norm * sign
        u = matrixAddition(x, e)
        u = numberColumnMultiply(u, 1 / columnNorm(u))
        H = [[0 if i!=j else 1 for i in range(n)] for j in range(n)]
        u_multip = vectorsMultiplication(u, transposeColumn(u))
        Z = [[0 for _ in range(n)] for _ in range(n)]
        i = 0
        while (k+i < n):
            j = 0
            while (k+j < n):
                Z[i+k][j+k] = u_multip[i][j]
                j += 1
            i+=1
        H = matrixAddition(H, numberMultiplication(Z, -2))
        R = matrixMultiplication(H, R)
        Q = matrixMultiplication(Q, transposeMatrix(H))

    return Q,R


def qr_algorithm(A):
    max_iterations = 1000
    n = len(A)
    V = [[0 if i!=j else 1 for i in range(n)] for j in range(n)]
    for i in range(max_iterations):
        Q, R = householder_method(A)
        V = matrixMultiplication(V, Q)
        A = matrixMultiplication(R, Q)
        if convergenceChecking(A):
            print(i)
            break
    return A, V

def generation_almost_singular_matrices(n):
    A = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        A[0][i] = float(float(random.randint(-12,12)) + float(random.randint(-12,12))/100)
        while A[0][i] == 0:
            A[0][i] = float(float(random.randint(-12,12)) + float(random.randint(-12,12))/100)
    for i in range(1,n):
        for j in range(n):
            if j % 2 == 0:
                A[i][j] = A[i-1][j]+i/10
            else:
                A[i][j] = A[i-1][j]-i/10
            while A[i][j] == 0:
                A[i][j] = float(float(random.randint(-12,12)) + float(random.randint(-12,12))/100)
    return A

def generation_random_matrices(n):
    A = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = float(float(random.randint(-12,12)) + float(random.randint(-12,12))/100)
            while A[i][j] == 0:
                A[i][j] = float(float(random.randint(-12,12)) + float(random.randint(-12,12))/100)
    return A

def print_eigenvectors(V):
    n = len(V)
    print("Eigenvectors:")
    for i in range(n):
        print("Eigenvector",i+1, ":")
        for j in range(n):
            print(V[j][i])
    return


def print_eigenvalues(A):
    n = len(A)
    print("Eigenvalues:")
    for i in range(n):
        print("λ",i+1," =  ",A[i][i])
    return

def print_matrix(matrix):
    max_width = max(len(str(element)) for row in matrix for element in row)
    for row in matrix:
        print(" ".join(f"{element:>{max_width}}" for element in row))


def process_for_matrix(key, size):
    for i in range (2, size):
        if key == 1:
            A = generation_almost_singular_matrices(i)
        else:
            A = generation_random_matrices(i)
        print_matrix(A)
        start = time.time()
        A, V = qr_algorithm(A)
        end = time.time()
        print_matrix(A)
        print_eigenvalues(A)
        print_eigenvectors(V)
        print("Lead time - ", end - start)

def process_for_custom_matrix():
    print("Введите размер матрицы")
    n = int(input())
    A = [[0 for _ in range(n)] for _ in range(n)]
    print("Идя построчно, вводите значение для каждого элемента матрицы")
    for i in range(n):
        for j in range(n):
            A[i][j] = float(input())
    print_matrix(A)
    start = time.time()
    A, V = qr_algorithm(A)
    end = time.time()
    print_matrix(A)
    print_eigenvalues(A)
    print_eigenvectors(V)
    print("Время выполнения - ", end - start)

def draw_graph_qr_random():
    times = [0 for i in range(SIZE)]
    matrices_size =[0 for i in range(SIZE)]
    for i in range(SIZE):
        matrices_size[i] = i+2
        A = generation_random_matrices(i + 2)
        start = time.time()
        A, V = qr_algorithm(A)
        end = time.time()
        times[i] = end - start
    coefficient = np.polyfit(matrices_size, times,3)
    poly = np.poly1d(coefficient)
    new_size = np.linspace(min(matrices_size),max(matrices_size),100)
    new_time = poly(new_size)
    print(coefficient)
    plt.scatter(matrices_size,times,label = "Lead Time",color = 'purple')
    plt.plot(new_size, new_time, color = 'blue', label='Cubic Regression')
    plt.xlabel('Size of matrix')
    plt.ylabel('Time')
    plt.title('Execution time of the QR algorithm on a random matrix')
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(min(matrices_size), max(matrices_size) + 1, 1.0))
    plt.show()
    return


def draw_graph_almost_singular_matrices():
    times = [0 for i in range(SIZE)]
    matrices_size =[0 for i in range(SIZE)]
    for i in range(SIZE):
        matrices_size[i] = i+2
        A = generation_almost_singular_matrices(i + 2)
        start = time.time()
        A, V = qr_algorithm(A)
        end = time.time()
        times[i] = end - start
    coefficient = np.polyfit(matrices_size, times,3)
    poly = np.poly1d(coefficient)
    new_sizes = np.linspace(min(matrices_size),max(matrices_size),100)
    new_time = poly(new_sizes)
    print(coefficient)
    plt.scatter(matrices_size,times,label="Lead Time",color='purple')
    plt.plot(new_sizes, new_time, color='blue', label='Cubic Regression')
    plt.xlabel('Size of matrix')
    plt.ylabel('Time')
    plt.title('Execution time of the QR algorithm on an almost singular matrix')
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(min(matrices_size), max(matrices_size) + 1, 1.0))
    plt.show()
    return

def run():
    x = 123
    while x != 0:
        print("You're welcome! It's QR-algorithm with Householder's method!")
        print("1 - QR алгоритм для случайных матриц. Выполняется до определенного размера матрицы -  n"
              "\n2 - QR алгоритм для почти вырожденных матриц. Выполняется до определенного размера матрицы - n"
              "\n3 - Ввести свою матрицу для QR алгоритма\n4 - Вывести график для случайных матриц"
              "\n5 - Вывести график для почти вырожденных матриц"
              "\n0 - Выйти\n")
        x = int(input())
        if x == 1:
            print("До какого размера матрицы будем выполнять поиск собственных чисел и векторов?")
            n = int(input())
            process_for_matrix(x, n + 1)
        elif x == 2:
            print("До какого размера матрицы будем выполнять поиск собственных чисел и векторов?")
            n = int(input())
            process_for_matrix(x, n + 1)
        elif x == 3:
            process_for_custom_matrix()
        elif x == 4:
            draw_graph_qr_random()
        elif x ==  5:
            draw_graph_almost_singular_matrices()
        elif x != 0:
            print("Ошибка: Введён некорректный номер")
    return

run()
