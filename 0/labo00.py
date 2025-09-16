#ej1
import numpy as np

def esCuadrada(a):
    return a.ndim == 2 and a.shape[0] == a.shape[1]

def triangSup(a):
    if not esCuadrada(a):
        return False

    filas, columnas = a.shape
    result = [[0 for _ in range(columnas)] for _ in range(filas)]
    result = np.array(result)
    for fila in range(filas):
        for columna in range(columnas): 
            if fila >= columna:
                result[fila][columna] = 0
            else :
                result[fila][columna] = a[fila][columna]
    return result

def triangInf(a):
    if not esCuadrada(a):
        return False

    filas, columnas = a.shape
    result = [[0 for _ in range(columnas)] for _ in range(filas)]
    result = np.array(result)
    for fila in range(filas):
        for columna in range(columnas): 
            if fila <= columna:
                result[fila][columna] = 0
            else:
                result[fila][columna] = a[fila][columna]
    return result

def diagonal(a):
    if not esCuadrada(a):
        return False

    filas, columnas = a.shape
    result = [[0 for _ in range(columnas)] for _ in range(filas)]
    result = np.array(result)
    for fila in range(filas):
        for columna in range(columnas): 
            if fila != columna:
                result[fila][columna] = 0
            else:
                result[fila][columna] = a[fila][columna]
    return result

def traza(a):
    if not esCuadrada(a):
        return False

    result = 0
    filas, columnas = a.shape
    for fila in range(filas):
        for columna in range(columnas): 
            if fila == columna:
                result = result + a[fila][columna]
    return result

def traspuesta(a):
    filas, columnas = a.shape
    result = [[0 for _ in range(filas)] for _ in range(columnas)]
    result = np.array(result)
    for fila in range(filas):
        for columna in range(columnas): 
                result[columna][fila] = a [fila][columna]
    return result

def esSimetrica(a):
    if not esCuadrada(a) :
        return False

    filas, columnas = a.shape
    res = [[0 for _ in range(filas)] for _ in range(columnas)]
    tras = traspuesta(a)

    return restar(a, tras) == res

def restar(a, b):
    # ponele que checkeamos que sean == las dim
    if a.shape != b.shape:
        raise Exception("No se puede :(")

    filas, columnas = a.shape
    res = [[0 for _ in range(filas)] for _ in range(columnas)]
    for i in range(filas):
        for j in range(columnas):
            res[i][j] = a[i][j] - b[i][j]

    return res

def calcularAx(matriz, vector_x):
    if matriz.shape[1] != vector_x.shape[0]:
        raise Exception("No se puede :(")

    result = [0 for _ in range(matriz.shape[0])]

    for i in range(matriz.shape[0]):
        for j in range(vector_x.shape[0]):
            result[i] += matriz[i][j] * vector_x[j]

    return result

def intercambiarFila(matriz, fila1, fila2):
    for j in range(matriz.shape[1]):
        tmp = matriz[fila1][j]
        matriz[fila1][j] = matriz[fila2][j]
        matriz[fila2][j] = tmp

def sumarFilaMultiplo(matriz, fila1, fila2, num):
    for j in range(matriz.shape[1]):
        matriz[fila1][j] += num*matriz[fila2][j]

def esDiagonalmenteDominante(matriz):
    if not esCuadrada(matriz):
        return False

    for i in range(matriz.shape[0]):
        elem_diag = abs(matriz[i][i])
        sum = 0
        for j in range(matriz.shape[1]):
            if i != j:
                sum += abs(matriz[i][j])

        if elem_diag <= sum:
            return False

    return True

def circulante(vector):

    result = [[0 for _ in range(vector.shape[0])] for _ in range(vector.shape[0])]

    for i in range(vector.shape[0]):
        for j in range(vector.shape[0]):
            result[i][j] = vector[(j-i)%vector.shape[0]]

    return result

def matrizVandermonde(vector):

    result = [[0 for _ in range(vector.shape[0])] for _ in range(vector.shape[0])]

    for i in range(vector.shape[0]):
        for j in range(vector.shape[0]):
            result[i][j] = vector[j]**i

    return result

def numeroAureo(n):
    a = 0
    b = 1

    for i in range(n):
        tmp = a
        a = b
        b += tmp

    if a == 0:
        return 0

    return b/a

if __name__ == '__main__':

    print("MATRIZ CUADRADA: ")
    matrizCuadrada = np.array([[1,2,3],[1,2,3],[1,2,3]])
    print(matrizCuadrada)

    print("MATRIZ NO CUADRADA: ")
    matrizNoCuadrada = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
    print(matrizNoCuadrada)

    print("Test esCuadrada: ")
    esCuad = esCuadrada(matrizCuadrada)
    print(esCuad)

    print("Test triangSup: ")
    triangs = triangSup(matrizCuadrada)
    print(triangs)

    print("Test triangInf: ")
    triangi = triangInf(matrizCuadrada)
    print(triangi)

    print("Test traspuesta: ")
    matrisNoCuadradaTraspuesta = traspuesta(matrizNoCuadrada)
    print(matrisNoCuadradaTraspuesta)

    matrizsimetrica = np.array([[1,2,3],
                              [2,1,2],
                              [3,2,1]])
    matrizNoSimetrica = np.array([[1,2,3],
                              [99,1,2],
                              [3,2,1]])

    print("Test simetrica 1: ")
    esSim = esSimetrica(matrizsimetrica)
    print(esSim)

    print("Test simetrica 2: ")
    noesSim = esSimetrica(matrizNoSimetrica)
    print(noesSim)

    print("Test simetrica 3: ")
    noesSim2 = esSimetrica(matrizNoCuadrada)
    print(noesSim2)

    print("Test Ax1:")
    testAx1 = calcularAx(matrizsimetrica, np.array([1,1,1]))
    print(testAx1)

    print("Test intercambiar fila: ")
    matrizCuadrada2 = np.array([[4,4,4],[1,2,3],[6,6,6]])
    intercambiarFila(matrizCuadrada2, 0, 1)
    print(matrizCuadrada2)

    print("Test diagonalmente dominante: ")
    matrizCuadrada3 = np.array([[4000, 4, 4], [1, 2000, 3], [6, 6, 6000]])
    print(esDiagonalmenteDominante(matrizCuadrada3))
    matrizCuadrada3 = np.array([[4000, 4, 4], [1, 1, 3], [6, 6, 6000]])
    print(esDiagonalmenteDominante(matrizCuadrada3))
    matrizCuadrada3 = np.array([[4, 4, 4], [1, 2, 3], [6, 6, 6]])
    print(esDiagonalmenteDominante(matrizCuadrada3))

    print("Test circulante: ")
    print(circulante(np.array([1,2,3,4,5])))

    print("Test Vandermonde")
    print(matrizVandermonde(np.array([1,2,3,4,5])))

    print("Test Aureo: ")
    for i in range(1, 10):
        print(numeroAureo(i))
