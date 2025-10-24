from fcntl import FASYNC

import numpy as np
import math

from soupsieve.util import lower


# labo 0

def esCuadrada(a):
    return a.ndim == 2 and a.shape[0] == a.shape[1]

def matrizDeCeros(filas, columnas):
    return [[0.0 for _ in range(columnas)] for _ in range(filas)]

def triangSup(a):
    if not esCuadrada(a):
        return False

    filas, columnas = a.shape
    result = matrizDeCeros(filas, columnas)
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
    result = matrizDeCeros(filas, columnas)
    result = np.array(result)
    for fila in range(filas):
        for columna in range(columnas):
            if fila <= columna:
                result[fila][columna] = 0
            else:
                result[fila][columna] = a[fila][columna]
    return result

def diagonal(a):
    a = np.array(a)
    if not esCuadrada(a):
        return False

    filas, columnas = a.shape
    result = matrizDeCeros(filas, columnas)
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
    result = matrizDeCeros(filas, columnas)
    result = np.array(result)
    for fila in range(filas):
        for columna in range(columnas):
            result[columna][fila] = a [fila][columna]
    return result

def esSimetrica(a):
    if not esCuadrada(a) :
        return False

    filas, columnas = a.shape
    res = matrizDeCeros(filas, columnas)
    tras = traspuesta(a)

    return restar(a, tras) == res

def restar(a, b):
    # ponele que checkeamos que sean == las dim
    if a.shape != b.shape:
        raise Exception("No se puede :(")

    filas, columnas = a.shape
    res = matrizDeCeros(filas, columnas)
    for i in range(filas):
        for j in range(columnas):
            res[i][j] = a[i][j] - b[i][j]

    return res

def calcularAx(matriz, vector_x):
    tamVector = len(vector_x)
    if matriz.shape[1] != tamVector:
        raise Exception("No se puede :(")

    result = [0 for _ in range(matriz.shape[0])]

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            result[i] += matriz[i][j] * vector_x[j]

    return np.array(result)

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

    result = matrizDeCeros(vector.shape[0], vector.shape[0])

    for i in range(vector.shape[0]):
        for j in range(vector.shape[0]):
            result[i][j] = vector[(j-i)%vector.shape[0]]

    return result

def matrizVandermonde(vector):

    result = matrizDeCeros(vector.shape[0], vector.shape[0])

    for i in range(vector.shape[0]):
        for j in range(vector.shape[0]):
            result[i][j] = vector[j]**i

    return result

def numeroAureo(n):
    a = 0
    b = 1

    for i in range(n+1):
        tmp = a
        a = b
        b += tmp

    if a == 0:
        return 0

    return b/a

def multiplicar(matrizA, matrizB):
    if matrizA.shape[1] != matrizB.shape[0]:
        raise Exception("No se puede :(")

    res = matrizDeCeros(matrizA.shape[0], matrizB.shape[1])

    for i in range(matrizA.shape[0]):
        for j in range(matrizB.shape[1]):
            for k in range(matrizA.shape[1]):
                res[i][j] += matrizA[i][k]*matrizB[k][j]

    return np.array(res)

def vectorPorEscalar(x, s):
    res = []
    for i in range(len(x)):
        res.append(x[i]*s)
    return np.array(res)

# labo 1

def error(x, y):
    return abs(x - y)

def error_relativo(x, y):
    return abs(x - y) / abs(x)

def matricesIguales(A, B):
    """
    Devuelve True si ambas matrices son iguales y False en otro caso.
    Considerar que las matrices pueden tener distintas dimensiones, ademas de distintos valores.
    """
    if A.shape != B.shape:
        return False

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if abs(A[i][j] - B[i][j]) >= 1e-08:
                return False

    return True

# labo 2

def rota(theta):
    """
    Recibe un angulo theta y retorna una matriz de 2x2
    que rota un vector dado en un angulo theta
    """
    res =np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ])
    return res

def escala(s):
    """
    Recibe una tira de numeros s y retorna una matriz cuadrada de
    n x n, donde n es el tamaño de s.
    La matriz escala la componente i de un vector de Rn
    en un factor s[i]
    """
    cantElems = len(s)
    res = matrizDeCeros(cantElems, cantElems)
    for i in range(cantElems):
        res[i][i] = s[i]

    return np.array(res)

def rota_y_escala(theta,s):
    """
    Recibe un angulo theta y una tira de numeros s,
    y retorna una matriz de 2x2 que rota el vector en un angulo theta
    y luego lo escala en un factor s
    """
    res = multiplicar(rota(theta), escala(s))
    return res

def afin(theta,s,b):
    """
    Recibe un angulo theta , una tira de numeros s (en R2) , y un vector
    b en R2.
    Retorna una matriz de 3x3 que rota el vector en un angulo theta,
    luego lo escala en un factor s y por ultimo lo mueve en un valor
    fijo b
    """
    matriz2x2 = rota_y_escala(theta,s)
    matriz3x3 = np.array(matrizDeCeros(3, 3))
    matriz3x3[:2, :2] = matriz2x2
    matriz3x3[0][2] = b[0]
    matriz3x3[1][2] = b[1]
    matriz3x3[2][2] = 1
    return np.array(matriz3x3)

def trans_afin(v,theta,s,b):
    """
    Recibe un vector v (en R2), un angulo theta,
    una tira de numeros s (en R2), y un vector b en R2.
    Retorna el vector w resultante de aplicar la transformacion afin a
    v
    """
    transf = afin(theta,s,b)
    vectorCon1 =  np.append(v, 1)
    vectorColumna = vectorCon1.T
    vectorColumnaRes = calcularAx(transf, vectorColumna)
    res = vectorColumnaRes.T[:2]
    return res

# labo 3

def norma(x, p):
    """
    Devuelve la norma p del vector x.
    """
    if p == 'inf':
        vectorAbs = [0 for _ in range(len(x))]
        for i in range(len(x)):
            vectorAbs[i] = abs(x[i])
        return np.max(vectorAbs)

    sum = 0
    for i in range(len(x)):
        sum += abs(x[i])**p

    return sum**(1/p)

def normaliza(X, p):
    """
    Recibe X, una lista de vectores no vacios, y un escalar p. Devuelve
    una lista donde cada elemento corresponde a normalizar los
    elementos de X con la norma p.
    """
    vectoresNormalizados = []
    for i in range(len(X)):
        vectorActual = X[i]
        vectoresNormalizados.append(vectorPorEscalar(vectorActual, (1/norma(vectorActual, p))))
    return vectoresNormalizados

def normaMatMC(A, q, p, Np):
    """
    Devuelve la norma A-{q,p} y el vector x en el cual se alcanza
    el maximo.
    """
    vectoresAlAzar = np.random.rand(Np, A.shape[1])
    vectoresNormalizados = normaliza(vectoresAlAzar, p)

    vectorConNorma = [0 for _ in range(len(vectoresNormalizados))]
    for i in range(len(vectoresNormalizados)):
        vectorConNorma[i] = [norma(calcularAx(A, vectoresNormalizados[i]), q), vectoresNormalizados[i]]

    max = [0, [0, 0]]
    for i in range(len(vectorConNorma)):
        if vectorConNorma[i][0] > max[0]:
            max = vectorConNorma[i]

    # return max(vectorConNorma, key=lambda p: p[0])
    return max

def normaExacta(A, p=[1, 'inf']):
    """
    Devuelve una lista con las normas 1 e infinito de una matriz A
    usando las expresiones del enunciado 2.(c).
    """
    if not p in [1,'inf']:
        return None
    if p == 1:
        vectorSumas = []
        for j in range(A.shape[1]):
            sum = 0
            for i in range(A.shape[0]):
                sum += abs(A[i][j])
            vectorSumas.append(sum)
        return np.max(vectorSumas)

    if p == 'inf':
        vectorSumas = []
        for i in range(A.shape[0]):
            sum = 0
            for j in range(A.shape[1]):
                sum += abs(A[i][j])
            vectorSumas.append(sum)
        return np.max(vectorSumas)

def condMC(A, p, cantVect):
    """
    Devuelve el numero de condicion de A usando la norma inducida p.
    """
    inversa = np.linalg.inv(A)
    return normaMatMC(A, p, p, cantVect)[0] * normaMatMC(inversa, p, p, cantVect)[0]

def condExacta(A, p):
    """
    Que devuelve el numero de condicion de A a partir de la formula de
    la ecuacion (1) usando la norma p.
    """
    inversa = np.linalg.inv(A)
    return normaExacta(A, p) * normaExacta(inversa, p)

# labo 4

def calculaLU(A):
    nops = 0
    upper = matrizDeCeros(A.shape[0], A.shape[1]) + A
    lower = escala([1 for _ in range(A.shape[0])])

    for fila in range(upper.shape[0]):
        numDiagonal = upper[fila][fila]

        if np.abs(numDiagonal)  < 1e-08:
            return [None, None, 0]

        for fila2 in range(fila+1, upper.shape[0]):
            nops += 1

            coef = upper[fila2][fila]/numDiagonal
            lower[fila2][fila] = coef

            upper[fila2][fila] = 0.0

            for columna in range(fila+1, upper.shape[1]):
                upper[fila2][columna] = upper[fila2][columna] - coef*upper[fila][columna]
                nops += 2

    return [lower, upper, nops]


def res_tri(L, b, inferior=True):

    n = L.shape[1]
    x_vector =  np.zeros(n)
    
    if inferior:
        for i in range(n):
            x_actual = b[i]
            #idea si es triangular inferior la solucion es de la pinta (b1/L11 , (b2-L21.X1)/L22 , b3-L32.X2-L31.X1   cada x_n se le restan todos los anteriores x_n
            for j in range(i):
                x_actual -= L[i][j]*x_vector[j]
            x_actual = x_actual/L[i][i]
            x_vector[i] = x_actual
    else:
        for i in range(n-1,-1,-1):
            x_actual = b[i]
            for j in range(i+1,n):
                x_actual -= L[i][j]*x_vector[j]
            x_actual = x_actual/L[i][i]
            x_vector[i] = x_actual

    return x_vector

def inversa(A):

    descomposicion = calculaLU(A)
    L = descomposicion[0]
    U = descomposicion[1]

    if L is None or U is None:
        return None

    filas, columnas = A.shape
    inversa = np.zeros((filas, columnas))
    for columna in range(columnas):
        vector_canonico = np.zeros(filas)
        vector_canonico[columna] = 1.
        y = res_tri(L, vector_canonico)
        inversa[columna] = res_tri(U, y, False)
    return traspuesta(inversa)

def calculaLDV(A):
    nops = 0
    descomposicion = calculaLU(A)
    L = descomposicion[0]
    U = descomposicion[1]
    nops += descomposicion[2]

    if L is None or U is None:
        return [None, None, None, 0]

    U_t = traspuesta(U)
    descomposicionU_t = calculaLU(U_t)

    V_t = descomposicionU_t[0]
    D = descomposicionU_t[1]
    nops += descomposicionU_t[2]

    V = traspuesta(V_t)

    return [L, D, V, nops]

def esSDP(A, atol=1e-08):
    if not esSimetrica(A):
        return False

    descLDV = calculaLDV(A)

    if descLDV[0] is None:
        return False

    D = descLDV[1]

    for i in range(D.shape[0]):
        if D[i][i] <= 0:
            return False

    return True

def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """
    nops = 0 # TODO: Ver si los nops estan bien calculados porque no hay tests sobre ellos
    n = A.shape[0]

    if n != A.shape[1]:
        return None

    Q = np.zeros(A.shape)
    R = np.zeros(A.shape)

    for j in range(0, n):
        Q[:, j] = A[:, j]
        for k in range(j):
            R[k, j] = np.dot(Q[:, k], Q[:, j])
            nops += n * 2 - 1 # n mult y n-1 sumas
            Q[:, j] = Q[:, j] - vectorPorEscalar(Q[:, k], R[k, j])
            nops += n * 2 # n mult y n restas

        R[j, j] = norma(Q[:, j], 2)
        nops += n * 2 # n mult y n-1 sumas y una raiz
        Q[:, j] = vectorPorEscalar(Q[:, j], 1/R[j, j])
        nops += n # n mult

    if retorna_nops:
        return [Q, R, nops]

    return [Q, R]

def QR_con_HH(A,tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """

# Tests para los labos

# funciones extras para los tests
def sonIguales(x, y, atol=1e-08):
    return np.allclose(error(x,y),0, atol=atol)

def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

def correrTestsLabos():
    #test_labo1()
    #test_labo2()
    #test_labo3()
    #test_labo4()
    test_labo5()

def test_labo1():
    assert(not sonIguales(1,1.1))
    assert(sonIguales(1,1 + np.finfo('float64').eps))
    assert(not sonIguales(1,1 + np.finfo('float32').eps))
    assert(not sonIguales(np.float16(1),np.float16(1) + np.finfo('float32').eps))
    assert(sonIguales(np.float16(1),np.float16(1) + np.finfo('float16').eps,atol=1e-3))

    assert(np.allclose(error_relativo(1,1.1),0.1))
    assert(np.allclose(error_relativo(2,1),0.5))
    assert(np.allclose(error_relativo(-1,-1),0))
    assert(np.allclose(error_relativo(1,-1),2))

    assert(matricesIguales(np.diag([1,1]),np.eye(2)))
    assert(matricesIguales(np.linalg.inv(np.array([[1,2],[3,4]]))@np.array([[1,2],[3,4]]),np.eye(2)))
    assert(not matricesIguales(np.array([[1,2],[3,4]]).T,np.array([[1,2],[3,4]])))

def test_labo2():
    # Tests para rota
    assert(np.allclose(rota(0), np.eye(2)))
    assert(np.allclose(rota(np.pi/2), np.array([[0, -1],[1, 0]])))
    assert(np.allclose(rota(np.pi), np.array([[-1, 0],[0, -1]])))

    # Tests para escala
    assert(np.allclose(escala([2,3]), np.array([[2,0],[0,3]])))
    assert(np.allclose(escala([1,1,1]), np.eye(3)))
    assert(
        np.allclose(escala([0.5,0.25]), np.array([[0.5,0],[0,0.25]]))
    )

    # Tests para rota-y-escala
    assert(
        np.allclose(rota_y_escala(0,[2,3]), np.array([[2,0],[0,3]]))
    )
    assert(np.allclose(
        rota_y_escala(np.pi/2,[1,1]), np.array([[0,-1],[1,0]])
    ))
    assert(np.allclose(
        rota_y_escala(np.pi,[2,2]), np.array([[-2,0],[0,-2]]))
    )

    # Tests para afin
    assert(np.allclose(
        afin(0,[1,1],[1,2]),
        np.array([[1,0,1],
                  [0,1,2],
                  [0,0,1]]))
    )

    assert(np.allclose(afin(np.pi/2,[1,1],[0,0]),
                       np.array([[0,-1,0],
                                 [1,0,0],
                                 [0,0,1]]))
           )

    assert(np.allclose(afin(0,[2,3],[1,1]),
                       np.array([[2,0,1],
                                 [0,3,1],
                                 [0,0,1]]))
           )

    # Tests para trans_afin
    assert(np.allclose(
        trans_afin(np.array([1,0]), np.pi/2,[1,1],[0,0]),
        np.array([0,1])
    ))

    assert(np.allclose(
        trans_afin(np.array([1,1]), 0,[2,3],[0,0]),
        np.array([2,3])
    ))

    assert(np.allclose(
        trans_afin(np.array([1,0]), np.pi/2,[3,2],[4,5]),
        np.array([4,8])
    ))

def test_labo3():
    # Tests norma
    assert (np.allclose(norma(np.array([1, 1]), 2), np.sqrt(2)))
    assert (np.allclose(norma(np.array([1] * 10), 2), np.sqrt(10)))
    assert (norma(np.random.rand(10), 2) <= np.sqrt(10))
    assert (norma(np.random.rand(10), 2) >= 0)

    # Tests normaliza
    # Tests normaliza
    for x in normaliza([np.array([1] * k) for k in range(1, 11)], 2):
        assert (np.allclose(norma(x, 2), 1))
    for x in normaliza([np.array([1] * k) for k in range(2, 11)], 1):
        assert (not np.allclose(norma(x, 2), 1))
    for x in normaliza([np.random.rand(k) for k in range(1, 11)], 'inf'):
        assert (np.allclose(norma(x, 'inf'), 1))

    # Tests normaExacta

    assert (np.allclose(normaExacta(np.array([[1, -1], [-1, -1]]), 1), 2))
    assert (np.allclose(normaExacta(np.array([[1, -2], [-3, -4]]), 1), 6))
    assert (np.allclose(normaExacta(np.array([[1, -2], [-3, -4]]), 'inf'), 7))
    assert (normaExacta(np.array([[1, -2], [-3, -4]]), 2) is None)
    assert (normaExacta(np.random.random((10, 10)), 1) <= 10)
    assert (normaExacta(np.random.random((4, 4)), 'inf') <= 4)

    # Test normaMC

    nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
    assert(np.allclose(nMC[0],1,atol=1e-3))
    assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
    assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

    nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
    assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
    assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

    A = np.array([[1,2],[3,4]])
    nMC = normaMatMC(A=A,q='inf',p='inf',Np=1000000)
    assert(np.allclose(nMC[0],normaExacta(A,'inf'),rtol=2e-1)) 

    # Test condMC

    A = np.array([[1, 1], [0, 1]])
    A_ = np.linalg.solve(A, np.eye(A.shape[0]))
    normaA = normaMatMC(A, 2, 2, 10000)
    normaA_ = normaMatMC(A_, 2, 2, 10000)
    condA = condMC(A, 2, 10000)
    print("ANTES DE ASSERT condMC 1: ")
    print("normaA: ", normaA)
    print("normaA_: ", normaA_)
    print("mult normas: ", normaA[0] * normaA_[0])
    print("condA: ", condA)
    assert (np.allclose(normaA[0] * normaA_[0], condA, atol=1e-3))

    A = np.array([[3, 2], [4, 1]])
    A_ = np.linalg.solve(A, np.eye(A.shape[0]))
    normaA = normaMatMC(A, 2, 2, 10000)
    normaA_ = normaMatMC(A_, 2, 2, 10000)
    condA = condMC(A, 2, 10000)
    assert (np.allclose(normaA[0] * normaA_[0], condA, atol=1e-3))

    # Test condExacta

    A = np.random.rand(10, 10)
    A_ = np.linalg.solve(A, np.eye(A.shape[0]))
    normaA = normaExacta(A, 1)
    normaA_ = normaExacta(A_, 1)
    condA = condExacta(A, 1)
    assert (np.allclose(normaA * normaA_, condA))

    A = np.random.rand(10, 10)
    A_ = np.linalg.solve(A, np.eye(A.shape[0]))
    normaA = normaExacta(A, 'inf')
    normaA_ = normaExacta(A_, 'inf')
    condA = condExacta(A, 'inf')
    assert (np.allclose(normaA * normaA_, condA))

def test_labo4():
    # Tests LU

    L0 = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]])
    U0 = np.array([[10, 1, 0], [0, 2, 1], [0, 0, 1]])
    A = L0 @ U0
    L, U, nops = calculaLU(A)
    assert (np.allclose(L, L0))
    assert (np.allclose(U, U0))

    L0 = np.array([[1, 0, 0], [1, 1.001, 0], [1, 1, 1]])
    U0 = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    A = L0 @ U0
    L, U, nops = calculaLU(A)
    assert (not np.allclose(L, L0))
    assert (not np.allclose(U, U0))
    assert (np.allclose(L, L0, atol=1e-3))
    assert (np.allclose(U, U0, atol=1e-3))
    assert (nops == 13)

    L0 = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
    U0 = np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]])
    A = L0 @ U0
    L, U, nops = calculaLU(A)
    assert (L is None)
    assert (U is None)
    assert (nops == 0)

    ## Tests res_tri

    A = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
    b = np.array([1, 1, 1])
    assert (np.allclose(res_tri(A, b), np.array([1, 0, 0])))
    b = np.array([0, 1, 0])
    assert (np.allclose(res_tri(A, b), np.array([0, 1, -1])))
    b = np.array([-1, 1, -1])
    assert (np.allclose(res_tri(A, b), np.array([-1, 2, -2])))
    b = np.array([-1, 1, -1])
    assert (np.allclose(res_tri(A, b, inferior=False), np.array([-1, 1, -1])))

    A = np.array([[3, 2, 1], [0, 2, 1], [0, 0, 1]])
    b = np.array([3, 2, 1])
    assert (np.allclose(res_tri(A, b, inferior=False), np.array([1 / 3, 1 / 2, 1])))

    A = np.array([[1, -1, 1], [0, 1, -1], [0, 0, 1]])
    b = np.array([1, 0, 1])
    assert (np.allclose(res_tri(A, b, inferior=False), np.array([1, 1, 1])))

    # Test inversa

    ntest = 10
    iter = 0
    while iter < ntest:
        A = np.random.random((4, 4))
        A_ = inversa(A)
        if not A_ is None:
            assert (np.allclose(np.linalg.inv(A), A_))
            iter += 1

    # Matriz singular devería devolver None
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert (inversa(A) is None)

    # Test LDV:

    L0 = np.array([[1, 0, 0], [1, 1., 0], [1, 1, 1]])
    D0 = np.diag([1, 2, 3])
    V0 = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    A = L0 @ D0 @ V0
    L, D, V, nops = calculaLDV(A)
    assert (np.allclose(L, L0))
    assert (np.allclose(D, D0))
    assert (np.allclose(V, V0))

    L0 = np.array([[1, 0, 0], [1, 1.001, 0], [1, 1, 1]])
    D0 = np.diag([3, 2, 1])
    V0 = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1.001]])
    A = L0 @ D0 @ V0
    L, D, V, nops = calculaLDV(A)
    assert (np.allclose(L, L0, 1e-3))
    assert (np.allclose(D, D0, 1e-3))
    assert (np.allclose(V, V0, 1e-3))

    # Tests SDP

    L0 = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
    D0 = np.diag([1, 1, 1])
    A = L0 @ D0 @ L0.T
    assert (esSDP(A))

    D0 = np.diag([1, -1, 1])
    A = L0 @ D0 @ L0.T
    assert (not esSDP(A))

    D0 = np.diag([1, 1, 1e-16])
    A = L0 @ D0 @ L0.T
    assert (not esSDP(A))

    L0 = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
    D0 = np.diag([1, 1, 1])
    V0 = np.array([[1, 0, 0], [1, 1, 0], [1, 1 + 1e-10, 1]]).T
    A = L0 @ D0 @ V0
    assert (not esSDP(A))

def test_labo5():
    # --- Matrices de prueba ---
    A2 = np.array([[1., 2.],
                   [3., 4.]])

    A3 = np.array([[1., 0., 1.],
                   [0., 1., 1.],
                   [1., 1., 0.]])

    A4 = np.array([[2., 0., 1., 3.],
                   [0., 1., 4., 1.],
                   [1., 0., 2., 0.],
                   [3., 1., 0., 2.]])

    # --- TESTS PARA QR_by_GS2 ---
    Q2, R2 = QR_con_GS(A2)
    check_QR(Q2, R2, A2)

    Q3, R3 = QR_con_GS(A3)
    check_QR(Q3, R3, A3)

    Q4, R4 = QR_con_GS(A4)
    check_QR(Q4, R4, A4)

    # --- TESTS PARA QR_by_HH ---
    Q2h, R2h = QR_con_GS(A2)
    check_QR(Q2h, R2h, A2)

    Q3h, R3h = QR_con_HH(A3)
    check_QR(Q3h, R3h, A3)

    Q4h, R4h = QR_con_HH(A4)
    check_QR(Q4h, R4h, A4)

    # --- TESTS PARA calculaQR ---
    Q2c, R2c = calculaQR(A2, metodo='RH')
    check_QR(Q2c, R2c, A2)

    Q3c, R3c = calculaQR(A3, metodo='GS')
    check_QR(Q3c, R3c, A3)

    Q4c, R4c = calculaQR(A4, metodo='RH')
    check_QR(Q4c, R4c, A4)

correrTestsLabos()
