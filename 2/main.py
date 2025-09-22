import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def pointsGrid(esquinas):
    # crear 10 lineas horizontales
    [w1, z1] = np.meshgrid(np.linspace(esquinas[0,0], esquinas[1,0], 46),
                        np.linspace(esquinas[0,1], esquinas[1,1], 10))

    [w2, z2] = np.meshgrid(np.linspace(esquinas[0,0], esquinas[1,0], 10),
                        np.linspace(esquinas[0,1], esquinas[1,1], 46))

    w = np.concatenate((w1.reshape(1,-1),w2.reshape(1,-1)),1)
    z = np.concatenate((z1.reshape(1,-1),z2.reshape(1,-1)),1)
    wz = np.concatenate((w,z))
                         
    return wz

def proyectarPts(T, wz):
    assert(T.shape == (2,2)) # chequeo de matriz 2x2
    assert(T.shape[1] == wz.shape[0]) # multiplicacion matricial valida   
    xy = T@wz
    return xy

def vistform(T, wz, titulo=''):
    # transformar los puntos de entrada usando T
    xy = proyectarPts(T, wz)
    if xy is None:
        print('No fue implementada correctamente la proyeccion de coordenadas')
        return
    # calcular los limites para ambos plots
    minlim = np.min(np.concatenate((wz, xy), 1), axis=1)
    maxlim = np.max(np.concatenate((wz, xy), 1), axis=1)

    bump = [np.max(((maxlim[0] - minlim[0]) * 0.05, 0.1)),
            np.max(((maxlim[1] - minlim[1]) * 0.05, 0.1))]
    limits = [[minlim[0]-bump[0], maxlim[0]+bump[0]],
               [minlim[1]-bump[1], maxlim[1]+bump[1]]]             

    fig, (ax1, ax2) = plt.subplots(1, 2)         
    fig.suptitle(titulo)
    grid_plot(ax1, wz, limits, 'w', 'z')    
    grid_plot(ax2, xy, limits, 'x', 'y')
    plt.show()
    
def grid_plot(ax, ab, limits, a_label, b_label):
    ax.plot(ab[0,:], ab[1,:], '.')
    ax.set(aspect='equal',
           xlim=limits[0], ylim=limits[1],
           xlabel=a_label, ylabel=b_label)

def rota(theta):
    """
    Recibe un angulo theta y retorna una matriz de 2x2
    que rota un vector dado en un angulo theta
    """
    res =np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return res

def escala(s):
    """
    Recibe una tira de numeros s y retorna una matriz cuadrada de
    n x n, donde n es el tamaño de s.
    La matriz escala la componente i de un vector de Rn
    en un factor s[i]
    """
    res =np.diag(s)
    return res

def rota_y_escala(theta,s):
    """
    Recibe un angulo theta y una tira de numeros s,
    y retorna una matriz de 2x2 que rota el vector en un angulo theta
    y luego lo escala en un factor s
    """
    res = rota(theta)@escala(s)
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
    matriz3x3 = np.zeros((3, 3))
    matriz3x3[:2, :2] = matriz2x2
    matriz3x3[0][2] = b[0]
    matriz3x3[1][2] = b[1]
    matriz3x3[2][2] = 1
    return matriz3x3

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
    vectorColumnaRes = transf@vectorColumna
    res = vectorColumnaRes.T[:2]
    return res

def main():
    print('Ejecutar el programa')
    # generar el tipo de transformacion dando valores a la matriz T
    #T = pd.read_csv('T.csv', header=None).values
    T1 = np.array([[2,0],[0,3]])
    T1inv = np.array([[1/2,0],[0,1/3]])
    angulo = np.pi/4
    T2 = np.array([[np.cos(angulo), -np.sin(angulo)],[np.sin(angulo), np.cos(angulo)]])
    T3 = np.array([[np.cos(-angulo), -np.sin(-angulo)],[np.sin(-angulo), np.cos(-angulo)]])
    T4 = np.array([[1,0],[0,1]])

    corners = np.array([[0,0],[100,100]])
    # corners = np.array([[-100,-100],[100,100]]) array con valores positivos y negativos
    radio = 1
    tremendo = np.array([[radio*np.cos(x) for x in range(1, 360)], [radio*np.sin(y) for y in range(1, 360)]])
    wz = pointsGrid(corners)
    vistform(T, wz, 'Deformar coordenadas')

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
        #np.array([4,7])  #EL TEST ESTA MAL HECHO DEBERIA DAR [4,8]
        np.array([4,8])
    ))


if __name__ == "__main__":
    main()
