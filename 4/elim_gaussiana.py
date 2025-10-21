#ESTAMOS TRABAJANDO EN MODULO ALC #ESTAMOS TRABAJANDO EN MODULO ALC#ESTAMOS TRABAJANDO EN MODULO ALC#ESTAMOS TRABAJANDO EN MODULO ALC#ESTAMOS TRABAJANDO EN MODULO ALC#ESTAMOS TRABAJANDO EN MODULO ALC
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np

def elim_gaussiana(A):
    cant_op = 0
    filas=A.shape[0]
    columnas=A.shape[1]
    Ac = A.copy()
    
    if filas!=columnas:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR
    cant_op = 0
    for i_diagonal in range(filas):
        for i_fila in range(i_diagonal+1, filas):

            coeficiente = Ac[i_fila][i_diagonal] / Ac[i_diagonal][i_diagonal]
            cant_op += 1
            Ac[i_fila][i_diagonal] = coeficiente

            for i_columna_Suma in  range(i_diagonal+1, columnas ): 
                Ac[i_fila][i_columna_Suma] = Ac[i_fila][i_columna_Suma] - (coeficiente*Ac[i_diagonal][i_columna_Suma])
                cant_op += 2

    L = np.zeros((filas,columnas))
    U = Ac 

    for i_fila_l in range(filas):
        for i_columna_l in range(0,i_fila_l+1):

            if(i_fila_l == i_columna_l):
                L[i_fila_l][i_columna_l] = 1
            else:
                L[i_fila_l][i_columna_l] = Ac[i_fila_l][i_columna_l]
                U[i_fila_l][i_columna_l] = 0
    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre 
    ## la matriz Ac
            
    
    return L, U, cant_op


def main():
    n = 7
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    print('Matriz B \n', B)
    
    L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

if __name__ == "__main__":
    main()
    
    
