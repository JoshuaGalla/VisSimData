import numpy as np
import numba as nb

def kernel(x, x_j, var, gamma): #rbf kernel
    K=np.dot(gamma*x,x_j.T)
    #element-wise multiply (*)

    TMP_x=np.empty(x.shape[0],dtype=x.dtype)
    for i in nb.prange(x.shape[0]):
         TMP_x[i]=np.dot(gamma,(x[i]**2))

    TMP_x_j=np.empty(x_j.shape[0],dtype=x_j.dtype)
    for i in nb.prange(x_j.shape[0]):
        TMP_x_j[i]=np.dot(gamma,(x_j[i]**2))

        
    for i in nb.prange(x.shape[0]):
        for j in range(x_j.shape[0]):
            K[i,j]=var*np.exp(-(-2.0*K[i,j]+TMP_x[i]+TMP_x_j[j]))

    return K