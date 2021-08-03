from poly2d import Poly2D
import numpy as np



def energy_x_grad_general(T,P,x,Gs,omegas):
        ###
            # work out the energy gradient
            # criteria for minimum =0 
        ### TODO check this is vectorised correctly
        ### TODO it seems just adding G_BA and Omega gets a square matrix rather than a vector (if they're both vectors.)
        ### TODO make a rank-3 tensor like the energy is? 


        A = Gs
        B = np.log(x/(1-x)) 
        C = omegas * (1-2*x)


        A = A.reshape(C.shape) ## if a matrix, make a vector## to deal with the cases where C is a N by 1 matrix, rather than a vector
        return A + (T*(B + C).T).T
        #return A + B + C

def energy_x_grad_biddle(T,P,x,poly_B:Poly2D,omega0:float):


    A = poly_B.grid(T,P)
    B = np.log(x/(1-x)) 
    
    
    C = bid_func_omega(T,P) * (1-2*x)


    A = A.reshape(C.shape) ## if a matrix, make a vector## to deal with the cases where C is a N by 1 matrix, rather than a vector
    return A + B + C

def bid_func_omega(t, del_p, omega_0): 
            return np.outer((1/t), (2 + omega_0*del_p))



