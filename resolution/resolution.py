import numpy as np


C = np.array([1 for k in range(1000)])

def c(V):
    c1 = np.sum(V) - delta_charge * K
    contr = [c1]
    for v in V.T:
        contr.append(v*(v+1))
    return np.array(contr)

def grad_c(V):
    n = len(V.T)
    grad = []
    ligne_1 = [1 for _ in range(n)]
    grad.append(ligne_1)
    for (k,v) in enumerate(V.T):
        grad.append([0 for _ in range(n)])
        grad[k + 1][k] = 2*v + 1
    return np.array(grad)

def f(V):
    return sum


def Uzawa_fixe(f, grad_f, c, grad_c, x0, l, rho, lambda0 = 1.0, max_iter = 100000, epsilon_grad_L = 1e-8):
    lam = 1*lambda0
    xk = 1*x0
    grad_L = grad_f(xk) + np.dot(lam,grad_c(xk))
    pk = -1*grad_L
    xk += l*pk
    lam = max(0, lam + rho*c(xk))
    num_iter = 0
    
    while num_iter < max_iter and np.linalg.norm(grad_L) > epsilon_grad_L:
        grad_L = grad_f(xk) + lam*grad_c(xk)
        pk = -1*grad_L
        xk += l*pk
        lam = max(0, lam + rho*c(xk))
        num_iter += 1
    
    return xk