import matplotlib.pyplot as plt 
import numpy as np
import time
## Constantes

a = 0.00184507
b = -0.00014853  # Négligé ici
P_max = 2000
U = 400  # Tension 

lambda0 = np.hstack((np.array([[ 1 for _ in range(11)] for _ in range(2)], dtype = float), np.array([[ 1 for _ in range(10)] for _ in range(2)], dtype = float)))
x0 = np.array([ [0,0] for _ in range(10)],dtype = float)

GLOBAL_dt = 1e-1
GLOBAL_Delta_charge = 0.005
GLOBAL_K = U/(a*P_max*GLOBAL_dt)
GLOBAL_num_voitures = 2


print(GLOBAL_K*GLOBAL_Delta_charge)

def cout(t):
    if t < 0.5:
        return 1
    else:
        return 2

GLOBAL_Cout = np.array([ cout(i * t) for i in range(int(1/dt))])

def f(V):
    global GLOBAL_dt
    Sum = 0

    for (i,v) in enumerate(V): # Pour chaque pas de temps
        Sum += sum(v)*cout[i]

    return Sum/i

def grad_f(V):
    global GLOBAL_dt
    grad = []
    n = len(V)

    for i in range(n):
        grad.append(cout(i*GLOBAL_dt))

    return np.array([grad])

def contrainte_couplee(V):
    return sum(V.T) - 1
    
def contraintes_solo(num, V):
    global GLOBAL_K
    global GLOBAL_Delta_charge
    global GLOBAL_P_allouable
    C = []
    for (k,v) in enumerate(V):
        C.append(v*(v - 1))
    C.append(GLOBAL_K*GLOBAL_Delta_charge - sum(V))
    for v in V:
        C.append(0)                                        ## La contrainte couplée est représentée ici
    return np.array(C)

def grad_contraintes_solo(num, V):
    n = len(V)
    grad_C = []
    for (i,v) in enumerate(V):
        grad_C.append([0 for _ in range(n)])
        grad_C[i][i] = 2*v - 1
        
    grad_C.append([-1 for _ in range(n)])
    for i in range(n):
        grad_C.append([0 for _ in range(n)])
        grad_C[11 + i][i] = 1
    return np.array(grad_C)

def Uzawa_1_voiture(Grad_L, c, lambda0, x0, l, rho, num, max_iter = 1000000, epsilon_grad_L = 1e-3):
    lam = lambda0
    xk = 1*x0
    def update_lam(lam, rho, c, xk):
        C = c(num, xk)
        for i in range(len(lam)):
            lam[i] = max(0, lam[i] + rho*C[i])
    grad_l = Grad_L(lam, xk, num)
    pk = -1*grad_l
    xk += l*pk[0]
    update_lam(lam, rho, c, xk)
    num_iter = 0
    
    while num_iter < max_iter and np.linalg.norm(grad_l) > epsilon_grad_L:
        grad_l = Grad_L(lam, xk, num)
        pk = -1*grad_l
        xk += l*pk[0]
        update_lam(lam, rho, c, xk)
        num_iter += 1
    return xk
    
def decomposition(grad_f, c, grad_c, num_max, lam, x0, l, rho_solo):
    def grad_L(lam, x, num):
        return grad_f(x) + np.dot(lam,grad_c(num, x))
    solutions = Uzawa_1_voiture(grad_L, c, lam[0], (x0.T)[0], l, rho_solo, 0)

    for num in range(1,num_max):
        solutions = np.dstack((solutions, [Uzawa_1_voiture(grad_L, c, lam[num], (x0.T)[num], l, rho_solo, num)]))[0]
    return solutions

# print(decomposition(grad_f, contraintes_solo, grad_contraintes_solo, 2, lambda0, x0, 1e-2, 1e-2))
# print(GLOBAL_P_allouable)

def coordination(solutions, lam, rho, c):
    global GLOBAL_num_voitures
    global GLOBAL_dt
    contraintes = c(solutions)
    for (i,c) in enumerate(contraintes):
        for k in range(GLOBAL_num_voitures):
            lam[k][int(1/GLOBAL_dt) + 1 + i] = max(0, lam[k][int(1/GLOBAL_dt) + 1 + i] + rho*c)

def decomposition_coordination(grad_f, c_solo, c_couplee, grad_c, num_max, lambda0, x0, l, rho_solo, rho, max_iter = 100_000, eps_diff_lam = 1e-5):
    lam = 1*lambda0
    num_iter = 0
    solutions = decomposition(grad_f, c_solo, grad_c, num_max, lam, x0, l, rho_solo)
    temp = 1*lam
    print(solutions)
    Liste = [[ solutions[3][0], solutions[3][1] ]]
    #print(solutions)
    coordination(solutions, lam, rho, c_couplee)
    while num_iter < max_iter and np.linalg.norm(lam - temp) > eps_diff_lam:    
        solutions = decomposition(grad_f, c_solo, grad_c, num_max, lam, solutions, l, rho_solo)
        temp = 1*lam
        #print("lam")
        #print(lam)
        coordination(solutions, lam, rho, c_couplee)
        # print("solutions")
        num_iter += 1
        print(solutions)
        Liste.append( [ solutions[3][0], solutions[3][1]  ])
        print(num_iter)
        if num_iter == 5500 : return Liste
    print("num iter")
    print(num_iter)
    return solutions

X = decomposition_coordination(grad_f, contraintes_solo, contrainte_couplee, grad_contraintes_solo, 2, lambda0, x0, 1e-1, 1e-1, 1e-4)	

plt.figure()
plt.title("évolution de la puissance allouée aux deux voitures sur un temps précis")
plt.xlabel("iteration")
plt.ylabel("puissance allouée")
numbers = [ k for k in range(len(X))]
plt.plot(numbers, [ x[0] for x in X] , label = "voiture 1")
plt.plot(numbers, [ x[1] for x in X] , label = "voiture 2")
plt.legend(loc = 0)
plt.show()