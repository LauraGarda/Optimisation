import matplotlib.pyplot as plt
import numpy as np
import time
# Constantes

a = 0.00184507
b = -0.00014853   # Négligé ici (ordonnée à l'origine du modèle de charge de batterie)
P_max = 2000
U = 400           # Tension secteur

x0 = np.array([0 for _ in range(10)], dtype = float)
P_allouable = np.array([1 for _ in range(10)], dtype = float)
lambda0 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype = float)
Var = np.array([1*x0, 1*x0, 1*P_allouable, 1*P_allouable])

GLOBAL_dt = 1e-1
GLOBAL_Delta_charge = np.array([0.0035, 0.0035])
GLOBAL_K = U/(a*P_max*GLOBAL_dt)

GLOBAL_P_necessaire = GLOBAL_K * GLOBAL_Delta_charge


def cout(t):
    return (1 - t)**2 + 1 

GLOBAL_Cout = np.array([ cout (i*GLOBAL_dt) for i in range(int(1/GLOBAL_dt))])

def forbiden(num, i):
    if num == 0:
        if i < 4:
            return 1
        else:
            return 0
    elif num == 1:
        if i > 7:
            return 1
        else:
            return 0


def transformation_probleme(forbiden, Var):
    global GLOBAL_dt
    global GLOBAL_P_necessaire
    global GLOBAL_Cout

    n = len(Var[0])
    # Gradient de f
    Grad_f = []
    for num in range(2):
        Sous_grad = []
        for i in range(n):
            if not forbiden(num, i):
                Sous_grad.append(GLOBAL_Cout[i])
            else:
                Sous_grad.append(0)
        Grad_f.append(Sous_grad)

    Grad_f = np.array(Grad_f, dtype = float)

    # Puissance allouable

    for num in range(2):
        for i in range(n):
            if forbiden(num, i):
                Var[2 + num][i] = 0

    # contraintes sur P

    def contraintes_solo(num, X, P):
        global GLOBAL_P_necessaire
        contraintes = []
        n = len(X)
        for i in range(n):
            if forbiden(num, i):
                contraintes.append(0)
            else:
                contraintes.append(X[i] * (X[i] - P[i]))

        contraintes.append( - np.sum(X) + GLOBAL_P_necessaire[num])
        return np.array(contraintes)

    def grad_contraintes_solo(num, X, P):
        grad = []
        n = len(X)

        for i in range(n):
            grad.append([0 for _ in range(n)])
            if not forbiden(num, i):
                grad[i][i] = 2*X[i] - P[i]

        grad.append([ -1 for _ in range(n)])
        for i in range(n):
            if forbiden(num, i):
                grad[-1][i] = 0

        return np.array(grad)

    def coordination(Var, rho):
        global GLOBAL_P_necessaire
        n  = len(Var[0])

        surchage = []

        for i in range(n):
            if Var[0][i] + Var[1][i] >= 1:
                surchage.append( Var[0][i] + Var[1][i] - 1)
            else:
                surchage.append(0)

        for num in range(2):
            for i in range(n):
                if not forbiden(num, i):
                    Var[num + 2][i] -= rho*surchage[i]

        while ( sum(Var[2]) - GLOBAL_P_necessaire[0] < 0 ) and ( sum(Var[3]) - GLOBAL_P_necessaire[1] < 0 ):
            for num in range(2):
                n_dispos = 0
                indices_dispos = []
                for i in range(n):
                    if not forbiden(num, i) and Var[num + 2][i] != 0 and Var[num + 2][i] != 1:
                        n_dispos += 1
                        indices_dispos.append(i)

            for i in indices_dispos:
                Var[num + 2][i] = min(1 , Var[num+2][i] + (GLOBAL_P_necessaire[num] - sum(Var[num + 2]))/n_dispos)

    return Grad_f, Var, contraintes_solo, grad_contraintes_solo, coordination

grad_f, Var, contraintes_solo, grad_contraintes_solo, coordination = transformation_probleme(forbiden, Var)


def Uzawa_1_voiture(Grad_L, c, lambda0, xk, P_max, l, rho, num, max_iter = 1000000, epsilon_grad_L = 1e-3):
    lam = lambda0
    def update_lam(lam, rho, c, xk):
        C = c(num, xk, P_max)
        for i in range(len(lam)):
            lam[i] = max(0, lam[i] + rho*C[i])
    grad_l = Grad_L(lam, xk, P_max, num)
    pk = -1*grad_l
    xk += l*pk
    update_lam(lam, rho, c, xk)
    num_iter = 0

    while num_iter < max_iter and np.linalg.norm(grad_l) > epsilon_grad_L:
        grad_l = Grad_L(lam, xk, P_max, num)
        pk = -1*grad_l
        xk += l*pk
        update_lam(lam, rho, c, xk)
        num_iter += 1

def decomposition(grad_f, c, grad_c, num_max, lam, Var, l, rho_solo):
    def grad_L(lam, X, P, num):
        return grad_f[num] + np.dot(lam,grad_c(num, X, P))
    Uzawa_1_voiture(grad_L, c, lam[0], Var[0], Var[2], l, rho_solo, 0)
    Uzawa_1_voiture(grad_L, c, lam[1], Var[1], Var[3], l, rho_solo, 1)

def decomposition_coordination(grad_f, c_solo, grad_c, num_max, lambda0, Var, l, rho_solo, rho, max_iter = 100_000, eps_diff = 1e-10):
    lam = 1*lambda0
    num_iter = 0
    decomposition(grad_f, c_solo, grad_c, num_max, lam, Var, l, rho_solo)
    Puissances_consommees = Var[0] + Var[1]
    Listes_puissances = [Puissances_consommees]
    # Liste = [[ Var[0][3], Var[1][3] ]]
    coordination(Var, rho)
    while num_iter < max_iter and (Puissances_consommees >= 1 + eps_diff).any() :    
        decomposition(grad_f, c_solo, grad_c, num_max, lam, Var, l, rho_solo)
        temp = 1*lam
        coordination(Var, rho)
        Puissances_consommees = Var[0] + Var[1]
        Listes_puissances.append(Puissances_consommees)
        num_iter += 1
        # Liste.append( [ Var[0][3], Var[1][3] ] )
        print( num_iter)
    return Listes_puissances

X = decomposition_coordination(grad_f, contraintes_solo, grad_contraintes_solo, 2, lambda0, Var, 1e-2, 1e-2, 1e-2)
plt.figure()
plt.title("évolution de la puissance allouée aux deux voitures sur un temps précis")
plt.xlabel("iteration")
plt.ylabel("puissance allouée")
numbers = [ k for k in range(len(X))]
for k in range(10):
    plt.plot(numbers, [ x[k] for x in X], label = str(k))
plt.legend(loc = 0)
plt.show()