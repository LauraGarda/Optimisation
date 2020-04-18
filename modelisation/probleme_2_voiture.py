import matplotlib.pyplot as plt 
import numpy as np
import time
## Constantes

t_init = 0
t_fin = 1 
a = 0.00184507
b = -0.00014853
P_max = 2000
U = 400
lambda0 = np.array([[ 1 for _ in range(12)]], dtype = float)
x0 = np.array([ np.array([0,0]) for _ in range(10)],dtype = float)


GLOBAL_dt = 1e-1
GLOBAL_Delta_charge = np.array([0.001,0.001])
GLOBAL_K = U/(a*P_max*GLOBAL_dt)
print(GLOBAL_K*GLOBAL_Delta_charge)
## Fonction principale et contraintes

def cout(t):
	if t < 0.5:
		return 1
	else:
		return 2

def f(num,V):
	global GLOBAL_dt
	Sum = 0

	for (i,v) in enumerate(V): # Pour chaque pas de temps
		Sum += sum(v)*cout(i*GLOBAL_dt)

	return Sum/i

def grad_f(num, V):
	global GLOBAL_dt
	grad = []
	n = len(V.T)

	for i in range(n):
		grad.append(cout(i*GLOBAL_dt))

	return np.array([grad])


def contrainte(num, V):
	global GLOBAL_K
	global GLOBAL_Delta_charge
	C = []
	for v in V[0]:
		C.append(v*(v-1))
	C.append(GLOBAL_K*GLOBAL_Delta_charge[num] - sum(V.T))
	C.append(sum(V.T) - 1)
	return np.array(C)

def grad_contraintes(num, V):
	grad_C = []
	n = len(V.T)
	for (i,v) in enumerate(V.T):
		grad_C.append([0 for _ in range(n)])
		grad_C[i][i] = 2*v - 1
	grad_C.append([-1 for _ in range(n)])
	grad_C.append([1 for _ in range(n)])
	return grad_C

def optim_gradient_fixed_step(grad_fun, num, lam, x0, l, max_iter = 100000, epsilon_grad_fun = 1e-12):
    xk = np.array([1*(x0.T)[num]])
    number_iter = 0
    while number_iter < max_iter and np.linalg.norm(grad_fun(num, lam, xk)) > epsilon_grad_fun :
        number_iter += 1
        xk -= l*grad_fun(num, lam, xk)
    print("ok")
    return xk

def decomposition(grad_f, grad_c, num_max, lam, x0, l):
	liste_solutions = []
	def grad_L(num, lam, x):
		return grad_f(num, x) + np.dot(lam,grad_c(num, x))

	for num in range(num_max):
		liste_solutions.append(optim_gradient_fixed_step(grad_L, num, lam, x0, l))

	return liste_solutions
	
def coordination(liste_solutions, lam, rho, c):
	contraintes = 0
	for (i,solution) in enumerate(liste_solutions):
		contraintes += c(i, solution)
	for i in range(len(lam[0])):
		lam[0][i] = max(0, lam[0][i] + rho*contraintes[i])

	return lam

def decomposition_coordination(grad_f, c, grad_c, num_max, lambda0, x0, l, rho, max_iter = 100_000, eps_diff_lam = 1e-8):
	lam = 1*lambda0
	x = 1*x0
	num_iter = 0

	liste_solutions = decomposition(grad_f, grad_c, num_max, lam, x, l)
	temp = 1*lam
	lam = coordination(liste_solutions, lam, rho, c)
	print(lam)
	print("---------------")

	while num_iter < max_iter and np.linalg.norm(lam - temp) > eps_diff_lam:

		liste_solutions = decomposition(grad_f, grad_c, num_max, lam, x, l)
		print(liste_solutions[0])
		print("-----------------")
		temp = 1*lam
		lam = coordination(liste_solutions, lam, rho, c)
	return liste_solutions

X = decomposition_coordination(grad_f, contrainte, grad_contraintes, 2, lambda0, x0, 1e-3, 1e-3)	
print(X)

