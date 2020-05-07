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
lambda0 = np.array([ 1 for _ in range(11)], dtype = float)
x0 = np.array([[ 0 for _ in range(10)]],dtype = float).T


GLOBAL_dt = 1e-1
GLOBAL_K = U/(a*P_max*GLOBAL_dt)
GLOBAL_Delta_charge = 0.9/GLOBAL_K

## Fonction principale et contraintes

def cout(t):
	if t < 0.5:
		return 1
	else:
		return 2

def batt_derive(i):
	return a*i

def f(V):
	global GLOBAL_dt
	Sum = 0

	for (i,v) in enumerate(V): # Pour chaque pas de temps
		Sum += sum(v)*cout(i*GLOBAL_dt)

	return Sum/i

def grad_f(V):
	global GLOBAL_dt
	grad = []
	n = len(V)

	for i in range(n):
		grad.append(cout(i*GLOBAL_dt))

	return np.array([grad])


def contraintes(V):
	global GLOBAL_K
	global GLOBAL_Delta_charge
	C = []
	for v in V:
		C.append(v*(v-1))
	C.append(GLOBAL_K*GLOBAL_Delta_charge - sum(V))
	return np.array(C)


def grad_contraintes(V):
	grad_C = []
	n = len(V)

	for (i,v) in enumerate(V):
		grad_C.append([0 for _ in range(n)])
		grad_C[i][i] = 2*v[0] - 1
	grad_C.append([-1 for _ in range(n)])

	return grad_C


def Uzawa_1_voiture(f, grad_f, c, grad_c, x0, l, rho, lambda0, max_iter = 100000, epsilon_grad_L = 1e-11):
	lam = 1*lambda0
	xk = 1*x0

	def Grad_L(grad_f, lam, grad_c, xk):
		return (grad_f(xk) + np.dot(lam,grad_c(xk))).T
	def update_lam(lam, rho, c, xk):
		C = c(xk)
		print(xk)
		print("C is ")
		print(C)
		for i in range(len(lam)):
			lam[i] = max(0, lam[i] + rho*C[i])
	grad_l = Grad_L(grad_f, lam, grad_c, xk)
	pk = -1*grad_l
	xk += l*pk
	update_lam(lam, rho, c, xk)
	num_iter = 0
	
	while num_iter < max_iter and np.linalg.norm(grad_l) > epsilon_grad_L:
		grad_l = Grad_L(grad_f, lam, grad_c, xk)
		pk = -1*grad_l
		xk += l*pk
		print("lam is")
		print(lam)
		update_lam(lam, rho, c, xk)
		num_iter += 1
	print(num_iter)
	return xk

X = Uzawa_1_voiture(f, grad_f, contraintes, grad_contraintes, x0, 1e-1, 1e-1, lambda0)
# print(X)

