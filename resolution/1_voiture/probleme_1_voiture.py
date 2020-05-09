import matplotlib.pyplot as plt 
import numpy as np
import time
import scipy.optimize as scp
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
GLOBAL_Delta_charge = 0.0092

print(GLOBAL_K*GLOBAL_Delta_charge)

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
		Sum += v*cout(i*GLOBAL_dt)

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
		update_lam(lam, rho, c, xk)
		num_iter += 1
	return num_iter

cons = ({'type': 'eq', 'fun': lambda V: - GLOBAL_K*GLOBAL_Delta_charge + sum(V)})
bnds = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1))
Liste_moi = []
Liste_mu = []
mu = 0
plt.figure()
for i in range(-20,15):
	mu = 1*10**(-i/10)
	Liste_mu.append(mu)
	Liste_actuelle = []
	for k in range(100):
		GLOBAL_Delta_charge = 0.0092 * (k + 1) / 100 
		print(GLOBAL_Delta_charge * GLOBAL_K)
		n = Uzawa_1_voiture(f, grad_f, contraintes, grad_contraintes, x0, mu, mu, lambda0)
		Liste_actuelle.append(n)
	Liste_moi.append( sum(Liste_actuelle) / 100)

plt.xlabel("steps")
plt.ylabel("average number of iterations")
plt.xscale('log')
plt.title("influence of the steps over the average number of iterations")
plt.plot(Liste_mu, Liste_moi)
plt.legend(loc = 0)
plt.show()