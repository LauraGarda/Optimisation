
## Packages

import os
import matplotlib.pyplot as plt 
import numpy as np

## Modélisation de la batterie

	## lecture du fichier

path = "modelisation/donnees-projet-gr1.txt"

file = open(path)
content = file.readlines()

Liste_t = []
Liste_intens = []
Liste_batt = []

Liste_Liste = [Liste_t, Liste_intens, Liste_batt]

for line in content:
	temp = ""
	index = 0
	for char in line:
		if char == '\t' or char == '\n':  ## On a un espace ou un saut de ligne
			if temp != "":
				Liste_Liste[index].append(float(temp))
				temp = ""
				index += 1
		else:
			temp += char

Liste_t = np.array(Liste_t)
Liste_intens= np.array(Liste_intens[1:])
Liste_batt = np.array(Liste_batt)
Liste_derive_batt = Liste_batt[1:] - Liste_batt[:-1]

## Identification des palliers de charge

Pallier_1 = []  # -4 < x -3
Pallier_2 = []	# -3 < x < -2
Pallier_3 = []	# -2 < x < -0.7
Pallier_4 = []	# -0.7 < x < -0.2
Pallier_5 = []	# -0.2 < x < 0.2
Pallier_6 = []  # 0.2 < x < 0.7
Pallier_7 = [] 	# 0.7 < x < 1.5

Palliers = [Pallier_1, Pallier_2, Pallier_3, Pallier_4, Pallier_5, Pallier_6, Pallier_7]

def search_pallier(intens):
	if -4 < intens < -3: return 0
	elif -3 < intens < -2: return 1
	elif -2 < intens < -.7 : return 2
	elif -.7 < intens < -0.2 : return 3
	elif -.2 < intens < .2: return 4
	elif .2 < intens <.7: return 5
	elif .7 < intens < 1.5: return 6

for (k, intens) in enumerate(Liste_intens):
	num = search_pallier(intens)
	Palliers[num].append(np.array([Liste_intens[k], Liste_derive_batt[k]]))
	
Liste_intens_moyenne = []
Liste_derive_batt_moyenne = []
	
for Pallier in Palliers:
	intens = np.mean(np.array([x[0] for x in Pallier]))
	batt = np.mean(np.array([x[1] for x in Pallier]))
	Liste_derive_batt_moyenne.append(batt)
	Liste_intens_moyenne.append(intens)

Liste_intens_moyenne = np.array(Liste_intens_moyenne)
Liste_derive_batt_moyenne = np.array(Liste_derive_batt_moyenne)

## Approximation linéaire

# Liste_coeff_directeurs = (Liste_derive_batt_moyenne[1:] - Liste_derive_batt_moyenne[:-1])/(Liste_intens_moyenne[1:] - Liste_intens_moyenne[:-1])
# Liste_ordonnees_origine = Liste_derive_batt_moyenne[1:] - Liste_coeff_directeurs*Liste_intens_moyenne[1:]
# 
# coeff_directeur = np.mean(Liste_coeff_directeurs)
# ordonnee_origine = np.mean(Liste_ordonnees_origine)
# 
# Liste_approx_linéaire = coeff_directeur*Liste_intens_moyenne + ordonnee_origine
# R_carre = 1 - np.sum((Liste_derive_batt_moyenne-Liste_approx_linéaire)**2)
# 
# 
# plt.figure()
# plt.ylabel("dérivée de la charge")
# plt.xlabel("intensité")
# plt.title("Test du modèle de décharge/recharge de la batterie")
# plt.plot(Liste_intens_moyenne, Liste_approx_linéaire, label = "approximation, R^2 = " + str(R_carre))
# plt.plot(Liste_intens_moyenne, Liste_derive_batt_moyenne, marker = '.', linestyle = " ", label = "donnees")
# plt.legend(loc = 0)
# plt.show()

## Moindres carrées

A = np.array([[intens, 1] for intens in Liste_intens_moyenne])
b = np.array([Liste_derive_batt_moyenne]).T
x = np.linalg.lstsq(A,b)
coeff_directeur, ordonnee_origine = x[0]
print(x[0])
# Liste_approx_lstsq = coeff_directeur*Liste_intens_moyenne + ordonnee_origine
# R_carre = 1 - np.sum((Liste_derive_batt_moyenne-Liste_approx_lstsq)**2)

# plt.figure()
# plt.ylabel("dérivée de la charge")
# plt.xlabel("intensité")
# plt.title("Test du modèle de décharge/recharge de la batterie")
# plt.plot(Liste_intens_moyenne, Liste_approx_lstsq, label = "approximation, R^2 = " + str(R_carre))
# plt.plot(Liste_intens_moyenne, Liste_derive_batt_moyenne, marker = '.', linestyle = " ", label = "donnees")
# plt.legend(loc = 0)
# plt.show()