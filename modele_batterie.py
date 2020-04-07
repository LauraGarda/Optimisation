
## Packages

import os
import matplotlib.pyplot as plt 

## lecture du fichier

path = "donnees-projet-gr1.txt"

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
				Liste_Liste[index].append(temp)
				temp = ""
				index += 1
		else:
			temp += char

Liste_Liste[index].append(temp) ## On ajoute le dernier élément, qui n'est pas détecté (la denière valeur de batterie) / On pourrait aussi rajouter un saut de ligne à la fin du doc

#print(Liste_t)
#print(Liste_intens)
#print(Liste_batt)

plt.figure()
plt.xlabel("temps")
plt.ylabel("valeurs")
plt.plot(Liste_t,Liste_batt, label = "batterie")
plt.plot(Liste_t, Liste_intens, label = "intensité")
plt.legend(loc = 0)
plt.show()