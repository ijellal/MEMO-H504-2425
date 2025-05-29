import numpy as np
import matplotlib.pyplot as plt
import os

# Chemin vers le fichier CSV à afficher
csv_path_1 = '1_11_21im\logs\plenoptic_test_v20/psnr_log.csv'
# C:\Users\monpc\Desktop\nerf-pytorch-master - Copy\1_11_21im\logs\plenoptic_test_v20

# Charger les données PSNR du premier fichier
data1 = np.loadtxt(csv_path_1, delimiter=',', skiprows=1)
iters1 = data1[:, 0]
psnrs1 = data1[:, 1]

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(iters1, psnrs1, label='Fujita', color='blue', linewidth=2)

# Ajout de détails
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.title('PSNR evolution for Fujita')
plt.legend()
plt.grid(True)

# Sauvegarder ou afficher
plt.savefig('comparaison_psnr_with_multiview.png')  # Modifie le chemin si besoin
plt.show()



"""
import numpy as np
import matplotlib.pyplot as plt
import os

# Chemins vers les deux fichiers CSV à comparer
csv_path_1 = '1im/logs/plenoptic_test_v14/psnr_log.csv'
csv_path_2 = 'rabbit/logs/plenoptic_test_v17/psnr_log.csv'

# 'param/fujita_cameras.txt'

# Charger les données PSNR du premier fichier
data1 = np.loadtxt(csv_path_1, delimiter=',', skiprows=1)
iters1 = data1[:, 0]
psnrs1 = data1[:, 1]

# Charger les données PSNR du deuxième fichier
data2 = np.loadtxt(csv_path_2, delimiter=',', skiprows=1)
iters2 = data2[:, 0]
psnrs2 = data2[:, 1]

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(iters1, psnrs1, label='Fujita', color='blue', linewidth=2)
plt.plot(iters2, psnrs2, label='Rabbit', color='orange', linewidth=2)

# Ajout de détails
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.title('Comparison of PSNR evolution for Fujita and Rabbit')
plt.legend()
plt.grid(True)

# Sauvegarder ou afficher
plt.savefig('comparaison_psnr_with_rbga.png')  # Modifie le chemin si besoin
plt.show()
"""
