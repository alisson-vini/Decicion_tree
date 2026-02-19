import numpy as np
import matplotlib.pyplot as plt

# classe aprovado
x1 = np.array([0, 0.3, 0.5, 0.7, 0.9, 1])
y1 = np.array([1, 1, 1.2, 1.5, 1.7, 2])

# classe reprovado
x2 = np.array([0, 0.3, 0.5, 1,  1.2, 1.5, 1.6, 1.9,  1.2, 1.5, 1.6, 1.9])
y2 = np.array([0.3, 0.5, 0.7, 0.8,  0.3, 0.5, 0.7, 0.8,  1.2, 1.5, 1.6, 1.9])

plt.scatter(x1, y1, color="green", alpha=0.75)
plt.scatter(x2, y2, color="red", alpha=0.75)

# retas (thresholds)
x_unicos = np.unique(np.concatenate( (x1,x2) ))
x_unicos = np.sort(x_unicos)
thresholds = (x_unicos[:-1] + x_unicos[1:]) / 2

for i in range(len(thresholds)):
    plt.axvline(x=thresholds[i])


plt.title("Decisão de sair de casa")
plt.xlabel("Quantidade de chuva")
plt.ylabel("Quantidade de Sol")

plt.show()