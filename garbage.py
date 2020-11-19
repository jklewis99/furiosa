import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
plt.figure(figsize=(4, 4))
y = np.arange(0, 1, 0.01)
y_ = np.arange(0.01, 1.01, 0.01)
plt.scatter(y, y_)
plt.title('Actual vs Predicted')
plt.ylabel('Actual Value')
plt.xlabel('Predicted Value')
r_squared = r2_score(y, y_)
plt.annotate("r^2 value = {:.3f}\nMSE=0.000".format(r_squared), (0, 0.9))
plt.savefig("figure-garbage.png")
plt.show()