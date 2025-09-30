import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

models = ["LR", "DT", "SVM", "RF"]
grid_cv_rmse = [67866, 58725, 69000 , 49673]
grid_test_rmse = [71076, 59890, 75125, 49198, ]

rnd_cv_rmse = [67874, 59587, 69000, 49047]
rnd_test_rmse = [72739, 59695, 75127, 48622]



x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, grid_cv_rmse, width, label="Grid Search", color="hotpink")
plt.bar(x + width/2, rnd_cv_rmse,  width, label="Random Search", color="skyblue")

plt.xticks(x, models)
plt.ylabel("RMSE")
plt.title("Grid vs Random Search (CV RMSE)")
plt.legend()
plt.show()