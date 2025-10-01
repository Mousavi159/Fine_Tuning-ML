import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error

# Load saved objects
best_forest = joblib.load("best_forest.pkl")
y_test = np.load("y_test.npy", allow_pickle=True)
y_pred = np.load("y_pred.npy", allow_pickle=True)


models = ["LR", "DT", "SVM", "RF"]
grid_cv_rmse = [67866, 58725, 69000 , 49673]
grid_test_rmse = [71076, 59890, 75125, 49198, ]

rnd_cv_rmse = [67874, 59587, 69000, 49047]
rnd_test_rmse = [72739, 59695, 75127, 48622]



#CV Bar chat
x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, grid_cv_rmse, width, label="Grid Search", color="hotpink")
plt.bar(x + width/2, rnd_cv_rmse,  width, label="Random Search", color="skyblue")

plt.xticks(x, models)
plt.ylabel("RMSE")
plt.title("Grid vs Random Search (CV RMSE)")
plt.legend()
plt.show()

# Scatter plot for Predicted vs Actual values
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Random Forest: Predicted vs Actual")
plt.show()

# Distribution error for the best model
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Residuals")
plt.title("Random Forest: Error Distribution")
plt.show()