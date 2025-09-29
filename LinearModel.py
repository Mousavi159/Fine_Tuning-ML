import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import seaborn as sns


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


    

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class Second_transformer(BaseEstimator, TransformerMixin):
    """
    Adds engineered features using numeric column *indices*:
    - rooms_per_household = total_rooms / households
    - population_per_household = population / households
    """
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 6]       # total_rooms / households
        population_per_household = X[:, 5] / X[:, 6]  # population / households
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 4] / X[:, 3]     # bedrooms / rooms
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
    
class Third_transformer(BaseEstimator, TransformerMixin):
    """Handling categorical data: Wraps sklearn's 
    OneHotEncoder to fit the same custom transformer pattern."""

    def __init__(self):
        self.cat_encoder = OneHotEncoder(sparse_output=False)
    
    def fit(self, X, y=None):
        self.cat_encoder.fit(X)
        return self    
    
    def transform(self, X):
        return self.cat_encoder.transform(X)
    
# Load & split
df = pd.read_csv("housing.csv")

X = df.drop("median_house_value", axis=1)  #input
Y = df["median_house_value"].copy()  #target


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


# Tell ColumnTransformer which columns are numeric vs categorical
num_attribs = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income"
]
cat_attribs = ["ocean_proximity"]


# Build the numeric pipeline (impute -> feature engineer -> scale)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', Second_transformer(add_bedrooms_per_room=True)),
    ('std_scaler', StandardScaler()),
])

# Build the categorical pipeline (one-hot)
cat_pipeline = Pipeline(steps=[
    ("onehot", Third_transformer()),
])

# Combine them together  
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)


# Linear Regressor Second model
np.random.seed(42)

#
ridge = Ridge()
# Define hyperparameter grid
param_grid = {
    "alpha": [0.1, 1.0, 10.0, 100.0, 500.0]
}

# Grid Search with 5-fold CV, using RMSE
grid_search = GridSearchCV(
    ridge,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    verbose=2
)



# Fit on training data
grid_search.fit(X_train_prepared, y_train)

# Best hyperparameters and RMSE
print("Best alpha:", grid_search.best_params_)
print("Best RMSE:", np.sqrt(-grid_search.best_score_))

# Predict on test set
best_ridge = grid_search.best_estimator_
y_pred = best_ridge.predict(X_test_prepared)

print("Best RMSE:", np.sqrt(-y_pred.best_score_))


#Best alpha: {'alpha': 1.0} 

#Best RMSE: 67865.56099114478  Grid search for train set
#Test RMSE: 71076.49983670865 Grid search for test set

best_ridge = grid_search.best_estimator_
y_pred = best_ridge.predict(X_test_prepared)

# RMSE on test set
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", test_rmse)
 
rnd_search = RandomizedSearchCV(
    ridge,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42
)

rnd_search.fit(X_train_prepared, y_train)


cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# Best CV RMSE and alpha
print("Best alpha:", rnd_search.best_params_)
print("Best CV RMSE:", np.sqrt(-rnd_search.best_score_))


best_ridge = rnd_search.best_estimator_
y_pred = best_ridge.predict(X_test_prepared)

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", test_rmse)

#Best alpha: {'alpha': 1.0}
#Best CV RMSE: 67865.56099114478 Randomized search for train set
#Test RMSE: 72710.70743953051 Randomized search for test set