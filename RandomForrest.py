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
from sklearn.ensemble import RandomForestRegressor
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




param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
]

forest_reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    forest_reg,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True
)

grid_search.fit(X_train_prepared, y_train)

# Best hyperparameters & CV RMSE
print("Best parameters:", grid_search.best_params_)
print("Best CV RMSE:", np.sqrt(-grid_search.best_score_))

# Test RMSE
best_forest = grid_search.best_estimator_
y_pred = best_forest.predict(X_test_prepared)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", test_rmse)

#Best parameters: {'max_features': 8, 'n_estimators': 30}
#Best CV RMSE: 49672.50940389753 Grid search for train set
#Test RMSE: 49198.020631676336 Grid search for test set

param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8),
}

forest_reg = RandomForestRegressor(random_state=42)

rnd_search = RandomizedSearchCV(
    forest_reg,
    param_distributions=param_distribs,
    n_iter=10,  # number of random combinations
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42
)

rnd_search.fit(X_train_prepared, y_train)

# CV results
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print("CV RMSE:", np.sqrt(-mean_score), "Params:", params)

# Best CV RMSE & parameters
print("Best parameters:", rnd_search.best_params_)
print("Best CV RMSE:", np.sqrt(-rnd_search.best_score_))

# Predictions on test set
best_forest = rnd_search.best_estimator_
y_pred = best_forest.predict(X_test_prepared)

# Test RMSE
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", test_rmse)

#Best parameters: {'max_features': 7, 'n_estimators': 180} 
#Best CV RMSE: 49047.201092644886 Randdomized search for train set
#Test RMSE: 48622.46702083209 Randomized search for test set