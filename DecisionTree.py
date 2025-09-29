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

print(X_test_prepared.shape)
print(X_train_prepared.shape)
