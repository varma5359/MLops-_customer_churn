import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(X: pd.DataFrame):
    X=X.drop(columns=["customerID","Churn"],errors="ignore")
    numeric=X.select_dtypes(include=["number"]).columns.tolist()
    categorical=[c for c in X.columns if c not in numeric]
    num=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
    cat=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric",num,numeric),("categorical",cat,categorical)],remainder="drop")
