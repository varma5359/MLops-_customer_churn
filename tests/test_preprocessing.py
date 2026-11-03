import pandas as pd
from src.features.preprocessing import build_preprocessor
def test_identifier_and_target_are_excluded():
    x=pd.DataFrame({"customerID":["a","b"],"Churn":["No","Yes"],"tenure":[1,2],"Contract":["Monthly","Annual"]})
    p=build_preprocessor(x)
    assert "customerID" not in p.transformers[0][2] and "Churn" not in p.transformers[0][2]
