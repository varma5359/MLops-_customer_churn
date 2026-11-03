import pandas as pd
from src.data.validation import EXPECTED_COLUMNS,validate_dataframe
def frame():
    d={c:["x","y"] for c in EXPECTED_COLUMNS}; d.update({"customerID":["1","2"],"SeniorCitizen":[0,1],"tenure":[2,20],"MonthlyCharges":[30.,80.],"TotalCharges":["30","1600"],"Churn":["No","Yes"]}); return pd.DataFrame(d)
def test_valid_frame(): assert validate_dataframe(frame()).passed
def test_duplicate_ids_fail():
    d=frame(); d.loc[1,"customerID"]="1"; assert not validate_dataframe(d).passed
def test_blank_total_charges_are_reported():
    d=frame(); d.loc[0,"TotalCharges"]=" "; r=validate_dataframe(d); assert r.passed and r.blank_values["TotalCharges"]==1
def test_bad_target_fails():
    d=frame(); d.loc[0,"Churn"]="Maybe"; assert not validate_dataframe(d).passed
