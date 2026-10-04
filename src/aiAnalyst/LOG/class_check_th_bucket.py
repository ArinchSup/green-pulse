import pandas as pd, numpy as np, joblib
import pillar2_v43_features as F
df = pd.read_csv("dataset_pillar2_mid_v43_volk05r06_8-40_cut20230630.csv.gz")
b = joblib.load("class_model/xgboost_mid_v43.joblib")
p = b["model"].predict_proba(df[b["feature_names"]])[:,1]   # in-sample, for date spread only
hi = df[p >= 0.70]
print(f"{len(hi)} rows, {hi.signal_date.nunique()} distinct dates, "
      f"{hi.signal_date.str[:4].nunique()} distinct years")
print(hi.signal_date.str[:4].value_counts().sort_index())