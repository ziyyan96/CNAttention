import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder




def load_tables(cfg):
    X = pd.read_csv(cfg["paths"]["X"], index_col=0)
    meta = pd.read_csv(cfg["paths"]["all_cna"], index_col=0)
    id_col = cfg["labels"]["id_col"]
    name_col = cfg["labels"]["name_col"]
    label_col = cfg["labels"].get("label_col", "cancer_label")


    # ensure label column exists
    if label_col not in meta.columns:
        le = LabelEncoder()
        meta[label_col] = le.fit_transform(meta[name_col])


    # align by sample id
    common = X.index.intersection(meta.index)
    X = X.loc[common]
    meta = meta.loc[common]


    y = meta[label_col].values
    genes = list(X.columns)
    subtypes = meta[name_col].unique()
    return X, y, genes, subtypes, meta