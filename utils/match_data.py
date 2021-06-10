import os

import pandas as pd


def match_data(files, labelPath=None):

    files = sorted(files)
    DF = pd.DataFrame(files, columns=["Filepath"])
    DF["ID"] = DF["Filepath"].apply(lambda x: os.path.splitext(os.path.basename(x))[0].split("preds_")[1])

    # HACK
    DF["ID"] = DF["ID"].apply(lambda x: str(x[:5]) if "notte" in x else x)
    DF["ID"] = DF["ID"].apply(lambda x: str(x[:8]) if "_p" in x else x)

    if labelPath is None:
        labelPath = "./data_master.csv"
    narcoDat = pd.read_csv(labelPath, delimiter=",")

    DFmerge = pd.merge(DF, narcoDat, how="inner", left_on="ID", right_on="ID")
    DFmerge["label"] = -1
    DFmerge.loc[DFmerge["Diagnosis"] == "'T1 NARCOLEPSY'", "label"] = 1
    DFmerge.loc[DFmerge["Diagnosis"] == "'NON-NARCOLEPSY CONTROL'", "label"] = 0
    DFmerge.loc[DFmerge["Diagnosis"] == "'OTHER HYPERSOMNIA'", "label"] = 0
    DFmerge = DFmerge[DFmerge["label"] != -1]
    DFmerge = DFmerge.sort_values("ID")
    return DFmerge
