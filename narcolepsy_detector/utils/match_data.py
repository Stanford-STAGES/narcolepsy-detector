import os
from pathlib import Path

import pandas as pd


def match_data(files, data_master=None):
    files = sorted(files)
    DF = pd.DataFrame(files, columns=["Filepath"])
    DF["ID"] = DF["Filepath"].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0].split("preds_")[1]
    )

    # HACK
    # DF["ID"] = DF["ID"].apply(lambda x: str(x[:5]) if "notte" in x else x)  # This removes unneccessary chars in IHC data
    # DF["ID"] = DF["ID"].apply(lambda x: str(x[:8]) if "_p" in x else x)  # This removes unnecessary chars in KHC data

    # if data_master is None:
    #     data_master = "./data_master.csv"
    if data_master is not None:
        narcoDat = pd.read_csv(data_master, delimiter=",")
        DFmerge = pd.merge(
            DF, narcoDat, how="inner", left_on="ID", right_on="OakFileName"
        )
        DFmerge["label"] = 0
        DFmerge.loc[DFmerge["Dx"] == "NT1", "label"] = 1
    else:
        DF["Cohort"] = [Path(p).parent.stem.upper() for p in files]
        DF["label"] = 0
        DF["OakFileName"] = DF["ID"].copy()
        DFmerge = DF
    return DFmerge


# def match_data(files, data_master=None):

#     files = sorted(files)
#     DF = pd.DataFrame(files, columns=["Filepath"])
#     DF["ID"] = DF["Filepath"].apply(lambda x: os.path.splitext(os.path.basename(x))[0].split("preds_")[1])

#     # HACK
#     DF["ID"] = DF["ID"].apply(lambda x: str(x[:5]) if "notte" in x else x)  # This removes unneccessary chars in IHC data
#     DF["ID"] = DF["ID"].apply(lambda x: str(x[:8]) if "_p" in x else x)  # This removes unnecessary chars in KHC data

#     if data_master is None:
#         data_master = "./data_master.csv"
#     narcoDat = pd.read_csv(data_master, delimiter=",")

#     DFmerge = pd.merge(DF, narcoDat, how="inner", left_on="ID", right_on="ID")
#     DFmerge["label"] = -1
#     DFmerge.loc[DFmerge["Diagnosis"] == "'T1 NARCOLEPSY'", "label"] = 1
#     DFmerge.loc[DFmerge["Diagnosis"] == "'NON-NARCOLEPSY CONTROL'", "label"] = 0
#     DFmerge.loc[DFmerge["Diagnosis"] == "'OTHER HYPERSOMNIA'", "label"] = 0
#     DFmerge = DFmerge[DFmerge["label"] != -1]
#     DFmerge = DFmerge.sort_values("ID")
#     return DFmerge
