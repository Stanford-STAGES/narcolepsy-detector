import os
from pathlib import Path

import pandas as pd


def match_data(files, data_master=None):
    files = sorted(files)
    df = pd.DataFrame(files, columns=["Filepath"])
    df["ID"] = df["Filepath"].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0].split("preds_")[1]
    )

    # HACK
    # df["ID"] = df["ID"].apply(lambda x: str(x[:5]) if "notte" in x else x)  # This removes unneccessary chars in IHC data
    # df["ID"] = df["ID"].apply(lambda x: str(x[:8]) if "_p" in x else x)  # This removes unnecessary chars in KHC data

    # if data_master is None:
    #     data_master = "./data_master.csv"
    if data_master is not None:
        if isinstance(data_master, pd.DataFrame):
            df_master = data_master
        else:
            df_master = pd.read_csv(data_master, delimiter=",")
        df_merge = pd.merge(
            df, df_master, how="inner", left_on="ID", right_on="OakFileName"
        )
        df_merge["label"] = 0
        df_merge.loc[df_merge["Dx"] == "NT1", "label"] = 1
    else:
        df["Cohort"] = [Path(p).parent.stem.upper() for p in files]
        df["label"] = 0
        df["OakFileName"] = df["ID"].copy()
        df_merge = df
    return df_merge


# def match_data(files, data_master=None):

#     files = sorted(files)
#     df = pd.DataFrame(files, columns=["Filepath"])
#     df["ID"] = df["Filepath"].apply(lambda x: os.path.splitext(os.path.basename(x))[0].split("preds_")[1])

#     # HACK
#     df["ID"] = df["ID"].apply(lambda x: str(x[:5]) if "notte" in x else x)  # This removes unneccessary chars in IHC data
#     df["ID"] = df["ID"].apply(lambda x: str(x[:8]) if "_p" in x else x)  # This removes unnecessary chars in KHC data

#     if data_master is None:
#         data_master = "./data_master.csv"
#     df_master = pd.read_csv(data_master, delimiter=",")

#     df_merge = pd.merge(df, df_master, how="inner", left_on="ID", right_on="ID")
#     df_merge["label"] = -1
#     df_merge.loc[df_merge["Diagnosis"] == "'T1 NARCOLEPSY'", "label"] = 1
#     df_merge.loc[df_merge["Diagnosis"] == "'NON-NARCOLEPSY CONTROL'", "label"] = 0
#     df_merge.loc[df_merge["Diagnosis"] == "'OTHER HYPERSOMNIA'", "label"] = 0
#     df_merge = df_merge[df_merge["label"] != -1]
#     df_merge = df_merge.sort_values("ID")
#     return df_merge
