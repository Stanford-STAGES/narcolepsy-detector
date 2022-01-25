import os
import pandas as pd
import pickle
from argparse import ArgumentParser
from datetime import datetime
from glob import glob

import numpy as np
from tqdm import tqdm

from utils import extract_features, calc_sorem, calc_nremfrag
from utils import get_logger
from utils import match_data
from utils import rolling_window
from utils.feature_extraction import multi_step_transition_matrix


logger = get_logger()


def merge_dfs(data_dir):
    df_list = glob(os.path.join(data_dir, "*_r*.csv"))
    print(f"{datetime.now()} | Merging {len(df_list)} files")
    dfs = []
    for df in df_list:
        dfs.append(pd.read_csv(df, index_col=0))
    dfs = pd.concat(dfs).reset_index(drop=True)
    saveFile = os.path.join(data_dir, f"{os.path.basename(data_dir)}_unscaled.csv")
    print("{} | Saving {}".format(datetime.now(), saveFile))
    dfs.to_csv(saveFile)


def prepare_data(data_dir, resolutions, output_dir, subset, saveFile=None):

    if not isinstance(resolutions, list):
        resolutions = [resolutions]

    for p in [data_dir, output_dir]:
        if not os.path.exists(p):
            # print("{} | Creating directory: {}".format(datetime.now(), p))
            logger.info(f"Creating directory: {p}")
            os.makedirs(p, exist_ok=True)

    foldersPath = os.listdir(data_dir)
    model_str = os.path.basename(data_dir)
    if "archive" in foldersPath:
        foldersPath.remove("archive")
    foldersPath.sort()

    # if True:
    dfs = []
    for resolution in resolutions:
        if resolution % 1 == 0:
            resolution = int(resolution)
        num_steps = int(np.ceil(np.log2(7200 / resolution)))
        # print(f"{datetime.now()} | Running feature extraction for {resolution} s resolution")
        logger.info(f"Running feature extraction for {resolution} s resolution")
        logger.info(f"Using {num_steps} for transition matrix calculations")
        # for f in foldersPath:
        # count += 1
        filesPath = glob(os.path.join(data_dir, "**", "pred*.pkl"), recursive=True)
        DF = match_data(filesPath)
        # if subset == "test":
        #     DF = DF[DF["Narcolepsy test data"] == 1]
        # else:
        #     DF = DF[DF["Narcolepsy training data"] == 1]
        N = len(DF)
        # for thr in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100, 150, 200, 250, 300, 500]:
        for thr in [1]:
            # print(f"{datetime.now()} | Current threshold: {thr}")
            logger.info(f"Current threshold: {thr}")
            # features = np.zeros([24 + 31 * 15, len(DF)])
            features = [[]] * N
            labels = np.zeros(N)
            ID = [[]] * N
            cohort = [[]] * N
            threshold = [thr] * N
            res = [resolution] * N
            # for i in tqdm(range(N)):
            for i in tqdm(range(N)):

                with open(DF.iloc[i].Filepath, "rb") as pkl:
                    contents = pickle.load(pkl)
                labels[i] = DF["label"].values[i]
                ID[i] = DF.iloc[i]["ID"]
                cohort[i] = DF.iloc[i]["Cohort"]
                try:
                    resolution_key = f"yhat_{resolution}s"
                except:
                    if resolution == 1:
                        resolution_key = "logits"
                    elif resolution == 30:
                        resolution_key = "predicted"
                    else:
                        resolution_key = f"yhat_{resolution}s"
                try:
                    hypnodensity = contents[resolution_key]
                except KeyError:
                    hypnodensity = contents["logits"]
                    hypnodensity = rolling_window(hypnodensity, resolution, resolution).mean(axis=1)
                hypnodensity_30s = contents["predicted"]
                if len(hypnodensity) == 0:
                    continue
                feature_vec = []
                feature_vec.extend(extract_features(hypnodensity, resolution, hypnodensity_30s))
                # feature_vec.extend(multi_step_transition_matrix(hypnodensity, resolution=resolution))
                features[i] = feature_vec
                # features[i] = extract_features(hypnodensity, resolution, hypnodensity_30s)
                # features[i] = multi_step_transition_matrix(hypnodensity, resolution=resolution)
                # _, *features[i] = calc_sorem(hypnodensity, resolution, [thr])[0]
                # _, *features[i] = calc_nremfrag(hypnodensity, resolution, [thr])[0]
                # threshold[i] = thr
            features = np.asarray(features)
            _df = pd.DataFrame({"ID": ID, "Cohort": cohort, "Label": labels, "Threshold": threshold, "Resolution": res}).join(
                pd.DataFrame(features)
            )
            dfs.append(_df)
            # features = np.asarray(features)

    # Save raw features
    # saveFile = os.path.join(output_dir, f"{f}_r{resolution:02}_trainD_unscaled.csv")
    # saveFile = os.path.join(output_dir, f"{model_str}_r{resolution:02}_unscaled.csv")
    if saveFile is None:
        saveFile = os.path.join(output_dir, "_".join(filter(None, (f"r{resolution:02}", subset, "unscaled.csv"))))
    else:
        saveFile = os.path.join(output_dir, saveFile)
    # print("{} | Saving {}".format(datetime.now(), saveFile))
    logger.info(f"Saving {saveFile}")
    data_df = pd.concat(dfs)
    # data_df = pd.DataFrame({"ID": ID, "Cohort": cohort, "Label": labels}).join(pd.DataFrame(features))
    data_df.to_csv(saveFile)
    # dfs.append(data_df)
    print("")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="./data/massc")
    parser.add_argument("-o", "--output_dir", type=str, default="./data/narco_features")
    parser.add_argument("-r", "--resolution", type=float, nargs="+", default=15)
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("-s", "--subset", type=str)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--save_file", default=None, type=str)
    args = parser.parse_args()

    # if args.test:
    #     subset = "test"
    # else:
    #     subset = "trainD"

    if args.merge:
        merge_dfs(args.output_dir)
    else:
        prepare_data(args.data_dir, args.resolution, args.output_dir, args.subset, args.save_file)

    # sleep_predictions_path = args.data_dir
    # narco_scaling_path = "./data/narco_scaling"
    # narco_feature_path = "./data/narco_features"

    # for p in [sleep_predictions_path, narco_scaling_path, narco_feature_path]:
    #     if not os.path.exists(p):
    #         print("{} | Creating directory: {}".format(datetime.now(), p))
    #         os.makedirs(p)

    # foldersPath = os.listdir(sleep_predictions_path)
    # model_str = os.path.basename(sleep_predictions_path)
    # if "archive" in foldersPath:
    #     foldersPath.remove("archive")
    # foldersPath.sort()

    # count = -1
    # if True:
    #     # for f in foldersPath:
    #     # count += 1
    #     filesPath = glob(os.path.join(sleep_predictions_path, "**", "pred*.pkl"), recursive=True)
    #     DF = match_data(filesPath)
    #     # DF = DF[DF["Narcolepsy training data"] == 1]
    #     N = len(DF)

    #     # features = np.zeros([24 + 31 * 15, len(DF)])
    #     features = [[]] * N
    #     labels = np.zeros(N)
    #     ID = [[]] * N
    #     cohort = [[]] * N
    #     for i in tqdm(range(N)):

    #         with open(DF.iloc[i].Filepath, "rb") as pkl:
    #             contents = pickle.load(pkl)
    #         labels[i] = DF["label"].values[i]
    #         ID[i] = DF.iloc[i]["ID"]
    #         cohort[i] = DF.iloc[i]["Cohort"]
    #         if args.resolution == 1:
    #             resolution_key = "logits"
    #         elif args.resolution == 30:
    #             resolution_key = "predicted"
    #         else:
    #             resolution_key = f"yhat_{args.resolution}s"
    #         try:
    #             hypnodensity = contents[resolution_key]
    #         except KeyError:
    #             hypnodensity = contents["logits"]
    #             hypnodensity = rolling_window(hypnodensity, args.resolution, args.resolution).mean(axis=1)
    #         hypnodensity_30s = contents["predicted"]
    #         if len(hypnodensity) == 0:
    #             continue

    #         features[i] = extract_features(hypnodensity, args.resolution, hypnodensity_30s)
    #     features = np.asarray(features)

    #     # Save raw features
    #     # saveFile = os.path.join(narco_feature_path, f"{f}_r{args.resolution:02}_trainD_unscaled.csv")
    #     saveFile = os.path.join(narco_feature_path, f"{model_str}_r{args.resolution:02}_trainD_unscaled.csv")
    #     print("{} | Saving {}".format(datetime.now(), saveFile))
    #     data_df = pd.DataFrame({"ID": ID, "Cohort": cohort, "Label": labels}).join(pd.DataFrame(features))
    #     data_df.to_csv(saveFile)

    #     # Scale data and save
    #     pct15_85 = np.expand_dims(np.nanpercentile(features, 85, axis=1) - np.nanpercentile(features, 15, axis=1), axis=1) + 1e-5
    #     pct50 = np.expand_dims(np.nanpercentile(features, 50, axis=1), axis=1)
    #     features = (features - pct50) / pct15_85
    #     #         features[features > 10] = 10
    #     #         features[features < -10] = -10

    #     # saveFile = os.path.join(narco_feature_path, f"{f}_r{args.resolution:02}_trainD.csv")
    #     saveFile = os.path.join(narco_feature_path, f"{model_str}_r{args.resolution:02}_trainD.csv")
    #     print("{} | Saving {}".format(datetime.now(), saveFile))
    #     scaled_data_df = pd.DataFrame({"ID": ID, "Cohort": cohort, "Label": labels}).join(pd.DataFrame(features))
    #     scaled_data_df.to_csv(saveFile)

    #     scaleFile = os.path.join(narco_scaling_path, f"{model_str}_r{args.resolution:02}_scale.csv")
    #     print("{} | Saving {}".format(datetime.now(), scaleFile))
    #     scale_df = pd.DataFrame({"meanV": pct50.squeeze(), "scaleV": pct15_85.squeeze()})
    #     scale_df.to_csv(scaleFile)
