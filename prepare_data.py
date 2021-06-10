import os
import pandas as pd
import pickle
from argparse import ArgumentParser
from datetime import datetime
from glob import glob

import numpy as np
from tqdm import tqdm

from utils import extract_features
from utils import match_data
from utils import rolling_window


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="./data/massc")
    parser.add_argument("-r", "--resolution", type=int, default=15)
    args = parser.parse_args()

    sleep_predictions_path = args.data_dir
    narco_scaling_path = "./data/narco_scaling"
    narco_feature_path = "./data/narco_features"

    for p in [sleep_predictions_path, narco_scaling_path, narco_feature_path]:
        if not os.path.exists(p):
            print("{} | Creating directory: {}".format(datetime.now(), p))
            os.makedirs(p)

    foldersPath = os.listdir(sleep_predictions_path)
    model_str = os.path.basename(sleep_predictions_path)
    if "archive" in foldersPath:
        foldersPath.remove("archive")
    foldersPath.sort()

    count = -1
    if True:
        # for f in foldersPath:
        # count += 1
        filesPath = glob(os.path.join(sleep_predictions_path, "**", "pred*.pkl"), recursive=True)
        DF = match_data(filesPath)
        DF = DF[DF["Narcolepsy training data"] == 1]
        N = len(DF)

        # features = np.zeros([24 + 31 * 15, len(DF)])
        features = [[]] * N
        labels = np.zeros(N)
        ID = [[]] * N
        cohort = [[]] * N
        for i in tqdm(range(N)):

            with open(DF.iloc[i].Filepath, "rb") as pkl:
                contents = pickle.load(pkl)
            labels[i] = DF["label"].values[i]
            ID[i] = DF.iloc[i]["ID"]
            cohort[i] = DF.iloc[i]["Cohort"]
            if args.resolution == 1:
                resolution_key = "logits"
            elif args.resolution == 30:
                resolution_key = "predicted"
            else:
                resolution_key = f"yhat_{args.resolution}s"
            try:
                hypnodensity = contents[resolution_key]
            except NameError:
                hypnodensity = contents["logits"]
                hypnodensity = rolling_window(hypnodensity, args.resolution, args.resolution).mean(axis=1)
            hypnodensity_30s = contents["predicted"]
            if len(hypnodensity) == 0:
                continue

            features[i] = extract_features(hypnodensity, args.resolution, hypnodensity_30s)
        features = np.asarray(features)

        # Save raw features
        # saveFile = os.path.join(narco_feature_path, f"{f}_r{args.resolution:02}_trainD_unscaled.csv")
        saveFile = os.path.join(narco_feature_path, f"{model_str}_r{args.resolution:02}_trainD_unscaled.csv")
        print("{} | Saving {}".format(datetime.now(), saveFile))
        data_df = pd.DataFrame({"ID": ID, "Cohort": cohort, "Label": labels}).join(pd.DataFrame(features))
        data_df.to_csv(saveFile)

        # Scale data and save
        pct15_85 = (
            np.expand_dims(np.nanpercentile(features, 85, axis=1) - np.nanpercentile(features, 15, axis=1), axis=1)
            + 1e-5
        )
        pct50 = np.expand_dims(np.nanpercentile(features, 50, axis=1), axis=1)
        features = (features - pct50) / pct15_85
        #         features[features > 10] = 10
        #         features[features < -10] = -10

        # saveFile = os.path.join(narco_feature_path, f"{f}_r{args.resolution:02}_trainD.csv")
        saveFile = os.path.join(narco_feature_path, f"{model_str}_r{args.resolution:02}_trainD.csv")
        print("{} | Saving {}".format(datetime.now(), saveFile))
        scaled_data_df = pd.DataFrame({"ID": ID, "Cohort": cohort, "Label": labels}).join(pd.DataFrame(features))
        scaled_data_df.to_csv(saveFile)

        scaleFile = os.path.join(narco_scaling_path, f"{model_str}_r{args.resolution:02}_scale.csv")
        print("{} | Saving {}".format(datetime.now(), scaleFile))
        scale_df = pd.DataFrame({"meanV": pct50.squeeze(), "scaleV": pct15_85.squeeze()})
        scale_df.to_csv(scaleFile)
