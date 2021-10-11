import argparse
from json import load
import logging
import os
import pickle
import random
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

# from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.feature_selection import RFECV

# from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.svm import SVC

from utils import extract_features

logger = logging.getLogger("FEATURE SELECTION")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s  | %(name)-8s | [%(levelname)-5.5s]  |  %(message)s", datefmt="%H:%M:%S")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


def load_data(data_dir, sleep_model, resolution):
    df_master = pd.read_csv("data_master.csv", index_col=0)
    try:
        df = pd.read_csv(
            os.path.join(data_dir, f"{sleep_model}_long_r{resolution:02}_trainD_unscaled.csv"), index_col=0
        )
    except FileNotFoundError:
        df = pd.read_csv(os.path.join(data_dir, f"{sleep_model}_long_r{resolution:02}_unscaled.csv"), index_col=0)
        train_idx = df.merge(df_master, left_on=["ID", "Cohort"], right_on=["ID", "Cohort"])[
            "Narcolepsy training data"
        ]
        df = df.loc[train_idx == 1]
    X = df.iloc[:, 3:].values
    y = df["Label"].values[:].astype(int)
    return X, y


def run_selection(sleep_model_path, resolution, n_kfold, n_repeats, seed, *args, **kwargs):

    sleep_model = "_".join(os.path.basename(sleep_model_path).split("_")[:2])
    feature_selection_base_path = os.path.join("data", "feature_selection", f"{sleep_model}_r{resolution:04}")

    # sherlock_base_path = "./data"
    # features_base_path = os.path.join(sherlock_base_path, "narco_features")
    # feature_selection_base_path = os.path.join(sherlock_base_path, "feature_selection")
    # features_path = os.path.join(features_base_path, sleep_model + "_trainD.p")
    # scaling_path = os.path.join(sherlock_base_path, "narco_scaling")
    logger.info(f"Creating directory: {feature_selection_base_path}")
    os.makedirs(feature_selection_base_path, exist_ok=True)

    # sherlock_base_path = '/scratch/users/alexno/narco_ml'
    # features_base_path = os.path.join(sherlock_base_path, 'narco_features', 'pickles')
    # feature_selection_base_path = os.path.join(sherlock_base_path, 'narco_feature_selection')
    # features_path = os.path.join(features_base_path, sleep_model + '_trainD.p')
    # scaling_path = os.path.join(sherlock_base_path, 'narco_scaling')
    # config = type("obj", (object,), {"narco_feature_selection_path": None, "narco_scaling_path": scaling_path})
    # extract_features = ExtractFeatures(config)
    # print("{} | Processing {}".format(datetime.now(), features_path))
    logger.info(f"Processing {sleep_model}")
    logger.info(f"Reading data at {resolution} s resolution")
    features, labels = load_data(sleep_model_path, sleep_model, resolution)

    scaler = sklearn.preprocessing.RobustScaler(quantile_range=(15.0, 85.0))
    features = scaler.fit_transform(features)
    logger.info(f"Scaling features using {scaler}")
    # with open(features_path, "rb") as f:
    #     contents = pickle.load(f)

    # labels = np.asarray(contents["labels"])
    # trainOrTest = np.random.rand(len(labels))
    # trainInd = trainOrTest>0.4
    # labels = labels[trainInd]
    # features = contents["features"].T
    # features = extract_features.scale_features(contents['features'], sleep_model).T

    # DEBUG
    # features = features[:60]
    # labels = labels[:60]
    logger.info(f"X dimensions: {features.shape}")
    # print("{} | Shape of X: {}".format(datetime.now(), features.shape))

    # features = features[trainInd,:]
    features2 = np.square(features)

    X = np.concatenate([features, features2], axis=1)
    y = labels

    # skf = RepeatedStratifiedKFold(n_splits=n_kfold, n_repeats=n_repeats)
    skf = StratifiedKFold(n_splits=n_kfold)
    logger.info("CV strategy:")
    logger.info(f"\tType: {skf}")
    logger.info(f"\tNumber of folds: {n_kfold}")
    logger.info(f"\tNumber of repeats: {n_repeats}")

    estimator = SVC(kernel="linear")
    # estimator = sklearn.linear_model.LogisticRegression(penalty="l1", solver="liblinear")
    rfecv = RFECV(estimator=estimator, step=1, cv=skf, scoring="accuracy", n_jobs=-1, verbose=1)
    logger.info("RFE strategy:")
    logger.info(f"\tEstimator: {estimator}")

    logger.info("Starting RFE procedure")
    start = time.time()
    rfecv.fit(X, y)
    end = time.time()
    relevant = rfecv.get_support()
    score = max(rfecv.grid_scores_)
    logger.info(f"\tElapsed time: {end - start}")
    logger.info(f"\tOptimal number of features: {rfecv.n_features_}")
    logger.info(
        f"\tOptimal number of unique features: {sum(relevant[:len(relevant)//2] + relevant[len(relevant)//2:])}"
    )

    # Save selected features
    save_file_path = os.path.join(feature_selection_base_path, f"{sleep_model}-{seed}.pkl")
    logger.info(f"Saving objects at {save_file_path}")
    selected_features = {"features": relevant, "score": score, "model": rfecv}
    with open(save_file_path, "wb") as fp:
        pickle.dump(selected_features, fp)

    # Plot number of features VS. cross-validation scores
    figure_path = os.path.join(feature_selection_base_path, f"feature_selection-{seed}.png")
    logger.info(f"Plotting and saving figure(s) at {figure_path}")
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", nargs="+", required=True)
    parser.add_argument("-r", "--resolution", type=int)
    parser.add_argument("-k", "--n_kfold", default=20, type=int)
    parser.add_argument("-n", "--n_repeats", default=5, type=int)
    parser.add_argument("-s", "--seed", default=1337, type=int)
    args = parser.parse_args()

    logger.info(f'Usage: {" ".join([x for x in sys.argv])}\n')
    logger.info("Settings:")
    logger.info("---------")
    for idx, (k, v) in enumerate(sorted(vars(args).items())):
        if idx == len(vars(args)) - 1:
            logger.info(f"{k:>15}\t{v}\n")
        else:
            logger.info(f"{k:>15}\t{v}")

    random.seed(1337 + args.seed)
    np.random.seed(1337 + args.seed)

    for sleep_model_path in sorted(args.data_dir):
        run_selection(sleep_model_path=sleep_model_path, **vars(args))

    # sherlock_base_path = "./data"
    # features_base_path = os.path.join(sherlock_base_path, "narco_features")

    # for sleep_model in sorted(os.listdir(features_base_path)):
    #     run_selection(sleep_model[:-9])
