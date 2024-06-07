from copy import copy
import itertools
import json
import logging
import os
import pickle
import random
from typing import List, Optional
import warnings

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# import absl.logging
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm

from train_gp_model import THRESHOLD
from utils import plot_roc_ensemble

# from narcolepsy_detector.data import get_datamaster, collect_data
from narcolepsy_detector.data.data_class import SingleResolutionDataset
from narcolepsy_detector.data.dataset_collection import DatasetCollection
from narcolepsy_detector.model.narcolepsy_model import NarcolepsyModel
from narcolepsy_detector.utils import get_logger


# logging.getLogger("absl").setLevel(logging.INFO)
logger = get_logger()

SEED = 1337
os.environ["PL_GLOBAL_SEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def train_ensemble(
    experiment_dir: Path,
    resolutions: Optional[List[int]],
    data_master: Optional[Path],
    n_repeats: Optional[int] = None,
    n_kfold: Optional[int] = None,
):

    # Select models based on resolution
    experiments = []
    for model_path in experiment_dir.iterdir():
        with open(model_path / "settings.json", "r") as fp:
            settings = json.load(fp)
            if resolutions:
                if settings["resolution"] in resolutions:
                    experiments.append(model_path)
            else:
                experiments.append(model_path)

    logger.info("Collecting feature data ...")
    data_list = dict()
    for i, exp in enumerate(experiments):

        # print("")
        logger.info(f"\tCollecting data from: {exp}")

        # Get settings
        with open(os.path.join(exp, "settings.json"), "r") as fp:
            settings = json.load(fp)
        resolution = settings["resolution"]
        cv_model = settings["model"]

        # Init dataclass
        dataset = SingleResolutionDataset(**settings)

        # Append
        data_list[resolution] = dataset

    dataset = DatasetCollection(data_list)
    dataset.split_data()

    if not n_repeats:
        n_repeats = settings["n_repeats"]
    if not n_kfold:
        n_kfold = settings["n_kfold"]

    models = []
    datasets = []
    cv_preds = [[]] * n_repeats
    repeat_preds = dict(train=[], eval=[])
    auc_values = dict(train=[], eval=[])
    accuracy = dict(train=[], eval=[])

    cv_acc = np.empty((n_repeats, n_kfold))
    n_resolutions = dataset.n_resolutions
    logger.info(f"Training ensemble model using {n_resolutions} resolutions")
    skf = RepeatedStratifiedKFold(n_splits=n_kfold, n_repeats=n_repeats, random_state=1337)

    for (current_repeat, current_fold), (cv_train_idx, cv_test_idx) in zip(
        itertools.product(range(n_repeats), range(n_kfold)), skf.split(range(dataset.train_data.N), dataset.train_data.y)
    ):
        # current_fold = 0
        logger.info(f"Current repeat: {current_repeat + 1} / {n_repeats}")
        logger.info(f"Current fold: {current_fold + 1} / {n_kfold}")

        # ds = dataset.copy()

        logger.info("Selecting specific features for each dataset...")
        logger.info(f"\tInitial number of features: {dataset.train_data.n_features}")
        dataset.select_features(current_repeat=current_repeat)
        logger.info(f"\tReduced number of features: {dataset.n_features}")

        logger.info("Scaling features...")
        cv_train_data = dataset.get_data(subject_idx=cv_train_idx)
        cv_train_data = dataset.scale_features(cv_train_data)
        logger.info("\tScaling CV test features")
        cv_test_data = dataset.get_data(subject_idx=cv_test_idx)
        cv_test_data = dataset.scale_features(cv_test_data)

        logger.info("Setting up model...")
        model_params = settings.copy()
        [
            model_params.pop(k)
            for k in (
                "data_dir",
                "data_master",
                "feature_selection",
                "feature_set",
                "model",
                "n_kfold",
                "n_repeats",
                "resolution",
                "save_dir",
            )
        ]
        # model_params["n_iter"] = 10
        cv_model = NarcolepsyModel(**model_params)

        logger.info("\tFitting model to CV training data")
        cv_model.fit(cv_train_data.X, cv_train_data.y)

        logger.info("Getting predictions on CV test data")
        cv_mean_pred, cv_var_pred = cv_model.predict(cv_test_data.X)

        # Get the prediction accuracy at 50% cutoff
        # y_pred = np.ones(cv_mean_pred.shape)
        # y_pred[cv_mean_pred < 0.5] = -1
        # acc = np.mean(np.squeeze(y_pred) == np.squeeze(cv_test_data.y.values.squeeze()))
        # # cv_acc.append(acc)
        acc, auc, cv_y_pred = cv_model.evaluate_performance(cv_mean_pred, cv_test_data.y.values)
        logger.info(f"\tFold accuracy: {acc}")

        # Record CV performances
        cv_acc[current_repeat, current_fold] = acc
        cv_preds_df = cv_test_data.df.copy()
        cv_preds_df[f"mean_n{current_repeat + 1:02}_k{current_fold + 1:02}"] = cv_mean_pred
        cv_preds_df[f"var_n{current_repeat + 1:02}_k{current_fold + 1:02}"] = cv_var_pred
        cv_preds_df[f"y_n{current_repeat + 1:02}_k{current_fold + 1:02}"] = cv_y_pred
        cv_preds[current_repeat].append(cv_preds_df)

        dataset.clean()

        if current_fold + 1 == n_kfold:

            logger.info("Fitting model to training data")
            dataset.select_features(current_repeat=current_repeat)
            train_data = dataset.get_data(subject_idx=slice(dataset.train_data.N))
            train_data = dataset.scale_features(train_data)
            eval_data = dataset.get_data(subject_idx=slice(dataset.eval_data.N), group="eval")
            eval_data = dataset.scale_features(eval_data)
            model = NarcolepsyModel(**model_params)
            model.fit(train_data.X, train_data.y)

            # Getting train and test predictions
            mean_pred_train, var_pred_train = model.predict(train_data.X)
            mean_pred_eval, var_pred_eval = model.predict(eval_data.X)

            # Get the prediction accuracy at 50% cutoff
            acc_train, auc_train, y_pred_train = model.evaluate_performance(mean_pred_train, train_data.y.values)
            acc_eval, auc_eval, y_pred_eval = model.evaluate_performance(mean_pred_eval, eval_data.y.values)
            accuracy["train"].append(acc_train)
            accuracy["eval"].append(acc_eval)
            auc_values["train"].append(auc_train)
            auc_values["eval"].append(auc_eval)
            logger.info(f"Train/test AUC: {auc_train:.3f} / {auc_eval:.3f}")
            logger.info(f"Train/test ACC: {acc_train:.3f} / {acc_eval:.3f}")

            # Saving model
            model_dir = experiment_dir.parent / "ensemble" / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(model_dir / f"n{current_repeat + 1:02}")

            # Save predictions
            def append_to_df(df, vars_to_append):
                for var_name, var in vars_to_append:
                    df[var_name] = var
                return df

            df_train = append_to_df(
                train_data.df.copy(),
                [
                    (f"mean_n{current_repeat + 1:02}", mean_pred_train.squeeze()),
                    (f"var_n{current_repeat + 1:02}", var_pred_train.squeeze()),
                    (f"y_pred_n{current_repeat + 1:02}", y_pred_train),
                ],
            )
            df_eval = append_to_df(
                eval_data.df.copy(),
                [
                    (f"mean_n{current_repeat + 1:02}", mean_pred_eval.squeeze()),
                    (f"var_n{current_repeat + 1:02}", var_pred_eval.squeeze()),
                    (f"y_pred_n{current_repeat + 1:02}", y_pred_eval),
                ],
            )
            predictions_dir = model_dir.with_name("predictions")
            predictions_dir.mkdir(parents=True, exist_ok=True)
            pd.concat([df_train, df_eval]).to_csv(predictions_dir / f"predictions_n{current_repeat + 1:02}.csv")

            # # Save dataset
            # with open(model_dir / f"n{current_repeat + 1:02}" / "dataset.pkl", "wb") as pkl:
            #     pickle.dump(dataset, pkl)

            dataset.clean()

    # # Select specific features
    # logger.info("Selecting features...")
    # if feature_selection:
    #     X_train = ds.select_features(df=X_train, features=[])

    # # Run over experiments
    # logger.info(f"Training ensemble model using {n_exp} resolutions")

    # logger.info(f"Running {n_repeats} repeats ...")
    # for n in n_repeats:

    #     logging.info(f"Run {n} of {n_repeats}")
    # logging.info(f"Collecting data features...")

    # if data_master is None:
    #     df["Dx"] = df["Label"].map({0: "control", 1: "NT1"})
    # y = df.Dx.map({"control": -1, "NT1": 1}).values
    # y = E['labels']

    # # Get feature scalers
    # with open(os.path.join(exp, "feature_scales.pkl"), "rb") as fp:
    #     scalers = pickle.load(fp)

    # # Get PCA objects
    # if do_pca:
    #     with open(os.path.join(exp, "pca_objects.pkl"), "rb") as fp:
    #         pcas = pickle.load(fp)

    # # Get list of model repeats and folds
    # models = sorted(Path(exp).rglob("*.gpm"))

    # # Get model scale weights
    # model_weights = pd.read_csv(os.path.join(exp, "model_scales.csv"), index_col=0).cv_auc.values[:, np.newaxis]

    # # Load CV data to obtain CV test AUC
    # with open(os.path.join(exp, "cv_data.pkl"), "rb") as fp:
    #     cv_data = pickle.load(fp)
    # cv_test_p = (cv_data["mean"].T @ model_weights).squeeze() / model_weights.sum()
    # cv_test_dir = os.path.join(os.path.dirname(savedir_output), "cv_preds")
    # os.makedirs(cv_test_dir, exist_ok=True)
    # # print(df.ID.values)
    # # print(cv_data["true_class"].squeeze().shape)
    # df_cv_preds = pd.DataFrame.from_dict(
    #     {
    #         "ID": df_train,
    #         "PseudoID": np.arange(len(cv_test_p)),
    #         "p": cv_test_p,
    #         "y": cv_data["true_class"][0].squeeze(),
    #         "Resolution": [resolution] * len(cv_test_p),
    #     }
    # )
    # df_cv_preds.to_csv(os.path.join(cv_test_dir, f"cv_test_p-r{resolution:04}.csv"))
    # df_cv_preds_total.append(df_cv_preds)
    # # p_ens_cv[i] = cv_test_p
    # # np.savetxt(os.path.join(cv_test_dir, f"cv_test_p-r{resolution:04}.out"), (df.ID.values, cv_test_p))

    # continue

    # cv_test_y = cv_data["true_class"][0]
    # cv_test_auc = roc_auc_score(cv_test_y, cv_test_p)


#     # Run over all models
#     mean_pred = np.zeros((n_repeats, n_kfold, N, 1))
#     var_pred = np.zeros((n_repeats, n_kfold, N, 1))
#     p_pred = np.zeros(N)
#     y_pred = np.ones(N)

#     for model in tqdm(models):
#         # Get model info and saved model
#         repeat_num = int(model.name.split(".")[0].split("_")[1][1:]) - 1
#         fold_num = int(model.name.split(".")[0].split("_")[2][1:]) - 1
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             loaded_model = tf.saved_model.load(str(model))

#         # Load and apply correct feature scaling
#         scaler = scalers[repeat_num][fold_num]
#         X_scaled = scaler.transform(X)

#         # Possibly do PCA
#         if do_pca:
#             pca_obj = pcas[repeat_num][fold_num]["obj"]
#             X_scaled = pca_obj.transform(X_scaled)

#         # If feature selection procedure was performed, select correct features
#         if feature_selection:
#             fs_path = os.path.join(exp, "feature-selection", f"selected-features_n{repeat_num+1:02}_k{fold_num+1:02}.pkl")
#             with open(fs_path, "rb") as pkl:
#                 selected_features = np.array(pickle.load(pkl)["features"]).astype(int)
#             selected_features = (
#                 selected_features[: len(selected_features) // 2] + selected_features[len(selected_features) // 2 :]
#             ) > 0
#             X_scaled = X_scaled[:, selected_features]
#             # print(f"X.shape: {X_scaled.shape}")

#         # Get predictions
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             mean_pred[repeat_num, fold_num], var_pred[repeat_num, fold_num] = loaded_model.predict_y_compiled(X_scaled)

#     # Average over folds and apply scaling to each repeat
#     mean_pred = mean_pred.squeeze(-1)
#     var_pred = var_pred.squeeze(-1)
#     p_pred = (model_weights.T @ mean_pred.mean(axis=1) / model_weights.sum()).squeeze(0)
#     y_pred[p_pred < THRESHOLD[likelihood]] = -1

#     # Get performance
#     # plot_roc(
#     #     y, p_pred, THRESHOLD[likelihood], "Test data", savepath=os.path.join(exp, "figures", "roc_test_long.png")
#     # )
#     try:
#         auc_test = roc_auc_score(y, p_pred)
#     except ValueError:
#         auc_test = np.nan
#     print("AUC test:", auc_test)
#     print(
#         f'Classification report, test:\n{classification_report(y > THRESHOLD[likelihood], p_pred > THRESHOLD[likelihood], target_names=["CTRL", "NT1"])}'
#     )
#     print(f"Confusion matrix, test:\n{confusion_matrix(y > THRESHOLD[likelihood], p_pred > THRESHOLD[likelihood])}")
#     # logger.info(f"AUC on CV test data: {auc_eval}")
#     try:
#         y_cv[i] = cv_test_y
#         y_ens[i] = y_pred
#         p_ens[i] = p_pred
#         p_ens_cv[i] = cv_test_p
#         model_aucs[i] = auc_test
#         auc_scales[i] = cv_test_auc
#         # auc_scales[i] = cv_test_auc[resolution]
#     except:
#         pass

#     if True:
#         # Save predictions to .csv
#         # master = (
#         #     pd.read_csv("data_master.csv")
#         #     # pd.read_excel("overview_file_cohorts.xlsx", engine="openpyxl", usecols="A:O")
#         #     .reset_index(drop=True)
#         #     .drop(["Index", "Unnamed: 0"], axis=1)
#         #     .rename_axis(index=None)
#         # )
#         df_preds = (
#             # df[["ID", "Cohort", "Label_x"]]
#             df[["ID", "Cohort", "Dx"]]
#             .copy()
#             .reset_index(drop=True)
#             .rename_axis(index=None, columns=None)
#             # .rename(columns={"Label_x": "Label"})
#         )
#         df_preds["p"] = p_pred
#         df_preds["y"] = y_pred
#         df_preds["var"] = var_pred.mean(axis=(0, 1))
#         df_preds["var_w"] = (model_weights.T @ var_pred.mean(axis=1) / model_weights.sum()).squeeze()
#         df_preds["std"] = df_preds["var"].pow(1.0 / 2)
#         df_preds["std_w"] = df_preds["var_w"].pow(1.0 / 2)
#         if data_master is not None:
#             try:
#                 df_preds = (
#                     df_preds.merge(df_master, left_on=["ID", "Cohort"], right_on=["ID", "Cohort"])
#                     .drop(
#                         [
#                             "Sleep scoring training data",
#                             "Sleep scoring test data",
#                             "Narcolepsy training data",
#                             "Narcolepsy test data",
#                             "Replication data",
#                             "High pre-test",
#                             "Label_y",
#                             "Added by Alex",
#                         ],
#                         axis=1,
#                     )
#                     .rename(columns={"Label_x": "Label"})
#                 )
#             except:
#                 try:
#                     df_preds = (
#                         df_preds.merge(df_master, left_on=["ID", "Cohort"], right_on=["ID", "Cohort"])
#                         .drop(
#                             [
#                                 "Sleep scoring train",
#                                 "Sleep scoring test",
#                                 "Narcolepsy train",
#                                 "Narcolepsy test",
#                                 "Replication",
#                                 "High pretest",
#                                 "Label_y",
#                             ],
#                             axis=1,
#                         )
#                         .rename(columns={"Label_x": "Label"})
#                     )
#                 except:
#                     df_preds = (
#                         df_preds.merge(df_master, left_on=["ID", "Cohort", "Dx"], right_on=["OakFileName", "Cohort", "Dx"])
#                         # .drop(
#                         #     [
#                         #         "Sleep scoring train",
#                         #         "Sleep scoring test",
#                         #         "Narcolepsy train",
#                         #         "Narcolepsy test",
#                         #         "Replication",
#                         #         "High pretest",
#                         #         "Label_y",
#                         #     ],
#                         #     axis=1,
#                         # )
#                         # .rename(columns={"Label_x": "Label"})
#                     )
#         df_preds["Resolution"] = f"{resolution} s"
#         df_ensemble.append(df_preds)
#         # df_preds.to_csv(os.path.join(exp, "test_predictions.csv"))


# df_cv_preds = pd.DataFrame.from_dict(
#     {
#         "ID": df_train,
#         "PseudoID": np.arange(len(cv_test_p)),
#         "p": np.asarray(p_ens_cv).mean(axis=0),
#         "y": cv_data["true_class"][0].squeeze(),
#         "Resolution": ["Ensemble"] * len(cv_test_p),
#     }
# )
# df_cv_preds_total.append(df_cv_preds)
# df_cv_preds_total = pd.concat(df_cv_preds_total).reset_index(drop=True)
# df_cv_preds_total.to_csv(os.path.join(cv_test_dir, "cv_test_p.csv"))

# if n_exp > 1:
#     print(" ")
#     print("Running ensemble predictions")
#     print("============================")

#     os.makedirs(savedir_output, exist_ok=True)

#     print(" ")
#     print("CV test predictions:")
#     plot_roc_ensemble(
#         [y_cv[0]] * (n_exp + 1),
#         p_ens_cv + [np.asarray(p_ens_cv).mean(axis=0)],
#         THRESHOLD[likelihood],
#         # [os.path.basename(x).split("_")[0] for x in experiment] + ["ensemble"],
#         [f"{r} s" for r in resolutions] + ["ensemble"],
#         figtitle=f"ROC, cv test data, n={np.asarray(y_cv).shape[1]}",
#         savepath=os.path.join(savedir_output, "cvtest"),
#     )

#     auc_cv_test = roc_auc_score(y_cv[0], np.asarray(p_ens_cv).mean(axis=0))
#     print("\nCV AUC, individual models: ", *(f"{x:.4f}," for x in auc_scales))
#     print(f"\nCV AUC, ensemble model: {auc_cv_test:.4f}")
#     print(
#         f'Classification report:\n{classification_report(y_cv[0] > THRESHOLD[likelihood], np.asarray(p_ens_cv).mean(axis=0) > THRESHOLD[likelihood], target_names=["CTRL", "NT1"])}'
#     )
#     print(
#         f"Confusion matrix:\n{confusion_matrix(y_cv[0] > THRESHOLD[likelihood], np.asarray(p_ens_cv).mean(axis=0) > THRESHOLD[likelihood])}"
#     )

#     if len(np.unique(y)) > 1:
#         print("Holdout test predictions:")
#         plot_roc_ensemble(
#             [y] * (n_exp + 1),
#             p_ens + [np.asarray(p_ens).mean(axis=0)],
#             THRESHOLD[likelihood],
#             # [os.path.basename(x).split("_")[0] for x in experiment] + ["ensemble"],
#             [f"{r} s" for r in resolutions] + ["ensemble"],
#             figtitle=f"ROC, test data, n={np.asarray(y_ens).shape[1]}",
#             savepath=os.path.join(savedir_output, "test"),
#         )
#     auc_scales = np.asarray(auc_scales)
#     # y_ens = np.asarray(y_ens)
#     p_ens = np.asarray(p_ens)
#     y_ens = np.ones_like(p_ens.mean(axis=0))
#     y_ens[p_ens.mean(axis=0) < THRESHOLD[likelihood]] = -1
#     p_ens_w = (p_ens.T @ auc_scales[:, np.newaxis]).squeeze() / auc_scales.sum()
#     if len(np.unique(y)) > 1:
#         auc_ens = roc_auc_score(y, p_ens.mean(axis=0))
#         auc_ens_w = roc_auc_score(y, p_ens_w)

#     df_preds = (
#         # df[["ID", "Cohort", "Label_x"]]
#         df[["ID", "Cohort", "Dx"]]
#         .copy()
#         .reset_index(drop=True)
#         .rename_axis(index=None, columns=None)
#         # .rename(columns={"Label_x": "Label"})
#     )
#     df_preds["p"] = p_ens.mean(axis=0)
#     df_preds["y"] = y_ens
#     if data_master is not None:
#         df_preds = (
#             df_preds.merge(df_master, left_on=["ID", "Cohort", "Dx"], right_on=["OakFileName", "Cohort", "Dx"])
#             # .drop(
#             #     [
#             #         "Sleep scoring train",
#             #         "Sleep scoring test",
#             #         "Narcolepsy train",
#             #         "Narcolepsy test",
#             #         "Replication",
#             #         "High pretest",
#             #         "Label_y",
#             #     ],
#             #     axis=1,
#             # )
#             # .rename(columns={"Label_x": "Label"})
#         )
#     df_preds["Resolution"] = "Ensemble"
#     df_ensemble.append(df_preds)
#     df_ensemble = pd.concat(df_ensemble)
#     df_ensemble.to_csv(os.path.join(os.path.dirname(savedir_output), "ensemble_predictions.csv"))

#     print(" ")
#     print("Ensemble model test predictions stratified by cohort")
#     cohorts = {k: [k] for k in df_preds["Cohort"].replace(["2AHC", "AHC"], "AHC").unique().tolist()}
#     try:
#         cohorts["AHC"] = ["2AHC", "AHC"]
#     except:
#         pass
#     y_stratified = [
#         df_preds.query(f"Cohort.isin({cohort})")["Dx"].map({"control": 0, "NT1": 1}).values for cohort in cohorts.values()
#     ]
#     p_stratified = [df_preds.query(f"Cohort.isin({cohort})")["p"].values for cohort in cohorts.values()]
#     n_stratified = [y.shape[0] for y in y_stratified]
#     if len(np.unique(y)) > 1:
#         plot_roc_ensemble(
#             y_stratified + [y],
#             p_stratified + [np.asarray(p_ens).mean(axis=0)],
#             THRESHOLD[likelihood],
#             [f"{cohort}, n={n}" for cohort, n in zip(cohorts.keys(), n_stratified)] + ["ensemble"],
#             figtitle="ROC, test data by cohort",
#             savepath=os.path.join(savedir_output, "test_stratitifed"),
#         )

#     if "HLA" in df_preds.columns:
#         print(" ")
#         print("Running HLA optimization on ensemble model")
#         if df_preds["HLA"].dtype != "int64":
#             hla = df_preds["HLA"].astype("Int64")  # .map({"0": 0, "1": 1})
#             print("Converting HLA to int")
#         else:
#             hla = df_preds["HLA"]
#         hla_adjusted_preds = df_preds["p"].copy()
#         hla_adjusted_preds[hla == 0] = 0.0
#         hla_y = df_preds["Dx"].map({"control": 0, "NT1": 1}).values
#         auc_test_hla = roc_auc_score(hla_y, hla_adjusted_preds)
#         print(f"y: {y}")
#         print(f"hla_y: {hla_y}")
#         print(f"p_ens: {p_ens}")
#         print(f"hla_adjusted_preds: {hla_adjusted_preds}")
#         plot_roc_ensemble(
#             [y, hla_y],
#             [p_ens.mean(axis=0), hla_adjusted_preds],
#             THRESHOLD[likelihood],
#             [f"w/o HLA, n={y_ens.shape[0]}", f"w/ HLA, n={hla_y.shape[0]}"],
#             figtitle="ROC, test data, HLA-dependence",
#             savepath=os.path.join(savedir_output, "test_hla"),
#         )

#     if len(np.unique(y)) > 1:
#         print("\nAUC, individual models: ", *(f"{x:.4f}," for x in model_aucs))
#         print(f"\nAUC, ensemble model: {auc_ens:.4f}")
#     print(
#         f'Classification report:\n{classification_report(y > THRESHOLD[likelihood], p_ens.mean(axis=0) > THRESHOLD[likelihood], target_names=["CTRL", "NT1"])}'
#     )
#     print(f"Confusion matrix:\n{confusion_matrix(y > THRESHOLD[likelihood], p_ens.mean(axis=0) > THRESHOLD[likelihood])}")
#     if len(np.unique(y)) > 1:
#         print(f"\nAUC, ensemble model, weighted: {auc_ens_w:.4f}")
#     print(
#         f'Classification report:\n{classification_report(y > THRESHOLD[likelihood], p_ens_w > THRESHOLD[likelihood], target_names=["CTRL", "NT1"])}'
#     )
#     print(f"Confusion matrix:\n{confusion_matrix(y > THRESHOLD[likelihood], p_ens_w > THRESHOLD[likelihood])}")

#     if "HLA" in df_preds.columns:
#         if len(np.unique(y)) > 1:
#             print(f"\nAUC, ensemble model, HLA: {auc_test_hla:.4f}")
#         print(
#             f'Classification report:\n{classification_report(hla_y > THRESHOLD[likelihood], hla_adjusted_preds.values > THRESHOLD[likelihood], target_names=["CTRL", "NT1"])}'
#         )
#         print(
#             f"Confusion matrix:\n{confusion_matrix(hla_y > THRESHOLD[likelihood], hla_adjusted_preds.values > THRESHOLD[likelihood])}"
#         )

#         print(" ")
#         print("Running HLA optimization ONLY!")
#         if df_preds["HLA"].dtype != "int64":
#             hla = df_preds["HLA"].astype("Int64")  # .map({"0": 0, "1": 1})
#         else:
#             hla = df_preds["HLA"]
#         hla_y = df_preds["Dx"].map({"control": 0, "NT1": 1}).values
#         if len(np.unique(y)) > 1:
#             auc_test_hla_only = roc_auc_score(hla_y, hla.values)
#             plot_roc_ensemble(
#                 [y, hla_y],
#                 [p_ens.mean(axis=0), hla.values],
#                 THRESHOLD[likelihood],
#                 [f"w/o HLA, n={y_ens.shape[0]}", f"w/ HLA, n={hla_y.shape[0]}"],
#                 figtitle="ROC, test data, HLA-dependence",
#                 savepath=os.path.join(savedir_output, "test_hla-only"),
#             )

#             print(f"\nAUC, ensemble model, HLA: {auc_test_hla_only:.4f}")
#         print(
#             f'Classification report:\n{classification_report(hla_y > THRESHOLD[likelihood], hla.values, target_names=["CTRL", "NT1"])}'
#         )
#         print(f"Confusion matrix:\n{confusion_matrix(hla_y > THRESHOLD[likelihood], hla.values)}")

#     # Looping over specific cohorts
#     # print(df_preds.Cohort.unique())
#     if len(df_preds.Cohort.unique()) > 1:
#         df_preds["Cohort"] = df_preds["Cohort"].replace("2AHC", "AHC")
#         # print(df_preds.Cohort.unique())
#         for cohort in df_preds["Cohort"].unique():
#             print(" ")
#             print(f"Removing {cohort} from ensemble model")
#             not_cohort = df_preds["Cohort"] != cohort
#             y_no_cohort = df_preds.loc[not_cohort, "Dx"].map({"control": 0, "NT1": 1}).values
#             p_no_cohort = df_preds.loc[not_cohort, "p"].values
#             if len(np.unique(y)) > 1:
#                 auc_test_no_cohort = roc_auc_score(y_no_cohort, p_no_cohort)
#                 plot_roc_ensemble(
#                     [y, y_no_cohort],
#                     [p_ens.mean(axis=0), p_no_cohort],
#                     THRESHOLD[likelihood],
#                     [f"w/ {cohort} n={y_ens.shape[0]}", f"w/o {cohort}, n={y_no_cohort.shape[0]}"],
#                     figtitle=f"ROC, test data, {cohort} dependence",
#                     savepath=os.path.join(savedir_output, f"test_{cohort}_no-{cohort}"),
#                 )

#                 print(f"\nAUC, ensemble model, no {cohort}: {auc_test_no_cohort:.4f}")
#             print(
#                 f'Classification report:\n{classification_report(y_no_cohort > THRESHOLD[likelihood], p_no_cohort > THRESHOLD[likelihood], target_names=["CTRL", "NT1"])}'
#             )
#             print(
#                 f"Confusion matrix:\n{confusion_matrix(y_no_cohort > THRESHOLD[likelihood], p_no_cohort > THRESHOLD[likelihood])}"
#             )

#             if "HLA" in df_preds.columns:
#                 print(" ")
#                 print(f"Running HLA optimization on ensemble model without {cohort}")
#                 not_cohort = df_preds["Cohort"] != cohort
#                 if df_preds["HLA"].dtype != "int64":
#                     hla_no_cohort = df_preds.loc[not_cohort, "HLA"].astype("Int64")  # .map({"0": 0, "1": 1})
#                 else:
#                     hla_no_cohort = df_preds.loc[not_cohort, "HLA"]
#                 hla_adjusted_preds_no_cohort = df_preds.loc[not_cohort, "p"].copy()
#                 hla_adjusted_preds_no_cohort[hla_no_cohort == 0] = 0.0
#                 hla_y_no_cohort = df_preds.loc[not_cohort, "Dx"].map({"control": 0, "NT1": 1}).values
#                 if len(np.unique(y)) > 1:
#                     auc_test_hla_no_cohort = roc_auc_score(hla_y_no_cohort, hla_adjusted_preds_no_cohort.values)
#                     plot_roc_ensemble(
#                         [y_no_cohort, hla_y_no_cohort],
#                         [p_no_cohort, hla_adjusted_preds_no_cohort.values],
#                         THRESHOLD[likelihood],
#                         [f"w/o HLA, n={y_no_cohort.shape[0]}", f"w/ HLA, n={hla_y_no_cohort.shape[0]}"],
#                         figtitle=f"ROC, test data ÷{cohort}, HLA-dependence",
#                         savepath=os.path.join(savedir_output, f"test_hla_no-{cohort}"),
#                     )

#                     print(f"\nAUC, ensemble model, HLA, no {cohort}: {auc_test_hla_no_cohort:.4f}")
#                 print(
#                     f'Classification report:\n{classification_report(hla_y_no_cohort > THRESHOLD[likelihood], hla_adjusted_preds_no_cohort.values > THRESHOLD[likelihood], target_names=["CTRL", "NT1"])}'
#                 )
#                 print(
#                     f"Confusion matrix:\n{confusion_matrix(hla_y_no_cohort > THRESHOLD[likelihood], hla_adjusted_preds_no_cohort.values > THRESHOLD[likelihood])}"
#                 )

#                 print(" ")
#                 print(f"Running HLA optimization on ensemble model {cohort} only")
#                 is_cohort = df_preds["Cohort"] == cohort
#                 if df_preds["HLA"].dtype != "int64":
#                     hla_is_cohort = df_preds.loc[is_cohort, "HLA"].astype("Int64")  # .map({"0": 0, "1": 1})
#                 else:
#                     hla_is_cohort = df_preds.loc[is_cohort, "HLA"]
#                 hla_adjusted_preds_is_cohort = df_preds.loc[is_cohort, "p"].copy()
#                 hla_adjusted_preds_is_cohort[hla_is_cohort == 0] = 0.0
#                 hla_y_is_cohort = df_preds.loc[is_cohort, "Dx"].map({"control": 0, "NT1": 1}).values
#                 y_is_cohort = df_preds.loc[is_cohort, "Dx"].map({"control": 0, "NT1": 1}).values
#                 p_is_cohort = df_preds.loc[is_cohort, "p"].values
#                 if len(np.unique(y)) > 1:
#                     auc_test_hla_is_cohort = roc_auc_score(hla_y_is_cohort, hla_adjusted_preds_is_cohort.values)
#                     auc_test_is_cohort = roc_auc_score(y_is_cohort, p_is_cohort)
#                     plot_roc_ensemble(
#                         [y_is_cohort, hla_y_is_cohort],
#                         [p_is_cohort, hla_adjusted_preds_is_cohort.values],
#                         THRESHOLD[likelihood],
#                         [f"w/o HLA, n={y_is_cohort.shape[0]}", f"w/ HLA, n={hla_y_is_cohort.shape[0]}"],
#                         figtitle=f"ROC, test data, {cohort}, HLA-dependence",
#                         savepath=os.path.join(savedir_output, f"test_hla_{cohort}-only"),
#                     )

#                     print(f"\nAUC, ensemble model, {cohort} only: {auc_test_is_cohort:.4f}")
#                 print(
#                     f'Classification report:\n{classification_report(y_is_cohort > THRESHOLD[likelihood], p_is_cohort > THRESHOLD[likelihood], target_names=["CTRL", "NT1"])}'
#                 )
#                 print(
#                     f"Confusion matrix:\n{confusion_matrix(y_is_cohort > THRESHOLD[likelihood], p_is_cohort > THRESHOLD[likelihood])}"
#                 )
#                 if len(np.unique(y)) > 1:
#                     print(f"\nAUC, ensemble model, HLA, {cohort} only: {auc_test_hla_is_cohort:.4f}")
#                 print(
#                     f'Classification report:\n{classification_report(hla_y_is_cohort > THRESHOLD[likelihood], hla_adjusted_preds_is_cohort.values > THRESHOLD[likelihood], target_names=["CTRL", "NT1"])}'
#                 )
#                 print(
#                     f"Confusion matrix:\n{confusion_matrix(hla_y_is_cohort > THRESHOLD[likelihood], hla_adjusted_preds_is_cohort.values > THRESHOLD[likelihood])}"
#                 )


if __name__ == "__main__":

    def none_or_str(value):
        if value == "None":
            return None

        return value

    def none_or_path(value):
        if value == "None":
            return None

        return Path(value)

    parser = ArgumentParser()
    parser.add_argument("-e", "--experiment-dir", required=True, type=Path, help="Path to directory containing trained models.")
    parser.add_argument(
        "-r",
        "--resolutions",
        default=None,
        nargs="+",
        type=int,
        help="Specific resolutions to include. If None, all resolutions will be used.",
    )
    parser.add_argument("-n", "--n_repeats", default=5, type=int)
    parser.add_argument("--data-master", type=none_or_path, default="data_files/data-overview-stanford-takeda.csv")
    args = parser.parse_args()

    train_ensemble(
        experiment_dir=args.experiment_dir, resolutions=args.resolutions, data_master=args.data_master, n_repeats=args.n_repeats
    )
