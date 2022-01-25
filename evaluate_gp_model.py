import json
import logging
import os
import pickle
import warnings

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm

from train_gp_model import THRESHOLD
from utils import plot_roc_ensemble


def evaluate_single_model(data=None, experiment=None, savedir_output=None):

    assert all(
        [data, experiment]
    ), f"Please specifiy both data and model to be evaluated, received data={data}, experiment={experiment}."
    assert (isinstance(experiment, str) and isinstance(data, str)) or (
        isinstance(experiment, list) and isinstance(data, list)
    ), "Please supply both data and experiment as either both lists, or strings!"

    if isinstance(experiment, str) and isinstance(data, str):
        data = [data]
        experiment = [experiment]
    else:
        n_exp = len(experiment)
        y_ens = [[]] * n_exp
        p_ens = [[]] * n_exp
        model_aucs = [[]] * n_exp
        auc_scales = [[]] * n_exp

    # Run over experiments
    print(f"Evaluating {len(experiment)} experiments:")
    for i, (d, exp) in enumerate(zip(data, experiment)):

        print("")
        print(f"Evaluating experiment: {exp}")
        print(f"Data: {d}")

        # Get settings
        with open(os.path.join(exp, "settings.json"), "r") as fp:
            settings = json.load(fp)
        feature_set = settings["feature_set"]
        resolution = settings["resolution"]
        model = settings["model"]
        n_kfold = settings["n_kfold"]
        n_repeats = settings["n_repeats"]
        likelihood = settings["likelihood"]
        do_pca = settings["pca"]
        feature_selection_path = os.path.join("data", "feature_selection", f"{model}_r{resolution:04}")

        # Get data with specific features
        df_master = pd.read_csv("data_master.csv", index_col=0)
        if "filtered" in feature_set:
            df = (
                pd.read_csv(d, index_col=0)
                # .pivot_table(index=["ID", "Cohort", "Label"], columns=["Resolution", "Feature"], values="Value")
                # .reset_index()
                # .sort_values(["Cohort", "ID"])
            )
            if feature_set == "filtered_long":
                filtered_features_idx = pd.read_csv(os.path.join(os.path.dirname(d), "filtered_features_long.csv"), index_col=0)
            elif feature_set == "filtered":
                filtered_features_idx = pd.read_csv(os.path.join(os.path.dirname(d), "filtered_features.csv"), index_col=0)
            df = (
                df.merge(filtered_features_idx)
                .drop("EffectSize", axis=1)
                .pivot_table(index=["ID", "Cohort", "Label"], columns=["Feature"], values="Value")
                .reset_index()
                .sort_values(["Cohort", "ID"])
            )
            X = df.iloc[:, 3:].values
            N, M = X.shape
            selected_features = list(df.columns[3:].values)

            # # Normalize data
            # X = (X - np.nanpercentile(X, 50, axis=0)) / (np.nanpercentile(X, 85, axis=0) - np.nanpercentile(X, 15, axis=0) + 1e-5)
        else:
            df = pd.read_csv(d, index_col=0)
            test_idx = df.merge(df_master, left_on=["ID", "Cohort"], right_on=["ID", "Cohort"])["Narcolepsy test data"]
            df = df.loc[test_idx == 1]
            X = df.iloc[:, 3:].values

            N, M = X.shape
            # fmt: off
            if feature_set == 'all':
                selected_features = range(M)
            elif feature_set == 'ncomm':
                # selected_features = [1, 11, 16, 22, 25, 41, 43, 49, 64, 65, 86, 87, 103, 119, 140, 147, 149, 166, 196, 201, 202, 220, 244, 245, 261, 276, 289, 296, 299, 390, 405, 450, 467, 468, 470, 474, 476, 477]
                # selected_features = [1, 4, 11, 16, 22, 25, 41, 43, 49, 64, 65, 86, 87, 103, 119, 140, 147, 149, 166, 196, 201, 202, 220, 244, 245, 261, 276, 289, 296, 390, 405, 450, 467, 468, 470, 474, 476, 477]
                selected_features = [1, 4, 11, 16, 22, 25, 41, 43, 49, 64, 65, 86, 87, 103, 119, 140, 147, 149, 166, 196, 201, 202, 220, 244, 245, 261, 276, 289, 296, 390, 405, 450, 467, 468, 470, 474, 476, 477]
            elif feature_set == 'selected':
                selected_features = []
                for pth in sorted([os.path.join(feature_selection_path, d) for d in os.listdir(feature_selection_path) if '.pkl' in d]):
                    with open(pth, 'rb') as fp:
                        selected_features.append(pickle.load(fp)['features'])
                selected_features = np.array(selected_features).astype(int)
                selected_features = (selected_features[:, :selected_features.shape[1]//2] + selected_features[:, selected_features.shape[1]//2:]).sum(axis=0) > 0
            # fmt: on
            X = X[:, selected_features]
        y = (df["Label"].values)[:]
        y[y == 0] = -1
        # y = E['labels']

        # Get feature scalers
        with open(os.path.join(exp, "feature_scales.pkl"), "rb") as fp:
            scalers = pickle.load(fp)

        # Get PCA objects
        if do_pca:
            with open(os.path.join(exp, "pca_objects.pkl"), "rb") as fp:
                pcas = pickle.load(fp)

        # Get list of model repeats and folds
        models = sorted(Path(exp).rglob("*.gpm"))

        # Get model scale weights
        model_weights = pd.read_csv(os.path.join(exp, "model_scales.csv"), index_col=0).cv_auc.values[:, np.newaxis]

        # Load CV data to obtain CV test AUC
        with open(os.path.join(exp, "cv_data.pkl"), "rb") as fp:
            cv_data = pickle.load(fp)
        cv_test_p = (cv_data["mean"].T @ model_weights).squeeze()
        cv_test_y = cv_data["true_class"][0]
        cv_test_auc = roc_auc_score(cv_test_y, cv_test_p)

        # Run over all models
        mean_pred = np.zeros((n_repeats, n_kfold, N, 1))
        var_pred = np.zeros((n_repeats, n_kfold, N, 1))
        p_pred = np.zeros(N)
        y_pred = np.ones(N)

        for model in tqdm(models):
            # Get model info and saved model
            repeat_num = int(model.name.split(".")[0].split("_")[1][1:]) - 1
            fold_num = int(model.name.split(".")[0].split("_")[2][1:]) - 1
            loaded_model = tf.saved_model.load(str(model))

            # Load and apply correct feature scaling
            scaler = scalers[repeat_num][fold_num]
            X_scaled = scaler.transform(X)

            # Possibly do PCA
            if do_pca:
                pca_obj = pcas[repeat_num][fold_num]["obj"]
                X_scaled = pca_obj.transform(X_scaled)

            # Get predictions
            mean_pred[repeat_num, fold_num], var_pred[repeat_num, fold_num] = loaded_model.predict_y_compiled(X_scaled)

        # Average over folds and apply scaling to each repeat
        mean_pred = mean_pred.squeeze(-1)
        var_pred = var_pred.squeeze(-1)
        p_pred = (model_weights.T @ mean_pred.mean(axis=1) / model_weights.sum()).squeeze(0)
        y_pred[p_pred < THRESHOLD[likelihood]] = -1

        # Get performance
        # plot_roc(
        #     y, p_pred, THRESHOLD[likelihood], "Test data", savepath=os.path.join(exp, "figures", "roc_test_long.png")
        # )
        auc_test = roc_auc_score(y, p_pred)
        print("AUC test:", auc_test)
        print(
            f'Classification report, test:\n{classification_report(y > THRESHOLD[likelihood], p_pred > THRESHOLD[likelihood], target_names=["CTRL", "NT1"])}'
        )
        print(f"Confusion matrix, test:\n{confusion_matrix(y > THRESHOLD[likelihood], p_pred > THRESHOLD[likelihood])}")
        # logger.info(f"AUC on CV test data: {auc_eval}")
        try:
            y_ens[i] = y_pred
            p_ens[i] = p_pred
            model_aucs[i] = auc_test
            auc_scales[i] = cv_test_auc
            # auc_scales[i] = cv_test_auc[resolution]
        except:
            pass

        if True:
            # Save predictions to .csv
            master = (
                pd.read_csv("data_master.csv")
                .reset_index(drop=True)
                .drop(["Index", "Unnamed: 0"], axis=1)
                .rename_axis(index=None)
            )
            df_preds = df[["ID", "Cohort", "Label"]].copy().reset_index(drop=True).rename_axis(index=None, columns=None)
            df_preds["p"] = p_pred
            df_preds["y"] = y_pred
            df_preds["var"] = var_pred.mean(axis=(0, 1))
            df_preds["var_w"] = (model_weights.T @ var_pred.mean(axis=1) / model_weights.sum()).squeeze()
            df_preds["std"] = df_preds["var"].pow(1.0 / 2)
            df_preds["std_w"] = df_preds["var_w"].pow(1.0 / 2)
            df_preds = (
                df_preds.merge(master, left_on=["ID", "Cohort"], right_on=["ID", "Cohort"])
                .drop(
                    [
                        "Sleep scoring training data",
                        "Sleep scoring test data",
                        "Narcolepsy training data",
                        "Narcolepsy test data",
                        "Replication data",
                        "High pre-test",
                        "Label_y",
                        "Added by Alex",
                    ],
                    axis=1,
                )
                .rename(columns={"Label_x": "Label"})
            )
            df_preds.to_csv(os.path.join(exp, "test_predictions.csv"))

    if n_exp > 1:
        print(" ")
        print("Running ensemble predictions")
        print("============================")
        plot_roc_ensemble(
            [y] * (n_exp + 1),
            p_ens + [np.asarray(p_ens).mean(axis=0)],
            THRESHOLD[likelihood],
            [os.path.basename(x).split("_")[0] for x in experiment] + ["ensemble"],
            figtitle="ROC, test data",
            savepath=savedir_output,
        )
        auc_scales = np.asarray(auc_scales)
        y_ens = np.asarray(y_ens)
        p_ens = np.asarray(p_ens)
        p_ens_w = (p_ens.T @ auc_scales[:, np.newaxis]).squeeze() / auc_scales.sum()
        auc_ens = roc_auc_score(y, p_ens.mean(axis=0))
        auc_ens_w = roc_auc_score(y, p_ens_w)
        # auc_ens = roc_auc_score(y, p_ens.mean(axis=0))
        # plot_roc(
        #     y,
        #     p_ens.mean(axis=0),
        #     THRESHOLD[likelihood],
        #     "Test data, ensemble model",
        #     savepath=os.path.join("outputs", "roc_test_ensemble.png"),
        # )
        # plot_roc(
        #     y,
        #     p_ens_w,
        #     THRESHOLD[likelihood],
        #     "Test data, weighted ensemble model",
        #     savepath=os.path.join("outputs", "roc_test_ensemble_weighted.png"),
        # )
        print("AUC, individual models: ", *(f"{x:.4f}," for x in model_aucs))
        print(f"AUC, ensemble model: {auc_ens:.4f}")
        print(
            f'Classification report:\n{classification_report(y > THRESHOLD[likelihood], p_ens.mean(axis=0) > THRESHOLD[likelihood], target_names=["CTRL", "NT1"])}'
        )
        print(f"Confusion matrix:\n{confusion_matrix(y > THRESHOLD[likelihood], p_ens.mean(axis=0) > THRESHOLD[likelihood])}")
        print(f"AUC, ensemble model, weighted: {auc_ens_w:.4f}")
        print(
            f'Classification report:\n{classification_report(y > THRESHOLD[likelihood], p_ens_w > THRESHOLD[likelihood], target_names=["CTRL", "NT1"])}'
        )
        print(f"Confusion matrix:\n{confusion_matrix(y > THRESHOLD[likelihood], p_ens_w > THRESHOLD[likelihood])}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dir",
        default="data/narco_features/avg_kw21_long_test_unscaled.csv",
        type=str,
        help="Path to supplied data directory",
        nargs="+",
    )
    parser.add_argument(
        "-e", "--experiment_dir", required=True, type=str, help="Path to directory containing trained model.", nargs="+",
    )
    parser.add_argument(
        "-s",
        "--savedir_output",
        default=os.path.join("outputs", "roc_ensemble_default.png"),
        type=str,
        help="Path to save output ROC curves.",
    )
    args = parser.parse_args()

    evaluate_single_model(data=args.data_dir, experiment=args.experiment_dir, savedir_output=args.savedir_output)
