import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from narcolepsy_detector.data.data_class import SingleResolutionDataset
from narcolepsy_detector.utils.logger import get_logger


logger = get_logger()
THRESHOLD = dict(bernoulli=0.5, gaussian=0.0)


def evaluate_single_model(data=None, experiment=None, savedir_output=None, data_master=None):
    assert all(
        [data, experiment]
    ), f"Please specifiy both data and model to be evaluated, received data={data}, experiment={experiment}."
    assert (isinstance(experiment, str) and isinstance(data, str)) or (
        isinstance(experiment, list) and isinstance(data, list)
    ), "Please supply both data and experiment as either both lists, or strings!"

    if isinstance(savedir_output, str):
        savedir_output = Path(savedir_output)
    if isinstance(experiment, str) and isinstance(data, str):
        data = [data]
        experiment = [experiment]
    data = [Path(d) for d in data]
    experiment = [Path(exp) for exp in experiment]
    # n_exp = len(experiment)
    df_ensemble = []

    if savedir_output.stem == 'roc_ensemble':
        savedir_output = savedir_output.parent
    logger.info(f'Creating directory for saving predictions at {savedir_output}')
    savedir_output.mkdir(exist_ok=True, parents=True)

    # Run over experiments
    logger.info(f"Evaluating {len(experiment)} experiments:")
    for i, (d, exp) in enumerate(zip(data, experiment)):
        logger.info("")
        logger.info(f"Evaluating experiment: {exp}")
        logger.info(f"Data: {d}")

        # Get settings
        settings_path = exp / 'settings.json'
        if not settings_path.exists():
            settings_path = list(exp.glob('settings*'))[0]
        with open(settings_path, "r") as fp:
            settings = json.load(fp)
        
        baseline_model = settings.get("baseline_model", False)
        n_kfold = settings["n_kfold"]
        n_repeats = settings["n_repeats"]
        likelihood = settings["likelihood"]
        do_pca = settings["pca"]
        feature_selection = settings["feature_selection"]
        
        settings.pop('data_dir')
        settings.pop('data_master')

        # initialize dataset
        ds = SingleResolutionDataset(data_dir=d, data_master=data_master, **settings)

        # initialize variables
        mean_pred = np.zeros((n_repeats, n_kfold, ds.data.N, 1))
        var_pred = np.zeros((n_repeats, n_kfold, ds.data.N, 1))

        # Loop over repeats and folds
        for i in range(n_repeats):
            for j in range(n_kfold):

                # Get feature scalers
                scalers_path = exp / "feature_scales"
                if scalers_path.with_suffix('.pkl').exists():
                    scalers_path = scalers_path.with_suffix('.pkl')
                elif scalers_path.is_dir():
                    scalers_path = scalers_path / f"feature_scales_n{i + 1:02}_k{j + 1:02}.pkl"
                else:
                    logger.error("No feature scalers found, program stopping!")
                    return -1
                with open(scalers_path, 'rb') as fp:
                    scalers = pickle.load(fp)

                # Get PCA objects
                if do_pca:
                    pca_path = exp / "pca_objects"
                    if pca_path.with_suffix('.pkl').exists():
                        pca_path = pca_path.with_suffix('.pkl')
                    elif pca_path.is_dir():
                        pca_path = pca_path / f"pca_objects_n{i + 1:02}_k{j + 1:02}.pkl"
                    else:
                        logger.error("No PCA objects found, program stopping!")
                        return -1
                    with open(pca_path, 'rb') as fp:
                        pcas = pickle.load(fp) 

                # Get model scale weights
                model_weights_path = exp / "model_scales"
                if model_weights_path.with_suffix('.csv').exists():
                    model_weights_path = model_weights_path.with_suffix('.csv')
                    model_weights = pd.read_csv(model_weights_path, index_col=0)
                elif model_weights_path.is_dir():
                    model_weights_path = model_weights_path / f"model_scales_n{i + 1:02}_k{j + 1:02}.csv"
                else:
                    logger.error("No model weights found, program stopping!")
                    return -1

                # Get model
                model_path = exp / "models" / f"model_n{i + 1:02}_k{j + 1:02}"
                if baseline_model:
                    model_path = model_path.with_suffix(".lrm")
                    with open(model_path, 'rb') as pkl:
                        loaded_model = pickle.load(pkl)
                else:
                    model_path = model_path.with_suffix(".gpm")
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        loaded_model = tf.saved_model.load(model_path)

                scaler = scalers[i][j]
                X_scaled = scaler.transform(ds.data.X.values)

                # Possibly do PCA
                if do_pca:
                    pca_obj = pcas[i][j]["obj"]
                    X_scaled = pca_obj.transform(X_scaled)

                # If feature selection procedure was performed, select correct features
                if feature_selection:
                    fs_path = exp / "feature-selection" / f"selected-features_n{i + 1:02}_k{j + 1:02}.pkl"
                    with open(fs_path, 'rb') as pkl:
                        selected_features = np.array(pickle.load(pkl)['features']).astype(int)
                    selected_features = (
                        selected_features[: len(selected_features) // 2] + selected_features[len(selected_features) // 2 :]
                    ) > 0
                    X_scaled = X_scaled[:, selected_features]

                # Get predictions:
                if baseline_model:
                    mean_pred[i, j] = loaded_model.predict_proba(X_scaled)[:, -1][:, np.newaxis]
                    var_pred[i, j] = np.zeros_like(mean_pred[i, j])
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        mean_pred[i, j], var_pred[i, j] = loaded_model.predict_y_compiled(X_scaled)

        # Average over folds and apply scaling to each repeat
        mean_pred = mean_pred.squeeze(-1)
        var_pred = var_pred.squeeze(-1)
        p_pred = (mean_pred.mean(axis=1).T @ model_weights.values / model_weights.values.sum()).squeeze() if 'model_weights' in locals() else mean_pred.mean(axis=1).squeeze()
        y_pred = (p_pred >= THRESHOLD[likelihood]).astype(int)

        # Check out naive performance
        if len(ds.data.y.unique()) > 1:
            auc_test = roc_auc_score(ds.data.y, p_pred)
        else:
            auc_test = np.nan
        logger.info(f'AUC, all data: {auc_test:.4f}')
        logger.info(f'Classification report, all data: \n{classification_report(ds.data.y, y_pred, target_names=["CTRL", "NT1"])}')
        logger.info(f"Confusion matrix, all data: \n{confusion_matrix(ds.data.y, y_pred)}")

        # Append predictions to dataframe list
        df_preds = ds.data.df
        df_preds['p'] = p_pred
        df_preds['y'] = y_pred
        df_preds['var'] = var_pred.mean(axis=(0, 1))
        df_preds['var_w'] = var_pred.mean(axis=(0, 1)) # TODO: fix model weights and this
        df_preds['std'] = df_preds['var'].pow(1.0 / 2)
        df_preds['std_w'] = df_preds['var_w'].pow(1.0 / 2)
        df_ensemble.append(df_preds)

    logger.info('Saving ensemble and resolution predictions')
    df_ensemble = pd.concat(df_ensemble)
    df_ensemble.to_csv(savedir_output / "ensemble_predictions.csv")
    logger.info('Evaluation finished!')
