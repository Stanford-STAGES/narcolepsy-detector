import json
import pickle
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from narcolepsy_detector.preprocessing.functions import process_file
from narcolepsy_detector.utils.argparser import inference_arguments
from narcolepsy_detector.utils.logger import get_logger

THRESHOLD = dict(bernoulli=0.5, gaussian=0.0)
logger = get_logger()


def predict_single_resolution(file, model_dir, resolution):

    # Check the model directories for settings.json
    updated_model_dir = f"{resolution:07.2f}".replace(".", "-") if resolution < 1 else f"{resolution:04d}"
    settings_file = list(model_dir.rglob(f"r{updated_model_dir}*/settings.json"))[0]
    updated_model_dir = settings_file.parent
    with open(settings_file, "r") as f:
        settings = json.load(f)

    # Extract features and extend feature_vector
    feature_set = settings["feature_set"]
    features = np.array(process_file(file, resolution, feature_set)).reshape(1, -1)

    # Loop over repeats and folds
    mean_pred = np.zeros((settings["n_repeats"], settings["n_kfold"]))
    var_pred = np.zeros((settings["n_repeats"], settings["n_kfold"]))
    for i in range(settings["n_repeats"]):
        for j in range(settings["n_kfold"]):

            # Get feature scalers
            scalers_path = updated_model_dir / "feature_scales"
            if scalers_path.with_suffix(".pkl").exists():
                scalers_path = scalers_path.with_suffix(".pkl")
            elif scalers_path.is_dir():
                scalers_path = scalers_path / f"feature_scales_n{i + 1:02}_k{j + 1:02}.pkl"
            else:
                logger.error("No feature scalers found, program stopping!")
                return -1
            with open(scalers_path, "rb") as fp:
                scalers = pickle.load(fp)

            # Get PCA objects
            if settings["pca"]:
                pca_path = updated_model_dir / "pca_objects"
                if pca_path.with_suffix(".pkl").exists():
                    pca_path = pca_path.with_suffix(".pkl")
                elif pca_path.is_dir():
                    pca_path = pca_path / f"pca_objects_n{i + 1:02}_k{j + 1:02}.pkl"
                else:
                    logger.error("No PCA objects found, program stopping!")
                    return -1
                with open(pca_path, "rb") as fp:
                    pcas = pickle.load(fp)

            # Get model scale weights
            model_weights_path = updated_model_dir / "model_scales"
            if model_weights_path.with_suffix(".csv").exists():
                model_weights_path = model_weights_path.with_suffix(".csv")
            elif model_weights_path.is_dir():
                model_weights_path = model_weights_path / f"model_scales_n{i + 1:02}_k{j + 1:02}.csv"
            else:
                logger.error("No model scales found, program stopping!")
                return -1
            model_weights = pd.read_csv(model_weights_path, index_col=0)

            # Get model. Baseline models are saved as .lrm files, while GP models are saved as .gpm files
            model_path = updated_model_dir / "models" / f"model_n{i + 1:02}_k{j + 1:02}.pkl"
            if settings["baseline_model"]:
                model_path = model_path.with_suffix(".lrm")
                with open(model_path, "rb") as pkl:
                    loaded_model = pickle.load(pkl)
            else:
                model_path = model_path.with_suffix(".gpm")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    loaded_model = tf.saved_model.load(model_path)

            # Scale data
            scaler = scalers[i][j]
            X_scaled = scaler.transform(features)

            # Possibly do PCA
            if settings["pca"]:
                pca_obj = pcas[i][j]["obj"]
                X_scaled = pca_obj.transform(X_scaled)

            # If feature selection procedure was performed, select correct features
            if settings["feature_selection"]:
                fs_path = updated_model_dir / "feature-selection" / f"selected-features_n{i + 1:02}_k{j + 1:02}.pkl"
                with open(fs_path, "rb") as fp:
                    selected_features = np.array(pickle.load(fp)["features"]).astype(int)
                selected_features = (
                    selected_features[: len(selected_features) // 2] + selected_features[len(selected_features) // 2 :]
                ) > 0
                X_scaled = X_scaled[:, selected_features]

            # Get predictions
            if settings["baseline_model"]:
                mean_pred[i, j] = loaded_model.predict_proba(X_scaled)[:, -1]
                var_pred[i, j] = np.zeros(mean_pred[i, j].shape)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mean_pred[i, j], var_pred[i, j] = loaded_model.predict_y_compiled(X_scaled)

    # Average over folds and apply scaling to each repeat
    p_pred = (
        (mean_pred.mean(axis=1).T @ model_weights.values / model_weights.values.sum()).squeeze()
        if "model_weights" in locals()
        else mean_pred.mean(axis=1).squeeze()
    )
    v_pred = (
        (var_pred.mean(axis=1).T @ model_weights.values / model_weights.values.sum()).squeeze()
        if "model_weights" in locals()
        else var_pred.mean(axis=1).squeeze()
    )

    return p_pred, v_pred, None


def compute_ensemble(probs, vars, aucs, ensembling_method):

    if ensembling_method == "mean":
        p_ens = np.mean(probs, axis=1)
        v_ens = np.mean(vars, axis=1)
    elif ensembling_method == "median":
        p_ens = np.median(probs, axis=1)
        v_ens = np.median(vars, axis=1)
    elif ensembling_method == "auc":
        raise NotImplementedError("AUC ensembling not yet implemented")
    elif ensembling_method == "auc-inv":
        raise NotImplementedError("AUC-inv ensembling not yet implemented")

    return p_ens, v_ens


def run_inference(data_dir, model_dir, savedir_output, resolutions, ensembling_method):

    # Get all the files in the directory
    files = list(sorted(data_dir.glob("*.pkl")))
    n_files = len(files)
    n_resolutions = len(resolutions)
    dfs = []
    probs = np.empty((n_files, n_resolutions))
    vars = np.empty((n_files, n_resolutions))
    res = np.empty((n_files, n_resolutions))
    aucs = np.empty((n_files, n_resolutions))

    # Predict for a single resolution
    logger.info(f"Running narcolepsy inference on {n_files} files for {n_resolutions} resolutions")
    for j, resolution in enumerate(resolutions):

        # Predict for each file in turn
        file_ids = []
        for i, file in enumerate(tqdm(files, desc=f"Resolution: {resolution}")):
            probs[i, j], vars[i, j], aucs[i, j] = predict_single_resolution(file, model_dir, resolution)
            file_ids.append(file.stem.split("preds_")[1])
            res[i, j] = resolution

        # Save df for this resolution
        logger.info(f'Saving predictions for resolution: {resolution}...')
        dfs.append(pd.DataFrame({"file_id": file_ids, "resolution": res[:, j], "prob": probs[:, j], "var": vars[:, j], "auc": aucs[:, j]}))
        dfs[-1].to_csv(savedir_output / f"predictions_r{resolution}.csv", index=False)

    # Save ensemble predictions
    p_ens, v_ens = compute_ensemble(probs, vars, aucs, ensembling_method)

    print("Saving ensemble predictions to disk...")
    dfs.append(pd.DataFrame({"file_id": file_ids, "resolution": ['ensemble'] * n_files, "prob": p_ens, "var": v_ens, "auc": [None] * n_files}))
    df = pd.concat(dfs).reset_index(drop=True)
    df.to_csv(savedir_output / "predictions.csv", index=False)


def main_cli():
    """
    Main entry point for running inference from the command line.
    """

    args = inference_arguments()
    args.savedir_output.mkdir(exist_ok=True, parents=True)
    run_inference(args.data_dir, args.model_dir, args.savedir_output, args.resolutions, args.ensembling_method)


if __name__ == "__main__":
    main_cli()
