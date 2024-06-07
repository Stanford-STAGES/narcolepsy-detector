import itertools
import json
import logging
import os
import pickle
import time
from functools import partial

import gpflow as gpf
import numpy as np
import pandas as pd
import sklearn as skl
import tensorflow as tf

# import wandb

from imblearn.over_sampling import SMOTE
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

from narcolepsy_detector.utils.adjusted_f1 import adjusted_f1_fn
from narcolepsy_detector.utils.logger import get_logger, print_args, add_file_handler, remove_file_handler
from narcolepsy_detector.utils.specific_sensitivity import specific_sensitivity_fn
from narcolepsy_detector.utils import plot_roc

logger = get_logger()

kernels = dict(
    rbf=gpf.kernels.SquaredExponential,
    rq=gpf.kernels.RationalQuadratic,
)

likelihoods = dict(
    bernoulli=gpf.likelihoods.Bernoulli,
    gaussian=gpf.likelihoods.Gaussian,
)
THRESHOLD = dict(bernoulli=0.5, gaussian=0.0)


def train_single_model(
    model,
    resolution,
    feature_set,
    n_kfold,
    n_repeats,
    data_dir,
    save_dir,
    kernel,
    inducing,
    likelihood,
    n_iter,
    log_frequency,
    n_inducing_points,
    gp_model,
    baseline,
    options=None,
    **kwargs,
):
    # cv_run = wandb.init(project="narcolepsy-detector")

    os.makedirs(save_dir, exist_ok=True)
    add_file_handler(logger, save_dir)
    print_args(logger, options)
    if not isinstance(options, dict):
        options = vars(options)

    # Adjust the save paths and file logging if we are in an array job
    if options.get('current_repeat', None) and options.get('current_fold'):
        added_string = f"_n{options['current_repeat']:02}_k{options['current_fold']:02}"
        add_file_handler(logger, save_dir, 'model' + added_string + ".log")
        settings_savepath = os.path.join(save_dir, "settings" + added_string + ".json")
    else:
        settings_savepath = os.path.join(save_dir, "settings.json")
    logger.info(f"Saving settings at {settings_savepath}")
    with open(settings_savepath, "w") as fp:
        json.dump(options, fp, sort_keys=True, indent=4)

    model_base = "_".join(model.split("_")[0:2])
    model_path = os.path.join(save_dir, "models")
    figure_path = os.path.join(save_dir, "figures")
    feature_selection_path = os.path.join("data", "feature_selection", f"{model_base}_r{resolution:04}")
    # tf_logdir = os.path.join(save_dir, "tboard")
    if not os.path.exists(model_path):
        # print('{} | Creating directory: {}'.format(datetime.now(), model_path))
        logger.info(f"Creating directory: {model_path}")
        logger.info(f"Creating directory: {figure_path}")
        # logger.info(f"Creating TensorBoard directory: {tf_logdir}")
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(figure_path, exist_ok=True)
        # os.makedirs(tf_logdir)

    # gpModelsPath = os.path.join(sherlock_scratch_path, 'narco_models')
    # currentGpModelPath = os.path.join(gpModelsPath, model)
    # if not os.path.exists(currentGpModelPath):
    #     print('{} | Creating directory: {}'.format(datetime.now(), currentGpModelPath))
    #     os.makedirs(currentGpModelPath)
    # for p in [scalingPath, featurePath]:
    #     if not os.path.exists(p):
    #         print('{} | Creating directory: {}'.format(datetime.now(), p))
    #         os.makedirs(p)
    # config = type('obj', (object,), {'narco_feature_selection_path': None, 'narco_scaling_path': scalingPath, 'resolution': resolution})  # This is just a dummy object, since I don't want to use the config file
    # num_induction_points = np.arange(50, 350, 50)
    # n_kfold = 5
    # skf = StratifiedKFold(n_kfold)

    # if inducing == 'data':
    #     num_induction_points = 350
    # num_induction_points = 350
    num_induction_points = n_inducing_points
    skf = model_selection.RepeatedStratifiedKFold(n_splits=n_kfold, n_repeats=n_repeats)
    ###

    # extract_features = ExtractFeatures(config)

    # print('{} | Current model: {}'.format(datetime.now(), model))
    logger.info(f"Current sleep scoring model: {model}")

    # Data loading
    # with open(os.path.join(featurePath, model + '_trainD.p'), 'rb') as f:
    #     E = pickle.load(f)
    # X = extract_features.scale_features(E['features'], model).T
    # X = E['features'].T
    # logger.info('Reading data')
    logger.info(f"Reading data and selecting {feature_set} features")
    try:
        df_master = pd.read_csv(options["data_master"], sep=",", index_col=0)
    except:
        df_master = pd.read_csv(options["data_master"], sep=";")

    # IF we need to remove some specific subjects, list them here
    subjects_to_remove = ["BOGN00088"]
    if subjects_to_remove:
        logger.info(f"Dropping {len(subjects_to_remove)} subjects from data master")
        df_master.drop(subjects_to_remove, inplace=True)

    # If we run baseline experiments, we drop STAGES and the sleep scoring eval data (currently)
    if baseline:
        logger.info("Baseline mode configuration; removing STAGES and sleep stage data from data master")
        cohorts_to_remove = ["stages", "cfs", "chat", "mesa", "mros", "shhs", "ssc", "wsc"]
        df_master = df_master[~df_master.Cohort.isin(cohorts_to_remove)]

    if options["no_stages"]:
        logger.info("'No STAGES' configuration; removing STAGES data")
        cohorts_to_remove = ["stages"]
        df_master = df_master[~df_master.Cohort.isin(cohorts_to_remove)]

    # df_master = pd.read_excel("overview_file_cohorts.xlsx", engine="openpyxl", sheet_name="All", index_col=0, usecols="A:O")
    # # Hack: KHC IDs sometime contains lings id as well, cut that part
    # df_master.loc[df_master["Cohort"] == "KHC", "ID"] = df_master.loc[df_master["Cohort"] == "KHC", "ID"].apply(lambda x: x[:8])
    # df_master.loc[df_master["Cohort"] == "IHC", "ID"] = df_master.loc[df_master["Cohort"] == "IHC", "ID"].apply(lambda x: x[:5])
    if "filtered" in feature_set:
        try:
            df = pd.read_csv(
                os.path.join(data_dir, f"{model_base}_long_train_unscaled.csv"),
                index_col=0,
            )
        except FileNotFoundError:
            df = pd.read_csv(
                os.path.join(data_dir, f"{os.path.basename(data_dir)}_unscaled.csv"),
                index_col=0,
            )
        if feature_set == "filtered_long":
            filtered_features_idx = pd.read_csv(os.path.join(data_dir, "filtered_features_long.csv"), index_col=0)
        elif feature_set == "filtered":
            filtered_features_idx = pd.read_csv(os.path.join(data_dir, "filtered_features.csv"), index_col=0)
        df = (
            df.merge(filtered_features_idx)
            .drop("EffectSize", axis=1)
            .pivot_table(index=["ID", "Cohort", "Label"], columns=["Feature"], values="Value")
            .reset_index()
            .sort_values(["Cohort", "ID"])
        )
        X = df.iloc[:, 5:].values
        N, M = X.shape
        selected_features = list(df.columns[3:].values)

        # # Normalize data
        # X = (X - np.nanpercentile(X, 50, axis=0)) / (np.nanpercentile(X, 85, axis=0) - np.nanpercentile(X, 15, axis=0) + 1e-5)
    # elif os.path.isdir
    else:
        try:
            df = pd.read_csv(
                os.path.join(data_dir, f"{model_base}_long_r{resolution:02}_trainD_unscaled.csv"),
                index_col=0,
            )
        except FileNotFoundError:
            try:
                df = pd.read_csv(
                    os.path.join(data_dir, f"{model_base}_long_r{resolution:02}_unscaled.csv"),
                    index_col=0,
                )
            except FileNotFoundError:
                if resolution == 0.5:
                    df = pd.read_csv(
                        os.path.join(data_dir, f"r{resolution:.2f}_unscaled.csv"),
                        index_col=0,
                    )
                else:
                    df = pd.read_csv(
                        os.path.join(data_dir, f"r{resolution:02}_unscaled.csv"),
                        index_col=0,
                    )
                    df["Label"] = df["Label"].apply(lambda x: int(x))
        try:
            train_idx = df.merge(df_master, left_on=["ID", "Cohort"], right_on=["ID", "Cohort"])[
                "Narcolepsy training data"
            ]
            df = df.loc[train_idx == 1]
        except:
            # df = df.merge(df_master, left_on=["ID", "Cohort"], right_on=["ID", "Cohort"])
            try:
                df = df.merge(
                    df_master[["ID", "Cohort", "Narcolepsy train"]],
                    left_on=["ID", "Cohort"],
                    right_on=["ID", "Cohort"],
                )
                df = df.loc[df["Narcolepsy train"] == 1]
            except:
                df = df_master.merge(df, left_on=["OakFileName", "Cohort"], right_on=["ID", "Cohort"]).rename(
                    columns={"training split": "Split", "Group": "Split"}
                )
                # df = df.merge(df_master, left_on=["ID", "Cohort"], right_on=["OakFileName", "Cohort"])
                # df["Split"] = df["training split"]
                # df = df.rename(columns={"training split": "Split"})
                df = df.query('Split == "training" or Split == "train"')

        # DEBUG
        # df = df.iloc[:1000]

        # IF we have any NaNs in the X data, we drop those subjects here
        # X = raw_data.drop(columns=self.drop_cols + ['Split'], errors='ignore').copy()
        notnan = df.drop(columns=df.columns[slice(0, int(np.where(df.columns == "0")[0]))]).notna().all(axis=1)
        logger.info(f'[ DATA ] \tSkipping {(notnan == False).sum()} subjects with NaN values')
        df_old = df.copy()
        df = df.loc[notnan]

        # df = pd.read_csv('data/narco_features/avg_kw21_r15_trainD.csv', index_col=0)
        # X = df.loc[:, [str(r) for r in range(564)]].values
        X = df.iloc[:, slice(int(np.where(df.columns == "0")[0]), None)].values
        # X = X[:, :489]

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
        elif feature_set in ['dissociation', 'hypnodensity']:
            selected_features = range(31 * 15)
        elif feature_set in ['clinical', 'cli']:
            selected_features = range(465, 489)
        elif feature_set == 'mtm':
            selected_features = range(489, M)
        elif feature_set in ['old', 'no-mtm']:
            selected_features = range(489)
        elif feature_set == 'soremp':
            selected_features = range(468, 469)
        # fmt: on
        X = X[:, selected_features]
        # y = E['labels']
    N, M = X.shape
    # y = (df["Label"].values)[:]
    y = df.Dx.map({"control": -1, "NT1": 1}).values
    # y[y == 0] = -1
    shuf = np.random.permutation(len(y))
    X = X[shuf, :]
    y = y[shuf]

    logger.info(f"Number of observations: {X.shape[0]}")
    logger.info(f"Number of features: {X.shape[1]}")
    logger.info(f"Number of narcoleptics: {(y == 1.0).sum()}")

    # Run cross-validation
    fold_count = -1
    cv_eval_preds = dict()
    cv_eval_preds["mean"] = np.zeros((n_repeats, len(y), 1))
    cv_eval_preds["var"] = np.zeros((n_repeats, len(y), 1))
    cv_eval_preds["pred_class"] = np.zeros((n_repeats, len(y), 1))
    cv_eval_preds["true_class"] = np.zeros((n_repeats, len(y)))
    cv_eval_preds["train_idx"] = [[[] for _ in range(n_kfold)] for _ in range(n_repeats)]
    cv_eval_preds["test_idx"] = [[[] for _ in range(n_kfold)] for _ in range(n_repeats)]
    cv_train_preds = dict()
    cv_train_preds["mean"] = np.zeros((n_repeats, n_kfold, len(y), 1))
    cv_train_preds["var"] = np.zeros((n_repeats, n_kfold, len(y), 1))
    elbo = [[[] for j in range(n_kfold)] for i in range(n_repeats)]
    cv_models = [[[] for j in range(n_kfold)] for i in range(n_repeats)]
    cv_scalers = [[[] for j in range(n_kfold)] for i in range(n_repeats)]
    pcas = (
        [[{"obj": None, "n_components": None, "components": None} for j in range(n_kfold)] for i in range(n_repeats)]
        if options["pca"]
        else None
    )
    cv_acc = []
    cv_auc = []
    cv_ap = []
    mean = []
    var = []
    for (current_repeat, current_fold), (train, test) in zip(
        itertools.product(range(n_repeats), range(n_kfold)), skf.split(X, y)
    ):
        fold_count += 1
        if options.get('current_repeat', None) and options.get('current_fold'):
            if current_repeat != options['current_repeat'] - 1: # job arrays start at 1
                continue
            if current_fold != options['current_fold'] - 1: # job arrays start at 1
                continue
        # if fold_count == 6:
        #     pass
        # else:
        #     continue
        # print('{} | Fold {} of {}'.format(datetime.now(), fold_count + 1, n_kfold))
        # logger.info(f"Fold {fold+1} of {n_kfold * n_repeats}")
        logger.info(f"Current repeat: {current_repeat + 1}/{n_repeats}")
        logger.info(f"Current fold: {current_fold + 1}/{n_kfold}")

        # fold_run = wandb.init(
        #     group=cv_run.sweep_id, job_type=cv_run.name, name=f"{current_repeat:02}-{current_fold:02}", config=options
        # )

        # Index into data
        X_train = X[train, :]
        y_train = y[train]
        X_test = X[test, :]
        y_test = y[test]
        cv_eval_preds["train_idx"][current_repeat][current_fold] = train
        cv_eval_preds["test_idx"][current_repeat][current_fold] = test

        # Run scaling
        # cv_scalers[current_repeat][current_fold] = RobustScaler(quantile_range=(15.0, 85.0)).fit(X_train)
        cv_scalers[current_repeat][current_fold] = StandardScaler().fit(X_train)
        X_train = cv_scalers[current_repeat][current_fold].transform(X_train)
        X_test = cv_scalers[current_repeat][current_fold].transform(X_test)

        if options["feature_selection"]:
            logger.info("Running feature selection procedure for current fold")
            feature_select_method = options["feature_select_method"]
            if feature_select_method == "specific_sensitivity":
                feature_select_specificity = float(options["feature_select_specificity"])
                feature_select_method = partial(specific_sensitivity_fn, specificity=feature_select_specificity)
                logger.info(
                    f"\tFeature selection using highest sensitivity at {feature_select_specificity} specificity"
                )
            elif feature_select_method == "adjusted_f1":
                data_prevalence = (y_train == 1).mean()
                feature_select_prevalence = float(options["feature_select_prevalence"])
                feature_select_specificity = float(options["feature_select_specificity"])
                feature_select_method = partial(
                    adjusted_f1_fn,
                    prevalence=feature_select_prevalence,
                    data_prevalence=data_prevalence,
                    specificity=feature_select_specificity,
                )
                logger.info(
                    f"\tFeature selection using adjusted F1 score with {feature_select_prevalence:.5f} prevalence and {data_prevalence:.5f} data prevalence"
                )
            elif feature_select_method == "f1":
                logger.info("\tFeature selection using F1 score")

            def _feature_selection(features, labels, n_folds, baseline_model=False):
                features_squared = np.square(features)
                _X = np.concatenate([features, features_squared], axis=1)
                _y = labels

                if baseline_model:
                    estimator = skl.linear_model.LogisticRegressionCV(
                        cv=skl.model_selection.StratifiedKFold(n_splits=n_folds),
                        fit_intercept=False,
                        max_iter=10000,
                        n_jobs=5,
                        penalty="l1",
                        scoring=feature_select_method,
                        solver="saga",
                        verbose=0,
                        refit=True,
                    )
                # selector = skl.feature_selection.SelectFromModel(estimator=estimator, prefit=False)
                skf = skl.model_selection.StratifiedKFold(n_splits=n_folds)
                # estimator = skl.svm.SVC(kernel="linear")
                # estimator = skl.svm.SVC(kernel="rbf")
                # estimator = skl.ensemble.RandomForestClassifier()
                # estimator = skl.linear_model.LogisticRegression(solver="saga", max_iter=100000)
                # estimator = skl.linear_model.LogisticRegression(solver="saga", max_iter=10000)
                estimator = skl.linear_model.LogisticRegression(max_iter=10000)
                # estimator = skl.linear_model.LogisticRegression(penalty="l2", solver="saga", max_iter=100000)
                # estimator = skl.linear_model.LogisticRegression(penalty="l2", solver="lbfgs", max_iter=100000)
                rfecv = RFECV(
                    estimator=estimator,
                    step=1,
                    # step=max(_X.shape[-1] // 100, 1),
                    cv=skf,
                    scoring=feature_select_method,
                    # n_jobs=5,
                    verbose=0,
                )
                logger.info("Starting feature selection procedure")
                start = time.time()
                # selector.fit(_X, _y)
                rfecv.fit(_X, _y)
                end = time.time()
                logger.info(f"\tElapsed time: {end-start}")
                # relevant = selector.get_support()
                relevant = rfecv.get_support()
                n_features = relevant.sum()
                # score = selector.estimator_.score(_X, _y)
                # score = np.max(rfecv.grid_scores_)
                score = rfecv.cv_results_["mean_test_score"]
                logger.info(f"\tOptimal number of features: {n_features}")
                logger.info(
                    f"\tOptimal number of unique features: {sum(relevant[:len(relevant)//2] + relevant[len(relevant)//2:])}"
                )

                # Save selected features
                save_file_path = os.path.join(
                    save_dir,
                    "feature-selection",
                    f"selected-features_n{current_repeat+1:02}_k{current_fold+1:02}.pkl",
                )
                os.makedirs(os.path.join(save_dir, "feature-selection"), exist_ok=True)
                logger.info(f"Saving objects at {save_file_path}")
                selected_features = {
                    "features": relevant,
                    "score": score,
                    # "model": selector,
                    "model": rfecv,
                }
                with open(save_file_path, "wb") as fp:
                    pickle.dump(selected_features, fp)

                selected_features = np.array(relevant).astype(int)
                selected_features = (
                    selected_features[: len(selected_features) // 2] + selected_features[len(selected_features) // 2 :]
                ) > 0

                return selected_features

            selected_features = _feature_selection(X_train, y_train, 5, baseline_model=options["baseline_model"])
            X_train = X_train[:, selected_features]
            X_test = X_test[:, selected_features]
            logger.info(f"Using {X_train.shape[1]} features")
        else:
            selected_features = range(M)

        # Do PCA. We select the number of components that explain 90% of the variance
        if options["pca"]:
            logger.info("Running PCA")
            n_components = options["pca_components"]
            while True:
                logger.info(f"\tFitting to train data using {n_components} components")
                pca = PCA(n_components=n_components, whiten=True).fit(X_train)
                explained_var = np.cumsum(pca.explained_variance_ratio_ * 100)
                if explained_var[-1] >= options["pca_explained"]:
                    if explained_var[-2] < options["pca_explained"]:
                        logger.info(
                            f"\tPCA successful: {n_components} components accounting for {explained_var[-1]:.1f}% variance"
                        )
                        break
                    else:
                        logger.info("\tPCA running with excess components, adjusting n_components")
                        n_components = np.argwhere(explained_var >= options["pca_explained"])[0][0] + 1
                else:
                    logger.info("\tNot enough variance explained, adjusting n_components")
                    n_components *= 2

            logger.info("\tTransforming train and test partitions with PCA")
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
            N, M = X_train.shape
            pcas[current_repeat][current_fold]["obj"] = pca
            pcas[current_repeat][current_fold]["components"] = pca.components_
            pcas[current_repeat][current_fold]["n_components"] = n_components

        # Run SMOTE to oversample narcolepsy class
        # print('{} | Running SMOTE'.format(datetime.now()))
        if options["smote"]:
            logger.info("Running SMOTE")
            # X_res, y_res = SMOTE(k_neighbors=10, m_neighbors=20, n_jobs=-1).fit_sample(X_train, y_train)
            X_res, y_res = SMOTE(k_neighbors=10, n_jobs=5).fit_resample(X_train, y_train)
            logger.info(f"\tShape of X: {X_res.shape}")
        else:
            X_res, y_res = X_train, y_train

        if options["baseline_model"]:
            if options["feature_selection"]:
                m = skl.linear_model.LogisticRegression(
                    max_iter=10000,
                    n_jobs=5,
                    penalty=None,
                    verbose=0,
                )
            else:
                feature_select_method = options["feature_select_method"]
                if feature_select_method == "specific_sensitivity":
                    feature_select_specificity = float(options["feature_select_specificity"])
                    feature_select_method = partial(specific_sensitivity_fn, specificity=feature_select_specificity)
                    logger.info(
                        f"\tFeature selection using highest sensitivity at {feature_select_specificity} specificity"
                    )
                elif feature_select_method == "adjusted_f1":
                    data_prevalence = (y_train == 1).mean()
                    feature_select_prevalence = float(options["feature_select_prevalence"])
                    feature_select_specificity = float(options["feature_select_specificity"])
                    feature_select_method = partial(
                        adjusted_f1_fn,
                        prevalence=feature_select_prevalence,
                        data_prevalence=data_prevalence,
                        specificity=feature_select_specificity,
                    )
                    logger.info(
                        f"\tFeature selection using adjusted F1 score with {feature_select_prevalence:.5f} prevalence and {data_prevalence:.5f} data prevalence"
                    )
                elif feature_select_method == "f1":
                    logger.info("\tFeature selection using F1 score")

                m = skl.linear_model.LogisticRegressionCV(
                    cv=skl.model_selection.StratifiedKFold(n_splits=5),
                    fit_intercept=False,
                    max_iter=10000,
                    n_jobs=5,
                    penalty="l1",
                    scoring=feature_select_method,
                    solver="saga",
                    verbose=0,
                    refit=True,
                )

            logger.info(f"Starting optimization of baseline LR model ...")
            start = time.time()
            m.fit(X_res, y_res)
            end = time.time()

        else:
            # Set induction points
            if inducing == "data":
                logger.info("Using data for inducing points")
                inductionP = np.arange(len(y_train))
                np.random.shuffle(inductionP)
                inductionP = inductionP[:num_induction_points]
                inductionP = X_res[inductionP, :]
            elif inducing == "kmeans":
                from scipy.cluster.vq import kmeans

                logger.info("Using kmeans for inducing points")
                inductionP = kmeans(X_res, num_induction_points)[0]

            # Here we run the GP optimization
            # kernel = kernels.RBF(np.ones(len(selected_features)))
            # m_fix = GaussianProcessClassifier(kernel=kernel, n_jobs=-1, max_iter_predict=500, optimizer=None)
            # m_fix.fit(X_res, y_res)
            # m = GaussianProcessClassifier(kernel=kernel, n_jobs=-1, max_iter_predict=500)

            # New GPflow
            data = (X_res, y_res[:, np.newaxis])
            # data_minibatch = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(10000).batch(100).prefetch(tf.data.AUTOTUNE)
            # data_minibatch_it = iter(data_minibatch)

            custom_config = gpf.config.Config(jitter=1e-6)

            # custom_config = gpf.settings.get_settings()
            # custom_config.verbosity.tf_compile_verb = True
            # current_jitter = custom_config.numerics.jitter_level
            # current_jitter_magnitude = -6
            # custom_config.numerics.jitter_level = 10 ** current_jitter_magnitude
            restart = True
            while restart:
                # try:
                # with gpf.settings.temp_settings(custom_config), gpf.session_manager.get_session().as_default():
                with gpf.config.as_context(custom_config):
                    # Create model
                    logger.info(f"Using {gp_model} for variational inference")
                    if gp_model == "svgp":
                        m = gpf.models.SVGP(
                            kernel=kernels[kernel](lengthscales=np.ones(X_res.shape[1])),
                            likelihood=likelihoods[likelihood](),
                            inducing_variable=inductionP,
                            num_data=X_res.shape[0],
                        )
                        # objective = m.training_loss_closure(data_minibatch_it)
                        objective = m.training_loss_closure(data)
                    elif gp_model == "vgp":
                        m = gpf.models.VGP(
                            data,
                            kernel=kernels[kernel](lengthscales=np.ones(X_res.shape[1])),
                            likelihood=likelihoods[likelihood](),
                        )
                        objective = m.training_loss

                    # Create optimizer
                    optimizer = gpf.optimizers.Scipy()
                    # gpf.set_trainable(m.q_mu, False)
                    # gpf.set_trainable(m.q_sqrt, False)
                    # optimizer = tf.optimizers.Adam(0.001)
                    # adam_opt = tf.optimizers.Adam(0.001)
                    # nat_opt = gpf.optimizers.NaturalGradient(gamma=0.1)
                    # variational_params = [(m.q_mu, m.q_sqrt)]

                    @tf.function
                    def calculate_elbo():
                        return -objective()

                    # cv_logdir = os.path.join(tf_logdir, f"cv-{current_repeat:02}-{current_fold:02}")
                    # model_task = ModelToTensorBoard(cv_logdir, m)
                    # lml_task = ScalarToTensorBoard(cv_logdir, calculate_elbo, "elbo")
                    # tasks = MonitorTaskGroup([model_task, lml_task], period=log_frequency)
                    # monitor = Monitor(tasks)

                    # @tf.function
                    # def optimization_step(i):
                    # optimizer.minimize(objective, m.trainable_variables)
                    # monitor(i)
                    # adam_opt.minimize(objective, var_list=m.trainable_variables)
                    # nat_opt.minimize(objective, var_list=variational_params)

                    # m = gpf.models.VGP(
                    #     data=(X_res, y_res),
                    #     kernel=kernels[kernel](lengthscales=np.ones(len(selected_features))),
                    #     likelihood=gpf.likelihoods.Bernoulli(),
                    # )
                    # optimizer = gpf.optimizers.Scipy()

                    # variational_params = [(m.q_mu, m.q_sqrt)]
                    # Old GPflow
                    # m = gpf.models.SVGP(
                    #     X_res,
                    #     y_res[:, np.newaxis],
                    #     kern=kernels[kernel](len(selected_features), ARD=True),
                    #     likelihood=likelihoods[likehood],
                    #     Z=inductionP,
                    # )

                    # if kernel == "rbf":
                    #     m = gpf.models.SVGP(
                    #         X_res,
                    #         y_res[:, np.newaxis],
                    #         kern=gpf.kernels.RBF(len(selected_features), ARD=True),
                    #         likelihood=gpf.likelihoods.Gaussian(),
                    #         Z=inductionP,
                    #     )
                    # elif kernel == "rq":
                    #     m = gpf.models.SVGP(
                    #         X_res,
                    #         y_res[:, np.newaxis],
                    #         kern=gpf.kernels.RationalQuadratic(len(selected_features), ARD=True),
                    #         likelihood=gpf.likelihoods.Gaussian(),
                    #         Z=inductionP,
                    #     )
                    # m = gpf.models.GPRFITC(X_res, y_res.reshape((-1, 1)), kern=gpf.kernels.RBF(len(selected_features), ARD=True), Z=inductionP)
                    # print('{} | Starting optimization of GP model | Negative log likelihood before optimization: {}'.format(datetime.now(), m.compute_log_likelihood()))
                    logger.info(f"Starting optimization of GP model with {kernel} kernel and {likelihood} likelihood")
                    # logger.info(f"\tELBO before optimization: {m.elbo(data).numpy()}")
                    elbo_init = calculate_elbo().numpy()
                    logger.info(f"\tELBO before optimization: {elbo_init}")
                    # logger.info(f"\tNegative log likelihood before optimization: {m.compute_log_likelihood()}")
                    # print('{} | Starting optimization of GP model | Negative log likelihood before optimization: {}'.format(datetime.now(), m_fix.log_marginal_likelihood()))
                    start = time.time()
                    # m.fit(X_res, y_res)
                    # optimizer.minimize(training_loss, m.trainable_variables, options=dict(maxiter=100000))
                    # bar = tqdm(range(1000))
                    # for i in tqdm(tf.range(n_iter)):

                    # with trange(n_iter) as t:
                    #     for step in t:
                    # optimizer.minimize(objective, m.trainable_variables)
                    # optimization_step(current_fold)
                    try:
                        optimizer.minimize(objective, m.trainable_variables, options=dict(maxiter=n_iter))
                        # if step % log_frequency == 0:
                        #     elbo[current_repeat][current_fold].append(calculate_elbo(fold_count).numpy())
                        #     t.set_postfix(elbo=elbo[current_repeat][current_fold][-1])
                        # fold_run.log({"step": step, "ELBO": elbo[current_repeat][current_fold][-1]})
                        # adam_opt.minimize(objective, var_list=m.trainable_variables)
                        # nat_opt.minimize(objective, var_list=variational_params)
                        # t.set_postfix(nll=m.elbo(data).numpy())
                        # gpf.train.ScipyOptimizer().minimize(m, maxiter=100000)
                        if calculate_elbo().numpy() > elbo_init:
                            end = time.time()
                            restart = False
                        else:
                            current_jitter = gpf.config.config().jitter
                            current_jitter_exponent = np.floor(np.log10(np.abs(current_jitter)))
                            new_jitter_exponent = current_jitter_exponent + 1
                            new_jitter = 10**new_jitter_exponent
                            custom_config = gpf.config.Config(jitter=new_jitter)
                            logger.info(
                                f"\tCaptured an error, most likely in the Cholesky decomp. Raising jitter levels from {current_jitter} to {new_jitter} and retrying"
                            )
                    except:
                        current_jitter = gpf.config.config().jitter
                        current_jitter_exponent = np.floor(np.log10(np.abs(current_jitter)))
                        new_jitter_exponent = current_jitter_exponent + 1
                        new_jitter = 10**new_jitter_exponent
                        custom_config = gpf.config.Config(jitter=new_jitter)
                        logger.info(
                            f"\tCaptured an error, most likely in the Cholesky decomp. Raising jitter levels from {current_jitter} to {new_jitter} and retrying"
                        )
                # except:
                #     current_jitter = 10 ** current_jitter_magnitude
                #     new_jitter_magnitude = current_jitter_magnitude + 1
                #     new_jitter = 10 ** new_jitter_magnitude
                #     logger.info(
                #         f"\tCaptured an error, most likely in the Cholesky decomp. Raising jitter levels from {current_jitter} to {new_jitter} and retrying"
                #     )
                #     current_jitter_magnitude = new_jitter_magnitude
                #     current_jitter = new_jitter
                #     custom_config.numerics.jitter_level = current_jitter
                #     # with gpf.settings.temp_settings(custom_config), gpf.session_manager.get_session().as_default():
                #     #     if kernel == 'rbf':
                #     #         m = gpf.models.SVGP(X_res, y_res[:, np.newaxis], kern=gpf.kernels.RBF(len(selected_features), ARD=True), likelihood=gpf.likelihoods.Gaussian(), Z=inductionP)
                #     #     elif kernel == 'rq':
                #     #         m = gpf.models.SVGP(X_res, y_res[:, np.newaxis], kern=gpf.kernels.RationalQuadratic(len(selected_features), ARD=True), likelihood=gpf.likelihoods.Gaussian(), Z=inductionP)
                #     #     # m = gpf.models.GPRFITC(X_res, y_res.reshape((-1, 1)), kern=gpf.kernels.RBF(len(selected_features), ARD=True), Z=inductionP)
                #     #     # print('{} | Starting optimization of GP model | Negative log likelihood before optimization: {}'.format(datetime.now(), m.compute_log_likelihood()))
                #     #     logger.info(f'Starting optimization of GP model with {kernel} kernel')
                #     #     logger.info(f'\tNegative log likelihood before optimization: {m.compute_log_likelihood()}')
                #     #     start = time.time()
                #     #     gpf.train.ScipyOptimizer().minimize(m, maxiter=100000)
                #     #     end = time.time()
        logger.info(f"\tElapsed time: {end - start}")
        # logger.info(f"\tNegative log likelihood after optimization: {m.compute_log_likelihood()}")
        # logger.info(f"\tTraining loss after optimization: {m.training_loss(data=(X_res, y_res[:, np.newaxis]))}")
        # logger.info(f"\tELBO after optimization: {m.elbo(data).numpy()}")
        if options["baseline_model"]:
            # Get predictions
            mean_pred = m.predict_proba(X_test)[:, 1, np.newaxis]
            var_pred = np.zeros_like(mean_pred)

        else:
            logger.info(f"\tELBO after optimization: {calculate_elbo().numpy()}")

            # Get predictions
            mean_pred, var_pred = m.predict_y(X_test)

        # Get the prediction accuracy at 50% cutoff
        y_pred = np.ones(mean_pred.shape)
        y_pred[mean_pred < THRESHOLD[likelihood]] = -1
        acc = np.mean(np.squeeze(y_pred) == np.squeeze(y_test))
        cv_acc.append(acc)
        logger.info(f"\tFold accuracy: {acc}")

        # Save the model and save predictions
        cv_auc.append(roc_auc_score(y_test, mean_pred))
        cv_ap.append(average_precision_score(y_test, mean_pred))
        cv_models[current_repeat][current_fold] = m
        cv_eval_preds["mean"][current_repeat, test] = mean_pred
        cv_eval_preds["var"][current_repeat, test] = var_pred
        cv_eval_preds["pred_class"][current_repeat, test] = y_pred
        cv_eval_preds["true_class"][current_repeat, test] = y_test

        if options["baseline_model"]:
            model_savepath = (
                os.path.join(model_path, f"model_n{current_repeat + 1:02}_k{current_fold + 1:02}") + ".lrm"
            )
            with open(model_savepath, "wb") as pkl:
                pickle.dump(m, pkl)
        else:
            model_savepath = (
                os.path.join(model_path, f"model_n{current_repeat + 1:02}_k{current_fold + 1:02}") + ".gpm"
            )
            m.predict_y_compiled = tf.function(
                m.predict_y,
                input_signature=[tf.TensorSpec(shape=[None, X_train.shape[1]], dtype=tf.float64)],
            )
            tf.saved_model.save(m, model_savepath)
        logger.info(f"\tSaving model {model_savepath}")

        # ROC CURVES ON TRAINING DATA
        plot_roc(
            y_test,
            mean_pred,
            THRESHOLD[likelihood],
            f"CV eval data {current_repeat+1:02}/{current_fold+1:02}",
            savepath=os.path.join(figure_path, f"roc_n{current_repeat+1:02}_k{current_fold+1:02}.png"),
        )

        # Get predictions on entire training set
        if options["pca"]:
            if options["baseline_model"]:
                mean_train = m.predict_proba(
                    pcas[current_repeat][current_fold]["obj"].transform(
                        cv_scalers[current_repeat][current_fold].transform(X)[:, selected_features]
                    )
                )[:, 1, np.newaxis]
                var_train = np.zeros_like(mean_train)
            else:
                mean_train, var_train = m.predict_y(
                    pcas[current_repeat][current_fold]["obj"].transform(
                        cv_scalers[current_repeat][current_fold].transform(X)[:, selected_features]
                    )
                )
        else:
            if options["baseline_model"]:
                mean_train = m.predict_proba(
                    cv_scalers[current_repeat][current_fold].transform(X)[:, selected_features]
                )[:, 1, np.newaxis]
                var_train = np.zeros_like(mean_train)
            else:
                mean_train, var_train = m.predict_y(
                    cv_scalers[current_repeat][current_fold].transform(X)[:, selected_features]
                )
        cv_train_preds["mean"][current_repeat, current_fold, :] = mean_train
        cv_train_preds["var"][current_repeat, current_fold, :] = var_train
        mean.append(mean_train)
        var.append(var_train)

        logger.info("Cross-validation performance:")
        logger.info(f"\tAccuracy: {np.mean(cv_acc):.3f} +/- {np.std(cv_acc):.3f}")
        logger.info(f"\tAUC: {np.mean(cv_auc):.3f} +/- {np.std(cv_auc):.3f}")
        logger.info(f"\tAUC-PR: {np.mean(cv_ap):.3f} +/- {np.std(cv_ap):.3f}\n")

        # del m, training_loss, optimizer
    
    current_repeat = options.get('current_repeat', None)
    current_fold = options.get('current_fold', None)

    # Save the CV train and test data predictions
    if current_repeat and current_fold:
        cv_data_savepath = os.path.join(save_dir, 'cv_data')
        os.makedirs(cv_data_savepath, exist_ok=True)
        cv_data_savepath = os.path.join(cv_data_savepath, f'cv_data_n{current_repeat:02}_k{current_fold:02}.pkl')
    else:
        cv_data_savepath = os.path.join(save_dir, "cv_data.pkl")
    logger.info(f"Saving CV test and train info at {cv_data_savepath}")
    with open(cv_data_savepath, "wb") as fp:
        pickle.dump(cv_eval_preds, fp)

    # Save the feature scalers in directory
    if current_repeat and current_fold:
        feature_scale_savepath = os.path.join(save_dir, 'feature_scales')
        os.makedirs(feature_scale_savepath, exist_ok=True)
        feature_scale_savepath = os.path.join(feature_scale_savepath, f"feature_scales_n{current_repeat:02}_k{current_fold:02}.pkl")
    else:
        feature_scale_savepath = os.path.join(save_dir, "feature_scales.pkl")
    logger.info(f"Saving feature scale values at {feature_scale_savepath}")
    with open(feature_scale_savepath, "wb") as pkl:
        pickle.dump(cv_scalers, pkl)

    # Save the PCA objects in directory
    if options["pca"]:
        if current_repeat and current_fold:
            pca_savepath = os.path.join(save_dir, 'pca_objects')
            os.makedirs(pca_savepath, exist_ok=True)
            pca_savepath = os.path.join(pca_savepath, f"pca_objects_n{current_repeat:02}_k{current_fold:02}.pkl")
        else:
            pca_savepath = os.path.join(save_dir, "pca_objects.pkl")
        logger.info(f"Saving PCA objects at {pca_savepath}")
        with open(pca_savepath, "wb") as pkl:
            pickle.dump(pcas, pkl)

    # Save scales
    if current_repeat and current_fold:
        scale_savepath = os.path.join(save_dir, 'model_scales')
        os.makedirs(scale_savepath, exist_ok=True)
        scale_savepath = os.path.join(scale_savepath, f"model_scales_n{current_repeat:02}_k{current_fold:02}.csv")
    else:
        scale_savepath = os.path.join(save_dir, "model_scales.csv")
    logger.info(f"Saving scale values at {scale_savepath}")
    scale_df = pd.DataFrame({"cv_auc": [roc_auc_score(y, x.squeeze()) for x in cv_eval_preds["mean"]]})
    scale_df.to_csv(scale_savepath)
    ap_df = pd.DataFrame({"cv_ap": [average_precision_score(y, x.squeeze()) for x in cv_eval_preds["mean"]]})
    if current_repeat and current_fold:
        ap_scale_savepath = os.path.join(save_dir, 'model_scales_ap')
        os.makedirs(ap_scale_savepath, exist_ok=True)
        ap_df.to_csv(os.path.join(ap_scale_savepath, f"model_scales_ap_n{current_repeat:02}_k{current_fold:02}.csv"))
    else:
        ap_df.to_csv(os.path.join(save_dir, "model_scales_ap.csv"))

    if current_repeat and current_fold:
        return 1

    # Evaluate on eval data from CV folds
    logger.info("Running model evaluation on CV test data.")
    scale = scale_df["cv_auc"].values
    y_pred = (cv_eval_preds["mean"].T @ scale[:, np.newaxis]).squeeze()
    plot_roc(
        y,
        y_pred,
        THRESHOLD[likelihood],
        "CV eval data",
        savepath=os.path.join(figure_path, "roc_cv_eval.png"),
    )
    auc_eval = roc_auc_score(y, y_pred)
    ap_eval = average_precision_score(y, y_pred)
    logger.info(f"AUC on CV test data: {auc_eval}")
    logger.info(f"AUC-PR on CV test data: {ap_eval}")

    logger.info(
        f'Classification report, CV test:\n{classification_report(y > THRESHOLD[likelihood], y_pred > THRESHOLD[likelihood], target_names=["CTRL", "NT1"])}'
    )
    logger.info(
        f"Confusion matrix, CV test:\n{confusion_matrix(y > THRESHOLD[likelihood], y_pred > THRESHOLD[likelihood])}"
    )

    # Evaluate on all training data
    logger.info("Running model evaluation on training data.")
    y_pred = cv_train_preds["mean"].squeeze(-1).mean(axis=1).T @ scale / scale.sum()
    plot_roc(
        y,
        y_pred,
        THRESHOLD[likelihood],
        "CV train data",
        savepath=os.path.join(figure_path, "roc_cv_train.png"),
    )
    auc_train = roc_auc_score(y, y_pred)
    ap_train = average_precision_score(y, y_pred)
    logger.info(f"AUC on CV train data: {auc_train}")
    logger.info(f"AUC-PR on CV train data: {ap_train}")
    # y_pred = (np.asarray(mean).squeeze(-1).T @ np.asarray(cv_acc)[:, np.newaxis]) / np.sum(cv_acc)
    # fpr, tpr, thr = roc_curve(y > 0.0, y_pred.squeeze())
    # plot_roc(fpr, tpr, thr, 0.0, "Model ensemble, train")
    # plt.savefig(
    # os.path.join(figure_path, f"roc_r{n_repeats:02}_train.png"), dpi=300, bbox_inches="tight",
    # )
    # print(classification_report(y > 0.0, y_pred > 0.0, target_names=['CTRL', 'NT1']))
    # print(confusion_matrix(y > 0.0, y_pred > 0.0))
    logger.info(
        f'Classification report, train:\n{classification_report(y > THRESHOLD[likelihood], y_pred > THRESHOLD[likelihood], target_names=["CTRL", "NT1"])}'
    )
    logger.info(
        f"Confusion matrix, train:\n{confusion_matrix(y > THRESHOLD[likelihood], y_pred > THRESHOLD[likelihood])}"
    )
    return 0
