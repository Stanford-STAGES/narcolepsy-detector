import itertools
import json
import logging
import os
import pickle
import sys
import time
from argparse import ArgumentParser

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# import wandb

from imblearn.over_sampling import SMOTE
from gpflow.ci_utils import ci_niter
from gpflow.monitor import ModelToTensorBoard, Monitor, MonitorTaskGroup, ScalarToTensorBoard
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import RobustScaler
from tqdm import trange, tqdm

# from inf_biomarker_config import ConfigBiomarker
# from inf_extract_features import ExtractFeatures
from utils import match_data, plot_roc

plt.switch_backend("agg")

# from xgboost_eval import print_report

# Configure logger to log to file
# logging.basicConfig(
# filename='model.log',
# format="%(asctime)s | %(message)s",
# level=logging.DEBUG,
# datefmt="%I:%M:%S"
# )
logger = logging.getLogger("GP TRAINING")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s  |  [%(levelname)-5.5s]  |  %(message)s", datefmt="%H:%M:%S")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


# log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
# logger = logging.getLogger('GP TRAINING')
# logger.setLevel(logging.INFO)
# file_handler = logging.FileHandler('model.log')
# file_handler.setLevel(logging.INFO)
# formatter = logging.Formatter("%(asctime)s | %(message)s")
# file_handler.setFormatter(log_formatter)
# logger.addHandler(file_handler)
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(log_formatter)
# logger.addHandler(console_handler)

kernels = dict(rbf=gpf.kernels.SquaredExponential, rq=gpf.kernels.RationalQuadratic,)

likelihoods = dict(bernoulli=gpf.likelihoods.Bernoulli(), gaussian=gpf.likelihoods.Gaussian(),)
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
    options=None,
    **kwargs,
):

    # cv_run = wandb.init(project="narcolepsy-detector")

    os.makedirs(save_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(save_dir, "model.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f'Usage: {" ".join([x for x in sys.argv])}\n')
    logger.info("Settings:")
    logger.info("---------")
    for idx, (k, v) in enumerate(sorted(options.items())):
        if idx == len(options) - 1:
            logger.info(f"{k:>15}\t{v}\n")
        else:
            logger.info(f"{k:>15}\t{v}")

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
        os.makedirs(model_path)
        os.makedirs(figure_path)
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
    df_master = pd.read_csv("data_master.csv", index_col=0)
    if "filtered" in feature_set:
        try:
            df = pd.read_csv(os.path.join(data_dir, f"{model_base}_long_train_unscaled.csv"), index_col=0)
        except FileNotFoundError:
            df = pd.read_csv(os.path.join(data_dir, f"{os.path.basename(data_dir)}_unscaled.csv"), index_col=0)
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
        X = df.iloc[:, 3:].values
        N, M = X.shape
        selected_features = list(df.columns[3:].values)

        # # Normalize data
        # X = (X - np.nanpercentile(X, 50, axis=0)) / (np.nanpercentile(X, 85, axis=0) - np.nanpercentile(X, 15, axis=0) + 1e-5)
    # elif os.path.isdir
    else:
        try:
            df = pd.read_csv(os.path.join(data_dir, f"{model_base}_long_r{resolution:02}_trainD_unscaled.csv"), index_col=0)
        except FileNotFoundError:
            try:
            df = pd.read_csv(os.path.join(data_dir, f"{model_base}_long_r{resolution:02}_unscaled.csv"), index_col=0)
            except FileNotFoundError:
                df = pd.read_csv(os.path.join(data_dir, f"r{resolution:02}_unscaled.csv"), index_col=0)
            train_idx = df.merge(df_master, left_on=["ID", "Cohort"], right_on=["ID", "Cohort"])["Narcolepsy training data"]
            df = df.loc[train_idx == 1]
        # df = pd.read_csv('data/narco_features/avg_kw21_r15_trainD.csv', index_col=0)
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
        # y = E['labels']
    N, M = X.shape
    y = (df["Label"].values)[:]
    y[y == 0] = -1
    shuf = np.random.permutation(len(y))
    X = X[shuf, :]
    y = y[shuf]

    # print('{} | Number of observations: {}'.format(datetime.now(), X.shape[0]))
    # print('{} | Number of features: {}'.format(datetime.now(), X.shape[1]))
    # print('{} | Number of narcoleptics: {}'.format(datetime.now(), (y == 1.0).sum()))
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
    models = [[[] for j in range(n_kfold)] for i in range(n_repeats)]
    scalers = [[[] for j in range(n_kfold)] for i in range(n_repeats)]
    pcas = (
        [[{"obj": None, "n_components": None, "components": None} for j in range(n_kfold)] for i in range(n_repeats)]
        if options["pca"]
        else None
    )
    cv_acc = []
    cv_auc = []
    mean = []
    var = []
    for (current_repeat, current_fold), (train, test) in zip(
        itertools.product(range(n_repeats), range(n_kfold)), skf.split(X, y)
    ):
        fold_count += 1
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
        scalers[current_repeat][current_fold] = RobustScaler(quantile_range=(15.0, 85.0)).fit(X_train)
        X_train = scalers[current_repeat][current_fold].transform(X_train)
        X_test = scalers[current_repeat][current_fold].transform(X_test)

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
                        logger.info(f"\tPCA running with excess components, adjusting n_components")
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
        logger.info("Running SMOTE")
        # X_res, y_res = SMOTE(k_neighbors=10, m_neighbors=20, n_jobs=-1).fit_sample(X_train, y_train)
        X_res, y_res = SMOTE(k_neighbors=10, n_jobs=-1).fit_resample(X_train, y_train)
        logger.info(f"\tShape of X: {X_res.shape}")

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
                        kernel=kernels[kernel](lengthscales=np.ones(M)),
                        likelihood=likelihoods[likelihood],
                        inducing_variable=inductionP,
                        num_data=X_res.shape[0],
                    )
                    # objective = m.training_loss_closure(data_minibatch_it)
                    objective = m.training_loss_closure(data)
                elif gp_model == "vgp":
                    m = gpf.models.VGP(
                        data, kernel=kernels[kernel](lengthscales=np.ones(M)), likelihood=likelihoods[likelihood],
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
                        new_jitter = 10 ** new_jitter_exponent
                        custom_config = gpf.config.Config(jitter=new_jitter)
                        logger.info(
                            f"\tCaptured an error, most likely in the Cholesky decomp. Raising jitter levels from {current_jitter} to {new_jitter} and retrying"
                        )
                except:
                    current_jitter = gpf.config.config().jitter
                    current_jitter_exponent = np.floor(np.log10(np.abs(current_jitter)))
                    new_jitter_exponent = current_jitter_exponent + 1
                    new_jitter = 10 ** new_jitter_exponent
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
        logger.info(f"\tELBO after optimization: {calculate_elbo().numpy()}")
        # print('{} | Elapsed time: {} | Negative log likelihood after optimization: {}'.format(datetime.now(), end - start, m.compute_log_likelihood()))
        # print('{} | Elapsed time: {} | Negative log likelihood after optimization: {}'.format(datetime.now(), end - start, m.log_marginal_likelihood()))

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
        models[current_repeat][current_fold] = m
        cv_eval_preds["mean"][current_repeat, test] = mean_pred
        cv_eval_preds["var"][current_repeat, test] = var_pred
        cv_eval_preds["pred_class"][current_repeat, test] = y_pred
        cv_eval_preds["true_class"][current_repeat, test] = y_test

        model_savepath = os.path.join(model_path, f"model_n{current_repeat + 1:02}_k{current_fold + 1:02}") + ".gpm"
        logger.info(f"\tSaving model {model_savepath}")
        m.predict_y_compiled = tf.function(m.predict_y, input_signature=[tf.TensorSpec(shape=[None, M], dtype=tf.float64)])
        tf.saved_model.save(m, model_savepath)

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
            mean_train, var_train = m.predict_y(
                pcas[current_repeat][current_fold]["obj"].transform(scalers[current_repeat][current_fold].transform(X))
            )
        else:
            mean_train, var_train = m.predict_y(scalers[current_repeat][current_fold].transform(X))
        cv_train_preds["mean"][current_repeat, current_fold, :] = mean_train
        cv_train_preds["var"][current_repeat, current_fold, :] = var_train
        mean.append(mean_train)
        var.append(var_train)

        logger.info(f"\tCross-validation accuracy: {np.mean(cv_acc)} +/- {np.std(cv_acc)}\n")
        logger.info(f"\tCross-validation AUC: {np.mean(cv_auc)} +/- {np.std(cv_auc)}\n")

        # del m, training_loss, optimizer

    # Save the CV train and test data predictions
    cv_data_savepath = os.path.join(save_dir, "cv_data.pkl")
    logger.info(f"Saving CV test and train info at {cv_data_savepath}")
    with open(cv_data_savepath, "wb") as fp:
        pickle.dump(cv_eval_preds, fp)

    # Save the feature scalers in directory
    feature_scale_savepath = os.path.join(save_dir, "feature_scales.pkl")
    logger.info(f"Saving feature scale values at {feature_scale_savepath}")
    with open(feature_scale_savepath, "wb") as pkl:
        pickle.dump(scalers, pkl)

    # Save the PCA objects in directory
    if options["pca"]:
        pca_savepath = os.path.join(save_dir, "pca_objects.pkl")
        logger.info(f"Saving PCA objects at {pca_savepath}")
        with open(pca_savepath, "wb") as pkl:
            pickle.dump(pcas, pkl)

    # elbo = np.asarray(elbo)
    # print(elbo)

    # def plot_elbo():
    #     plt.style.use("ggplot")
    #     plt.figure()
    #     plt.plot(np.arange(n_iter)[::log_frequency], elbo.reshape(-1, n_iter // log_frequency).T)
    #     plt.xlabel("Iteration")
    #     plt.ylabel("ELBO")
    #     plt.tight_layout()
    #     plt.savefig("tmp_elbo.png")

    # plot_elbo()

    # scale = {'mean': np.mean(cv_acc), 'std': np.std(cv_acc)}
    # scale_df = pd.DataFrame(scale, index=[0])
    scale_savepath = os.path.join(save_dir, "model_scales.csv")
    logger.info(f"Saving scale values at {scale_savepath}")
    scale_df = pd.DataFrame({"cv_auc": [roc_auc_score(y, x.squeeze()) for x in cv_eval_preds["mean"]]})
    scale_df.to_csv(scale_savepath)
    # scale_df = pd.DataFrame({"cv_acc": cv_acc, "cv_auc": cv_auc})
    # scale_df.to_csv(os.path.join(save_dir, 'model_scales.csv'))
    # with open(currentGpModelPath + '.gpscale', 'wb') as fp:
    #     pickle.dump(scale, fp)

    # Evaluate on eval data from CV folds
    logger.info("Running model evaluation on CV test data.")
    scale = scale_df["cv_auc"].values
    y_pred = (cv_eval_preds["mean"].T @ scale[:, np.newaxis]).squeeze()
    plot_roc(y, y_pred, THRESHOLD[likelihood], "CV eval data", savepath=os.path.join(figure_path, f"roc_cv_eval.png"))
    auc_eval = roc_auc_score(y, y_pred)
    logger.info(f"AUC on CV test data: {auc_eval}")

    logger.info(
        f'Classification report, CV test:\n{classification_report(y > THRESHOLD[likelihood], y_pred > THRESHOLD[likelihood], target_names=["CTRL", "NT1"])}'
    )
    logger.info(f"Confusion matrix, CV test:\n{confusion_matrix(y > THRESHOLD[likelihood], y_pred > THRESHOLD[likelihood])}")

    # Evaluate on all training data
    logger.info("Running model evaluation on training data.")
    y_pred = cv_train_preds["mean"].squeeze(-1).mean(axis=1).T @ scale / scale.sum()
    plot_roc(y, y_pred, THRESHOLD[likelihood], "CV train data", savepath=os.path.join(figure_path, "roc_cv_train.png"))
    auc_train = roc_auc_score(y, y_pred)
    logger.info(f"AUC on CV train data: {auc_train}")
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
    logger.info(f"Confusion matrix, train:\n{confusion_matrix(y > THRESHOLD[likelihood], y_pred > THRESHOLD[likelihood])}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-c", "--kernel", type=str, default="rbf")
    parser.add_argument("-d", "--data_dir", default="data/narco_features", type=str)
    parser.add_argument("-f", "--feature_set", default="all", type=str)
    parser.add_argument("-i", "--inducing", type=str, default="data", choices=["data", "kmeans"])
    parser.add_argument("-k", "--n_kfold", default=5, type=int)
    parser.add_argument("-l", "--likelihood", type=str, default="gaussian", choices=["bernoulli", "gaussian"])
    parser.add_argument("-m", "--model", required=True, type=str)
    parser.add_argument("-n", "--n_repeats", default=1, type=int)
    parser.add_argument("-p", "--n_inducing_points", default=350, type=int)
    parser.add_argument("-r", "--resolution", type=float)
    parser.add_argument("-s", "--save_dir", type=str)
    parser.add_argument("--gp_model", type=str, default="svgp")
    parser.add_argument("--log_frequency", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=10000)
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--pca_components", default=100)
    parser.add_argument("--pca_explained", default=90)
    args = parser.parse_args()

    # Set seed
    np.random.seed(42)

    # HACK: Parse the resolution argument
    args.resolution = int(args.resolution) if args.resolution % 1 == 0 else args.resolution

    # Parse experiment dir
    if not args.save_dir:
        if args.resolution:
            args.save_dir = os.path.join(
                "experiments",
                f"{args.model}_r{args.resolution:02}_{args.kernel}_{args.feature_set}_k{args.n_kfold:02}_n{args.n_repeats:02}_{args.inducing}",
            )
        else:
            args.save_dir = os.path.join(
                "experiments",
                f"{args.model}_{args.kernel}_{args.feature_set}_k{args.n_kfold:02}_n{args.n_repeats:02}_{args.inducing}",
            )

    train_single_model(**vars(args), options=vars(args))
    print("fin")
